//! Export exact frozen Mid procedural targets and their damaged inputs.
//!
//! This is a source-only research utility: it calls the existing generator and
//! degradation implementation, loads no model, and performs no inference.

#[path = "../src/backend.rs"]
#[allow(dead_code)]
mod backend;
#[path = "../src/dsp.rs"]
#[allow(dead_code)]
mod dsp;
#[path = "../src/model.rs"]
#[allow(dead_code)]
mod model;
#[path = "../src/native_audio.rs"]
#[allow(dead_code)]
mod native_audio;
#[path = "../src/native_dsp.rs"]
#[allow(dead_code)]
mod native_dsp;
#[cfg(feature = "opencl")]
#[path = "../src/opencl.rs"]
#[allow(dead_code)]
mod opencl;
#[path = "../src/rich_synth.rs"]
#[allow(dead_code)]
mod rich_synth;
#[path = "../src/scene.rs"]
#[allow(dead_code)]
mod scene;
#[path = "../src/scene_features.rs"]
#[allow(dead_code)]
mod scene_features;
#[path = "../src/scene_model.rs"]
#[allow(dead_code)]
mod scene_model;
#[path = "../src/synth.rs"]
#[allow(dead_code)]
mod synth;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use native_audio::Audio;
use serde::Serialize;
use serde_json::json;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

const DEVELOPMENT_SEED_START: u64 = 860_000;
const DEVELOPMENT_SEED_END_EXCLUSIVE: u64 = 870_000;
const MAX_SEEDS: usize = 8;
const SAMPLE_RATE: u32 = 48_000;
const EXPECTED_SAMPLES: usize = 16_384;

#[derive(Parser)]
#[command(about = "Export exact Mid procedural target/input controls without inference")]
struct Args {
    /// One to eight development seeds in 860000..870000; final seeds are refused.
    #[arg(long, required = true, num_args = 1..=8)]
    seeds: Vec<u64>,
    /// Fresh output directory; existing paths are never overwritten.
    #[arg(long, required = true)]
    out: PathBuf,
}

#[derive(Serialize)]
struct CaseRecipe {
    generator_version: u32,
    seed: u64,
    family: String,
    no_upper: bool,
    no_lower: bool,
    mono: bool,
    gap: Option<[f32; 2]>,
    damage: scene_features::Damage,
}

fn validate_seeds(seeds: &[u64]) -> Result<()> {
    ensure!(
        !seeds.is_empty() && seeds.len() <= MAX_SEEDS,
        "select one to eight development seeds"
    );
    let mut seen = std::collections::BTreeSet::new();
    for &seed in seeds {
        ensure!(
            (DEVELOPMENT_SEED_START..DEVELOPMENT_SEED_END_EXCLUSIVE).contains(&seed),
            "seed {seed} is outside the authorized development range 860000..869999"
        );
        ensure!(seen.insert(seed), "duplicate development seed {seed}");
    }
    Ok(())
}

fn sha256(path: &Path) -> Result<String> {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .with_context(|| format!("sha256sum failed to start for {}", path.display()))?;
    ensure!(
        output.status.success(),
        "sha256sum failed for {}",
        path.display()
    );
    let text = String::from_utf8(output.stdout)?;
    let hash = text.split_whitespace().next().unwrap_or_default();
    ensure!(
        hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "sha256sum returned an invalid digest for {}",
        path.display()
    );
    Ok(hash.to_ascii_lowercase())
}

fn git_value(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("refusing to overwrite {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn planar_f32le(audio: &Audio) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(audio.channels.len() * audio.frames() * 4);
    for sample in audio.channels.iter().flatten() {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn case_recipe(seed: u64, recipe: &rich_synth::Recipe) -> CaseRecipe {
    CaseRecipe {
        generator_version: recipe.version,
        seed,
        family: recipe.family.clone(),
        no_upper: recipe.no_upper,
        no_lower: recipe.no_lower,
        mono: recipe.mono,
        gap: recipe.gap,
        damage: recipe.damage,
    }
}

fn export_case(out: &Path, seed: u64) -> Result<serde_json::Value> {
    let (target, recipe) = rich_synth::generate_mid_multitrack(seed);
    ensure!(
        target.rate == SAMPLE_RATE
            && target.channels.len() == 2
            && target.frames() == EXPECTED_SAMPLES,
        "frozen Mid procedural generator geometry changed for seed {seed}"
    );
    ensure!(recipe.seed == seed, "generator returned a different seed");
    let damaged = recipe.damage.apply(&target);
    ensure!(
        damaged.rate == target.rate
            && damaged.channels.len() == target.channels.len()
            && damaged.frames() == target.frames(),
        "damage application changed source geometry"
    );
    ensure!(
        target.channels.iter().flatten().all(|x| x.is_finite())
            && damaged.channels.iter().flatten().all(|x| x.is_finite()),
        "generator or damage produced nonfinite samples"
    );

    let case_dir = out.join(format!("seed-{seed}"));
    fs::create_dir(&case_dir)?;
    let mut files = serde_json::Map::new();
    for (name, audio) in [("target", &target), ("degraded_input", &damaged)] {
        let wav = case_dir.join(format!("{name}.wav"));
        native_audio::write(&wav, audio, false)?;
        let raw = case_dir.join(format!("{name}.planar-f32le"));
        write_new(&raw, &planar_f32le(audio))?;
        files.insert(name.to_owned(), json!({
            "wav": {"path": wav.strip_prefix(out)?.to_string_lossy(), "sha256": sha256(&wav)?},
            "planar_f32le": {"path": raw.strip_prefix(out)?.to_string_lossy(), "sha256": sha256(&raw)?},
        }));
    }
    let recipe_file = case_dir.join("recipe.json");
    write_new(
        &recipe_file,
        &serde_json::to_vec_pretty(&case_recipe(seed, &recipe))?,
    )?;
    Ok(json!({
        "seed": seed,
        "family_index": seed % 12,
        "family": recipe.family,
        "samples_per_channel": target.frames(),
        "sample_rate_hz": target.rate,
        "duration_seconds": target.frames() as f64 / target.rate as f64,
        "recipe_path": recipe_file.strip_prefix(out)?.to_string_lossy(),
        "recipe_sha256": sha256(&recipe_file)?,
        "files": files,
    }))
}

fn export(seeds: &[u64], out: &Path) -> Result<serde_json::Value> {
    validate_seeds(seeds)?;
    ensure!(
        !out.exists(),
        "output path already exists; choose a fresh directory"
    );
    fs::create_dir(out)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let tracked_sources = [
        "examples/export_mid_controls.rs",
        "src/rich_synth.rs",
        "src/scene_features.rs",
        "src/dsp.rs",
        "src/native_dsp.rs",
        "src/native_audio.rs",
        "src/scene.rs",
        "src/synth.rs",
        "src/model.rs",
        "src/backend.rs",
        "src/scene_model.rs",
    ];
    let source_hashes: serde_json::Map<String, serde_json::Value> = tracked_sources
        .iter()
        .map(|relative| Ok((relative.to_string(), json!(sha256(&root.join(relative))?))))
        .collect::<Result<_>>()?;
    let executable = std::env::current_exe()?;
    let cases = seeds
        .iter()
        .map(|&seed| export_case(out, seed))
        .collect::<Result<Vec<_>>>()?;
    let manifest = json!({
        "schema": "mid-procedural-source-bundle-v1",
        "scope": "exact deterministic source export only; no Mid model load, inference, correction, or normalization",
        "generator": "existing rich_synth::generate_mid_multitrack(seed) followed by that returned recipe.damage.apply(target)",
        "source_decomposition": "not exported: the generator returns only the final composite target and recipe; this bundle must not be described as a component-stem decomposition",
        "long_coherence_limit": "16384 samples at 48000 Hz is about 0.3413 seconds and cannot establish long coherence; samples are not tiled",
        "seed_policy": {"minimum_inclusive": DEVELOPMENT_SEED_START,
                        "maximum_exclusive": DEVELOPMENT_SEED_END_EXCLUSIVE,
                        "maximum_cases": MAX_SEEDS,
                        "final_seeds_rejected": true},
        "geometry": {"sample_rate_hz": SAMPLE_RATE, "channels": 2,
                     "samples_per_channel": EXPECTED_SAMPLES,
                     "raw_layout": "channel-major planar f32 little-endian, L then R"},
        "provenance": {
            "git_head": git_value(root, &["rev-parse", "HEAD"]),
            "git_status_porcelain": git_value(root, &["status", "--porcelain"]),
            "executable_path": executable.display().to_string(),
            "executable_sha256": sha256(&executable)?,
            "source_sha256": source_hashes,
        },
        "cases": cases,
    });
    let manifest_path = out.join("manifest.json");
    write_new(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    Ok(json!({"out":out.display().to_string(),
              "manifest":manifest_path.display().to_string(),
              "manifest_sha256":sha256(&manifest_path)?,
              "cases":seeds.len(),"samples_per_channel":EXPECTED_SAMPLES,
              "duration_seconds":EXPECTED_SAMPLES as f64 / SAMPLE_RATE as f64}))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let receipt = export(&args.seeds, &args.out)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fresh_temp(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "export-mid-controls-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn seed_scope_rejects_duplicates_final_and_oversized_sets() {
        assert!(validate_seeds(&[860003, 860007]).is_ok());
        assert!(validate_seeds(&[860003, 860003]).is_err());
        assert!(validate_seeds(&[870000]).is_err());
        assert!(validate_seeds(&[859999]).is_err());
        assert!(validate_seeds(&(860000..860009).collect::<Vec<_>>()).is_err());
    }

    #[test]
    fn existing_generator_and_damage_are_seed_deterministic() {
        let (target_a, recipe_a) = rich_synth::generate_mid_multitrack(860003);
        let (target_b, recipe_b) = rich_synth::generate_mid_multitrack(860003);
        let damaged_a = recipe_a.damage.apply(&target_a);
        let damaged_b = recipe_b.damage.apply(&target_b);
        assert_eq!(target_a.channels, target_b.channels);
        assert_eq!(damaged_a.channels, damaged_b.channels);
        assert_eq!(recipe_a.family, "stopped_mid_abstention");
        assert_eq!(target_a.frames(), EXPECTED_SAMPLES);
        assert_eq!(target_a.rate, SAMPLE_RATE);
    }

    #[test]
    fn exported_float_wav_and_planar_samples_roundtrip_and_output_is_fresh() -> Result<()> {
        let out = fresh_temp("roundtrip");
        let receipt = export(&[860007], &out)?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("manifest.json"))?)?;
        assert_eq!(manifest["cases"][0]["family"], "vocal_with_aspiration");
        assert_eq!(
            manifest["cases"][0]["samples_per_channel"],
            EXPECTED_SAMPLES
        );
        assert_eq!(receipt["cases"], 1);
        let target = native_audio::read_entire(&out.join("seed-860007/target.wav"))?;
        let damaged = native_audio::read_entire(&out.join("seed-860007/degraded_input.wav"))?;
        let (expected_target, recipe) = rich_synth::generate_mid_multitrack(860007);
        let expected_damaged = recipe.damage.apply(&expected_target);
        assert_eq!(target.channels, expected_target.channels);
        assert_eq!(damaged.channels, expected_damaged.channels);
        assert!(export(&[860007], &out).is_err());
        fs::remove_dir_all(out)?;
        Ok(())
    }
}
