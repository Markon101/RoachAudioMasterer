//! Explicit, bounded observations of the native master finalizer.
use crate::master::{self, MasterObservationMeta, MasterObserver, MasterResult};
use crate::native_audio::Audio;
use anyhow::{ensure, Context, Result};
use clap::Args;
use hound::{SampleFormat, WavSpec, WavWriter};
use serde_json::json;
#[cfg(test)]
use serde_json::Value;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};

const RATE: u32 = 48_000;
const MAX_FRAMES: usize = 48 * RATE as usize;
const MAX_RECORDS: usize = 32;
const MAX_SNAPSHOT_BYTES: u64 = 320 * 1024 * 1024;
const TRACE_FILE: &str = "observations.jsonl";

#[derive(Args, Clone, Debug, Default)]
pub struct MasterAuditArgs {
    /// Create bounded finalizer observations in this fresh directory.
    #[arg(long, value_name = "DIR")]
    pub stage_audit_dir: Option<PathBuf>,
    /// Also write each observed signal as native float32 WAV (bounded to 320 MiB).
    #[arg(long, requires = "stage_audit_dir")]
    pub stage_audit_wavs: bool,
}

impl MasterAuditArgs {
    /// Validate without creating or modifying the requested path.
    pub fn validate_request(&self) -> Result<()> {
        ensure!(
            !self.stage_audit_wavs || self.stage_audit_dir.is_some(),
            "--stage-audit-wavs requires --stage-audit-dir"
        );
        if let Some(dir) = &self.stage_audit_dir {
            ensure!(
                !dir.exists(),
                "stage audit directory must be fresh: {}",
                dir.display()
            );
            ensure!(
                dir.parent().is_some(),
                "stage audit directory must have a parent"
            );
        }
        Ok(())
    }

    /// Validate that audit files cannot collide with the run output. This is read-only.
    pub fn validate_for_output(&self, out: &Path) -> Result<()> {
        self.validate_request()?;
        let Some(audit) = &self.stage_audit_dir else {
            return Ok(());
        };
        let out = normalize_path(out)?;
        let audit = normalize_path(audit)?;
        ensure!(
            audit != out,
            "stage audit directory must differ from run output directory"
        );
        let parent = audit
            .parent()
            .context("stage audit directory must have parent")?;
        let parent_exists_dir = parent.is_dir();
        ensure!(
            parent_exists_dir || parent == out,
            "stage audit parent must exist as a directory or be the new run output directory"
        );
        Ok(())
    }

    pub fn validate_audio(audio: &Audio) -> Result<()> {
        ensure!(
            audio.rate == RATE,
            "stage audit requires native 48000 Hz audio"
        );
        ensure!(
            (1..=2).contains(&audio.channels.len()),
            "stage audit supports mono/stereo"
        );
        let frames = audio.channels.first().map_or(0, Vec::len);
        ensure!(
            (1..=MAX_FRAMES).contains(&frames),
            "stage audit audio must contain 1 sample..48 seconds"
        );
        ensure!(
            audio
                .channels
                .iter()
                .all(|c| c.len() == frames && c.iter().all(|x| x.is_finite())),
            "stage audit requires aligned finite channels"
        );
        Ok(())
    }

    /// `None` means disabled: no path creation, audio copies, or retained buffers.
    pub fn create(&self, audio: &Audio) -> Result<Option<FileMasterObserver>> {
        self.validate_request()?;
        let Some(dir) = &self.stage_audit_dir else {
            return Ok(None);
        };
        Self::validate_audio(audio)?;
        fs::create_dir(dir)
            .with_context(|| format!("creating fresh audit directory {}", dir.display()))?;
        let jsonl = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join(TRACE_FILE))?;
        Ok(Some(FileMasterObserver {
            dir: dir.clone(),
            jsonl: BufWriter::new(jsonl),
            records: 0,
            rate: audio.rate,
            channels: audio.channels.len(),
            frames: audio.frames(),
            wavs: self.stage_audit_wavs,
            snapshot_bytes: 0,
            iteration_count: 0,
            limiter_passes: 0,
            gain_only_bypasses: 0,
            last_stage: None,
            last_fingerprint: None,
            last_retained_iteration: None,
        }))
    }
}

fn normalize_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    // Resolve the deepest existing ancestor, then append missing path components.
    let mut ancestor = normalized.as_path();
    let mut suffix: Vec<OsString> = Vec::new();
    while !ancestor.exists() {
        let name = ancestor
            .file_name()
            .context("path has no existing ancestor")?;
        suffix.push(name.to_os_string());
        ancestor = ancestor.parent().context("path has no existing ancestor")?;
    }
    let mut resolved = fs::canonicalize(ancestor).unwrap_or_else(|_| ancestor.to_path_buf());
    for part in suffix.iter().rev() {
        resolved.push(part);
    }
    return Ok(resolved);
}

pub struct FileMasterObserver {
    dir: PathBuf,
    jsonl: BufWriter<File>,
    records: usize,
    rate: u32,
    channels: usize,
    frames: usize,
    wavs: bool,
    snapshot_bytes: u64,
    iteration_count: usize,
    limiter_passes: usize,
    gain_only_bypasses: usize,
    last_stage: Option<master::MasterAuditStage>,
    last_fingerprint: Option<u64>,
    last_retained_iteration: Option<usize>,
}

impl FileMasterObserver {
    fn halt_if_requested(&mut self) -> Result<()> {
        let halt = self.dir.join("HALT");
        if !halt.exists() {
            return Ok(());
        }
        let marker = self.dir.join("HALTED.json");
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(marker)?;
        serde_json::to_writer(
            &mut f,
            &json!({"schema":"master-audit-halted-v1",
            "reason":"manual HALT marker observed; current render aborted; evidence was preserved",
            "records_written":self.records}),
        )?;
        f.write_all(b"\n")?;
        f.sync_all()?;
        anyhow::bail!("master audit halted by manual HALT marker; partial evidence preserved")
    }

    fn snapshot(
        &mut self,
        seq: usize,
        meta: &MasterObservationMeta,
        audio: &Audio,
    ) -> Result<Option<(String, u64)>> {
        if !self.wavs {
            return Ok(None);
        }
        let expected = (audio.frames() as u64)
            .checked_mul(audio.channels.len() as u64)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(128))
            .context("snapshot byte count overflow")?;
        ensure!(
            self.snapshot_bytes.saturating_add(expected) <= MAX_SNAPSHOT_BYTES,
            "stage audit WAV snapshot budget exceeded (320 MiB)"
        );
        let stage_value = serde_json::to_value(meta.stage)?;
        let stage = stage_value.as_str().context("stage must serialize as a name")?;
        let name = format!("{seq:02}_{stage}.wav");
        let path = self.dir.join(&name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let spec = WavSpec {
            channels: audio.channels.len() as u16,
            sample_rate: audio.rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::new(file, spec)?;
        for i in 0..audio.frames() {
            for c in &audio.channels {
                writer.write_sample(c[i])?;
            }
        }
        writer.finalize()?;
        let bytes = fs::metadata(&path)?.len();
        self.snapshot_bytes += bytes;
        Ok(Some((name, bytes)))
    }

    /// Finish only after successful finalization. JSONL and snapshots remain if this fails.
    pub fn finish(mut self, result: &MasterResult) -> Result<()> {
        self.halt_if_requested()?;
        ensure!(
            self.last_stage == Some(master::MasterAuditStage::FinalOutput),
            "cannot complete stage audit without a final-output observation"
        );
        MasterAuditArgs::validate_audio(&result.mastered)?;
        ensure!(
            result.mastered.rate == self.rate
                && result.mastered.channels.len() == self.channels
                && result.mastered.frames() == self.frames,
            "final result geometry does not match the observed render"
        );
        let result_fingerprint = fnv1a_planar(&result.mastered);
        ensure!(
            self.last_fingerprint == Some(result_fingerprint),
            "final-output observation does not match returned mastered audio"
        );
        self.jsonl.flush()?;
        self.jsonl.get_ref().sync_all()?;
        let path = self.dir.join("master_audit.json");
        let mut out = OpenOptions::new().write(true).create_new(true).open(path)?;
        serde_json::to_writer_pretty(
            &mut out,
            &json!({
                "schema":"master-stage-audit-v1", "complete":true, "rate":self.rate,
                "channels":self.channels, "frames":self.frames,
                "record_count":self.records, "iteration_count":self.iteration_count,
                "limiter_pass_count":self.limiter_passes,
                "gain_only_bypass_count":self.gain_only_bypasses,
                "last_retained_iteration":self.last_retained_iteration,
                "trace_file":TRACE_FILE,
                "final_planar_f32le_fnv1a64":format!("{result_fingerprint:016x}"),
                "fingerprint_definition":"FNV-1a 64 over planar channel order, each f32 little-endian; non-cryptographic integrity check only",
                "command_argv":std::env::args().collect::<Vec<_>>(),
                "command_argv_note":"descriptive runtime argv only; not authenticated provenance",
                "snapshot_wav_bytes":self.snapshot_bytes, "snapshots_enabled":self.wavs,
                "result": {"before_metrics":result.before_metrics,"after_metrics":result.after_metrics,
                    "glue_gain_reduction_db":result.glue_gr,"limiter_gain_reduction_db":result.limiter_gr,
                    "pre_gain_db":result.pre_gain_db}
            }),
        )?;
        out.write_all(b"\n")?;
        out.sync_all()?;
        Ok(())
    }
}

impl MasterObserver for FileMasterObserver {
    fn observe(&mut self, meta: &MasterObservationMeta, audio: &Audio) -> Result<()> {
        self.halt_if_requested()?;
        ensure!(
            self.records < MAX_RECORDS,
            "stage audit record cap (32) exceeded"
        );
        ensure!(
            audio.rate == self.rate
                && audio.channels.len() == self.channels
                && audio.channels.iter().all(|c| c.len() == self.frames),
            "stage audit geometry changed during finalization"
        );
        ensure!(
            audio
                .channels
                .iter()
                .all(|c| c.iter().all(|x| x.is_finite())),
            "stage audit refuses nonfinite audio"
        );
        for value in [
            meta.base_drive_db,
            meta.effective_ceiling_linear,
            meta.gain_reduction_db,
            meta.final_trim_gain_linear,
        ]
        .into_iter()
        .flatten()
        {
            ensure!(
                value.is_finite(),
                "stage audit metadata contains nonfinite value"
            );
        }
        let snap = self.snapshot(self.records, meta, audio)?;
        let record = json!({"sequence":self.records, "meta":meta, "rate":audio.rate,
            "channels":audio.channels.len(), "frames":audio.frames(),
            "metrics":master::metrics(audio), "planar_f32le_fnv1a64":format!("{:016x}", fnv1a_planar(audio)),
            "snapshot":snap.map(|(file,bytes)|json!({"file":file,"bytes":bytes}))});
        serde_json::to_writer(&mut self.jsonl, &record)?;
        self.jsonl.write_all(b"\n")?;
        self.jsonl.flush()?;
        match meta.stage {
            master::MasterAuditStage::PreLimiter => {
                self.iteration_count += 1;
                self.limiter_passes += 1;
                self.last_retained_iteration = meta.iteration;
            }
            master::MasterAuditStage::GainOnlyInput => {
                self.iteration_count += 1;
                self.gain_only_bypasses += 1;
                self.last_retained_iteration = meta.iteration;
            }
            _ => {}
        }
        self.last_stage = Some(meta.stage);
        self.last_fingerprint = Some(fnv1a_planar(audio));
        self.records += 1;
        Ok(())
    }
}

fn fnv1a_planar(audio: &Audio) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for channel in &audio.channels {
        for x in channel {
            for byte in x.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::master::{MasterAuditStage, MasterObservationMeta};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn temp() -> PathBuf {
        std::env::temp_dir().join(format!(
            "roach-master-audit-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }
    fn audio() -> Audio {
        Audio {
            rate: RATE,
            channels: vec![vec![0.25, -0.5, 1.0, -1.0], vec![0.0, 0.5, -0.25, 0.75]],
        }
    }
    fn meta(stage: MasterAuditStage) -> MasterObservationMeta {
        MasterObservationMeta {
            stage,
            iteration: None,
            limiter: None,
            base_drive_db: None,
            quiet_makeup_curve_enabled: false,
            effective_ceiling_linear: None,
            gain_reduction_db: None,
            final_trim_gain_linear: None,
        }
    }
    fn result(a: Audio) -> MasterResult {
        MasterResult {
            before_metrics: master::metrics(&a),
            after_metrics: master::metrics(&a),
            mastered: a,
            glue_gr: 0.0,
            limiter_gr: 0.0,
            pre_gain_db: 0.0,
            onset_audit: crate::assess::scan_onset_diagnostics(&audio()),
        }
    }

    #[test]
    fn disabled_is_noop_and_fresh_directory_is_required() -> Result<()> {
        let a = audio();
        let args = MasterAuditArgs::default();
        assert!(args.create(&a)?.is_none());
        let path = temp();
        fs::create_dir(&path)?;
        let args = MasterAuditArgs {
            stage_audit_dir: Some(path.clone()),
            stage_audit_wavs: false,
        };
        assert!(args.validate_request().is_err());
        fs::remove_dir_all(path)?;
        Ok(())
    }

    #[test]
    fn output_path_validation_is_read_only_and_accepts_output_child() -> Result<()> {
        let base = temp();
        fs::create_dir(&base)?;
        let out = base.join("run");
        let same = MasterAuditArgs {
            stage_audit_dir: Some(out.join("..").join("run")),
            stage_audit_wavs: false,
        };
        assert!(same.validate_for_output(&out).is_err());
        assert!(!out.exists());

        let child = MasterAuditArgs {
            stage_audit_dir: Some(out.join("audit")),
            stage_audit_wavs: false,
        };
        child.validate_for_output(&out)?;
        assert!(!out.exists());

        let invalid = MasterAuditArgs {
            stage_audit_dir: Some(base.join("missing-parent").join("audit")),
            stage_audit_wavs: false,
        };
        assert!(invalid.validate_for_output(&out).is_err());
        assert!(!base.join("missing-parent").exists());
        fs::remove_dir_all(base)?;
        Ok(())
    }

    #[test]
    fn observations_write_fingerprinted_native_float_wav_and_summary() -> Result<()> {
        let path = temp();
        let a = audio();
        let args = MasterAuditArgs {
            stage_audit_dir: Some(path.clone()),
            stage_audit_wavs: true,
        };
        let mut obs = args.create(&a)?.unwrap();
        obs.observe(&meta(MasterAuditStage::PreLimiter), &a)?;
        let post = meta(MasterAuditStage::PostLimiter);
        obs.observe(&post, &a)?;
        obs.observe(&meta(MasterAuditStage::FinalOutput), &a)?;
        obs.finish(&result(a.clone()))?;
        let rows = fs::read_to_string(path.join("observations.jsonl"))?;
        let v: Value = serde_json::from_str(rows.lines().next().unwrap())?;
        assert_eq!(
            v["planar_f32le_fnv1a64"],
            format!("{:016x}", fnv1a_planar(&a))
        );
        let wav =
            crate::native_audio::read_entire(&path.join(v["snapshot"]["file"].as_str().unwrap()))?;
        assert_eq!(wav.rate, RATE);
        assert_eq!(wav.channels.len(), 2);
        for (x, y) in wav
            .channels
            .iter()
            .flatten()
            .zip(a.channels.iter().flatten())
        {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        let summary: Value = serde_json::from_slice(&fs::read(path.join("master_audit.json"))?)?;
        assert_eq!(summary["complete"], true);
        assert_eq!(summary["record_count"], 3);
        assert_eq!(summary["iteration_count"], 1);
        assert_eq!(summary["limiter_pass_count"], 1);
        assert_eq!(summary["gain_only_bypass_count"], 0);
        fs::remove_dir_all(path)?;
        Ok(())
    }

    #[test]
    fn invalid_geometry_and_halt_marker_fail_without_erasing_evidence() -> Result<()> {
        let mut bad = audio();
        bad.rate = 44100;
        assert!(MasterAuditArgs::validate_audio(&bad).is_err());
        let path = temp();
        let a = audio();
        let args = MasterAuditArgs {
            stage_audit_dir: Some(path.clone()),
            stage_audit_wavs: false,
        };
        let mut obs = args.create(&a)?.unwrap();
        let mut bad_geometry = a.clone();
        bad_geometry.channels[1].pop();
        assert!(obs
            .observe(&meta(MasterAuditStage::Input), &bad_geometry)
            .is_err());
        File::create(path.join("HALT"))?.write_all(b"stop")?;
        assert!(obs.observe(&meta(MasterAuditStage::Input), &a).is_err());
        assert!(path.join("HALTED.json").exists());
        assert!(path.join("observations.jsonl").exists());
        fs::remove_dir_all(path)?;
        Ok(())
    }

    #[test]
    fn gain_only_is_bypass_and_incomplete_trace_cannot_complete() -> Result<()> {
        let path = temp();
        let a = audio();
        let args = MasterAuditArgs {
            stage_audit_dir: Some(path.clone()),
            stage_audit_wavs: false,
        };
        let mut obs = args.create(&a)?.unwrap();
        let mut bypass = meta(MasterAuditStage::GainOnlyInput);
        bypass.iteration = Some(2);
        obs.observe(&bypass, &a)?;
        obs.observe(&meta(MasterAuditStage::GainOnlyOutput), &a)?;
        assert!(obs.finish(&result(a)).is_err());
        assert!(!path.join("master_audit.json").exists());
        fs::remove_dir_all(path)?;

        let path = temp();
        let a = audio();
        let args = MasterAuditArgs {
            stage_audit_dir: Some(path.clone()),
            stage_audit_wavs: false,
        };
        let mut obs = args.create(&a)?.unwrap();
        let mut bypass = meta(MasterAuditStage::GainOnlyInput);
        bypass.iteration = Some(0);
        obs.observe(&bypass, &a)?;
        obs.observe(&meta(MasterAuditStage::GainOnlyOutput), &a)?;
        obs.observe(&meta(MasterAuditStage::FinalOutput), &a)?;
        obs.finish(&result(a))?;
        let summary: Value = serde_json::from_slice(&fs::read(path.join("master_audit.json"))?)?;
        assert_eq!(summary["iteration_count"], 1);
        assert_eq!(summary["limiter_pass_count"], 0);
        assert_eq!(summary["gain_only_bypass_count"], 1);
        fs::remove_dir_all(path)?;
        Ok(())
    }
}
