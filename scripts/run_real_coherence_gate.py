#!/usr/bin/env python3
"""Run the pinned target-free real-song Mid coherence eligibility gate.

The plan is JSON with objects `candidate_binary`, `baseline_binary`, `model`,
`source_wav`, and `protocol` (each has `path` and `sha256`), a full
`baseline_commit`, and `source_pins` for exactly src/main.rs, src/rich_mid.rs,
src/scene_features.rs, and src/synth.rs. The protocol is the fixed
artifacts/listening-coherence-20261020/eligibility_protocol.json. This runner
uses no targets or gain normalization. It runs 11 bounded native children;
the human listening decision remains separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "artifacts/listening-coherence-20261020/eligibility_protocol.json"
RATE = 48_000
BASELINE_SOURCE_COMMIT = "07e9ac90fee6bae65f84e1aab80cf2aaf37f22b3"
BASELINE_PATH = ROOT / "runs/stage-accountability-20261020/master-observer-candidate-v1"
BASELINE_SHA256 = "857e065fba22b9d3676c0ff318558e3faaa64e938451988dd804bd7b970b8437"
SOURCE_NAMES = ("src/main.rs", "src/rich_mid.rs", "src/scene_features.rs", "src/synth.rs")
SOURCE_SHA256 = "1ee053d29036ecba69ddbf0d7cf49483950c717c946a1ec45fd4cb34fc83b559"
MODEL_SHA256 = "e2ca85b64521ca62d72278b33177b426d4386bc0add0ff2d94526f76a5d09ae3"
MEMORY_LIMIT_KIB = 768 * 1024
TIMEOUT_SECONDS = 180.0
POLL_SECONDS = 0.2

sys.path.insert(0, str(ROOT / "scripts"))
from long_coherence_fixture import _write_float_wav  # noqa: E402
from roach_ears.audio_io import read_wav  # noqa: E402
from run_mid_profile import memory  # noqa: E402


class GateStop(RuntimeError):
    """Execution/provenance stop. A stop is latched and partial evidence kept."""


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _valid_sha(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _fresh_root(path: Path) -> None:
    if not path.is_absolute() or path == Path("/"):
        raise ValueError("--out must be an absolute non-root path")
    if path.exists():
        if not path.is_dir() or {p.name for p in path.iterdir()} - {"HALT"}:
            raise FileExistsError(f"output root must be fresh: {path}")


def _obj_pin(plan: dict, key: str) -> tuple[Path, str]:
    item = plan.get(key)
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError(f"plan.{key} must contain path and sha256")
    path = Path(item["path"])
    digest = _valid_sha(item.get("sha256"), f"{key}.sha256")
    if not path.is_file() or sha256(path) != digest:
        raise ValueError(f"{key} missing or SHA-256 mismatch: {path}")
    return path.resolve(), digest


def _validate_protocol(protocol: dict) -> dict:
    if protocol.get("schema") != "real-coherence-listening-eligibility-protocol-v1":
        raise ValueError("unsupported listening eligibility protocol")
    source = protocol.get("source", {})
    cond = protocol.get("conditioning", {})
    views = protocol.get("views", {})
    gates = protocol.get("eligibility_gates", {})
    tail = protocol.get("tail_probe", protocol.get("tail", protocol.get("tail_control")))
    if (source.get("sha256") != SOURCE_SHA256 or source.get("duration_seconds") != 140.4
            or cond.get("model_sha256") != MODEL_SHA256):
        raise ValueError("protocol source/model SHA does not match the frozen eligibility pins")
    expected_cond = {"steps": 8, "mid_cutoff": 1500, "mid_ceiling": 8000,
                     "strength": 0.6, "chunk_seconds": 10, "overlap_seconds": 2,
                     "quiet_abstention": True, "post_denoise": False,
                     "post_auto_eq": False, "backend": "opencl"}
    for key, expected in expected_cond.items():
        if cond.get(key) != expected:
            raise ValueError(f"protocol conditioning.{key} must be {expected!r}")
    if protocol.get("arms") != ["restart", "continuous_with_final_trusted_band_lock"]:
        raise ValueError("protocol arm order changed")
    if (views.get("source_start_sample") != 2_304_000
            or views.get("full_frames") != 1_536_000
            or views.get("view_frames") != 1_248_000
            or views.get("view_starts_samples") != [0, 144_384]
            or views.get("comparison_global_samples") != [384_000, 1_008_000]):
        raise ValueError("protocol view geometry changed")
    if not isinstance(tail, dict):
        raise ValueError("protocol must freeze the unpaired tail control")
    tail_expected = {"source_start_sample": 5_760_000, "source_end_sample": 6_739_200,
                     "frames": 979_200, "local_chunk_starts_samples": [0, 384_000, 499_200],
                     "absolute_chunk_starts_samples": [5_760_000, 6_144_000, 6_259_200]}
    for key, value in tail_expected.items():
        actual = tail.get(key)
        if actual != value:
            raise ValueError(f"protocol tail.{key} must be {value!r}")
    if "quiet_mid_denoise=true" not in protocol.get("quiet_gate_requirement", "") or "force_specialists=false" not in protocol.get("quiet_gate_requirement", ""):
        raise ValueError("protocol must preserve quiet abstention and specialist routing abstention")
    expected_gates = {"layout_rms_reduction_min": 0.5,
                      "max_whole_view_protected_error_rms": 1e-6,
                      "residual_rms_growth_vs_restart_max": 0.1,
                      "backend_sample_max_abs": 2e-6,
                      "process_hwm_mib_cap_excerpt": 768}
    for key, expected in expected_gates.items():
        if gates.get(key) != expected:
            raise ValueError(f"protocol eligibility_gates.{key} must be {expected}")
    return {"conditioning": cond, "views": views, "tail": tail, "gates": gates}


def preflight(plan_path: Path, out: Path) -> tuple[dict, dict, list[dict]]:
    _fresh_root(out)
    plan_path = plan_path.resolve()
    if out.resolve() in plan_path.parents:
        raise ValueError("plan must live outside the fresh output directory")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "real-coherence-gate-plan-v1":
        raise ValueError("unsupported plan schema")
    if plan.get("baseline_source_commit") != BASELINE_SOURCE_COMMIT:
        raise ValueError("baseline source identity must be the observer-qualified 07e9ac90 build")
    candidate, candidate_sha = _obj_pin(plan, "candidate_binary")
    baseline, baseline_sha = _obj_pin(plan, "baseline_binary")
    if baseline != BASELINE_PATH.resolve():
        raise ValueError("baseline binary path must be the pinned master-observer-candidate-v1 artifact")
    model, model_sha = _obj_pin(plan, "model")
    source_wav, source_sha = _obj_pin(plan, "source_wav")
    protocol_obj = plan.get("protocol")
    if not isinstance(protocol_obj, dict) or not isinstance(protocol_obj.get("path"), str):
        raise ValueError("plan.protocol must contain path and sha256")
    protocol_path = Path(protocol_obj["path"]).resolve()
    if protocol_path != PROTOCOL_PATH.resolve():
        raise ValueError("protocol path is not the frozen listening protocol")
    protocol_sha = _valid_sha(protocol_obj.get("sha256"), "protocol.sha256")
    if sha256(protocol_path) != protocol_sha:
        raise ValueError("protocol SHA-256 mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fixed = _validate_protocol(protocol)
    if baseline_sha != BASELINE_SHA256:
        raise ValueError("baseline binary SHA does not match the pinned observer-qualified baseline")
    if model_sha != fixed["conditioning"]["model_sha256"] or source_sha != protocol["source"]["sha256"]:
        raise ValueError("model/source pin differs from the frozen protocol")
    expected_source = Path(protocol["source"]["path"]).resolve()
    if source_wav != expected_source or source_sha != SOURCE_SHA256:
        raise ValueError("source path/hash does not match the frozen Chasing Horizons input")
    for key, path in (("candidate_binary", candidate), ("baseline_binary", baseline)):
        if not os.access(path, os.X_OK):
            raise ValueError(f"{key} is not executable: {path}")

    pins = plan.get("source_pins")
    if not isinstance(pins, list) or len(pins) != len(SOURCE_NAMES):
        raise ValueError("source_pins must pin the four coherence Rust source files")
    by_name = {}
    for pin in pins:
        if not isinstance(pin, dict) or pin.get("name") not in SOURCE_NAMES or pin["name"] in by_name:
            raise ValueError("invalid or duplicate source pin")
        expected_path = (ROOT / pin["name"]).resolve()
        path = Path(pin.get("path", "")).resolve()
        digest = _valid_sha(pin.get("sha256"), pin["name"])
        if path != expected_path or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"source pin mismatch: {pin['name']}")
        by_name[pin["name"]] = {"name": pin["name"], "path": str(path), "sha256": digest}
    if set(by_name) != set(SOURCE_NAMES):
        raise ValueError("source_pins must contain all four current Rust files")

    script_path = Path(__file__).resolve()
    all_pins = [
        {"name": "candidate_binary", "path": str(candidate), "sha256": candidate_sha},
        {"name": "baseline_binary_857e065f", "path": str(baseline), "sha256": baseline_sha},
        {"name": "model", "path": str(model), "sha256": model_sha},
        {"name": "source_wav", "path": str(source_wav), "sha256": source_sha},
        {"name": "protocol", "path": str(protocol_path), "sha256": protocol_sha},
        *[by_name[name] for name in SOURCE_NAMES],
        {"name": "gate_script", "path": str(script_path), "sha256": sha256(script_path)},
        {"name": "plan", "path": str(plan_path), "sha256": sha256(plan_path)},
        *[{"name": "python_dependency_" + name, "path": str(ROOT / "scripts" / name),
           "sha256": sha256(ROOT / "scripts" / name)} for name in
          ("long_coherence_fixture.py", "roach_ears/audio_io.py", "run_mid_profile.py")],
    ]
    return plan, protocol, all_pins


def _read_stereo_float(path: Path, frames: int) -> np.ndarray:
    audio = read_wav(path)
    if audio.sample_rate != RATE or audio.channels != 2 or audio.num_samples != frames:
        raise ValueError(f"wrong WAV geometry at {path}: {audio.sample_rate}, {audio.samples.shape}")
    samples = np.asarray(audio.samples, dtype=np.float32)
    if not np.isfinite(samples).all():
        raise ValueError(f"nonfinite float samples at {path}")
    return samples


def _write_derived(root: Path, name: str, samples: np.ndarray, origin: int) -> dict:
    if samples.ndim != 2 or samples.shape[0] != 2 or samples.shape[1] == 0:
        raise ValueError(f"invalid derived stereo input {name}: {samples.shape}")
    if not np.isfinite(samples).all():
        raise ValueError(f"nonfinite derived input {name}")
    path = root / "inputs" / f"{name}.wav"
    if path.exists():
        raise FileExistsError(path)
    _write_float_wav(path, np.asarray(samples.T, dtype=np.float32))
    reread = _read_stereo_float(path, samples.shape[1])
    if not np.array_equal(reread.view(np.uint32), samples.view(np.uint32)):
        raise ValueError(f"derived WAV changed float32 samples: {name}")
    return {"name": name, "path": str(path), "sha256": sha256(path),
            "sample_rate_hz": RATE, "channels": 2, "frames": samples.shape[1],
            "source_origin_sample": origin}


def derive_inputs(source_wav: Path, protocol: dict, root: Path) -> tuple[dict, dict[str, dict]]:
    source = read_wav(source_wav)
    if source.sample_rate != RATE or source.channels != 2:
        raise ValueError("source WAV must be 48 kHz stereo")
    if source.num_samples != round(float(protocol["source"]["duration_seconds"]) * RATE):
        raise ValueError("source WAV duration does not match frozen protocol")
    if not np.isfinite(source.samples).all():
        raise ValueError("source WAV contains nonfinite samples")
    views = protocol["views"]
    source_start = views["source_start_sample"]
    full_end = source_start + views["full_frames"]
    if full_end > source.num_samples:
        raise ValueError("selected 48–80 second range exceeds source WAV")
    full_start = source.samples[:, source_start:full_end]
    artifacts = {}
    rows = []
    for index, start in enumerate(views["view_starts_samples"]):
        frames = views["view_frames"]
        view = full_start[:, start:start + frames]
        if view.shape[1] != frames:
            raise ValueError("frozen view exceeds selected source interval")
        row = _write_derived(root, f"view{index}", view,
                             views["source_start_sample"] + start)
        artifacts[row["name"]] = row
        rows.append(row)
    tail = protocol.get("tail_probe", protocol.get("tail", protocol.get("tail_control")))
    tail_start, tail_frames = tail["source_start_sample"], tail["frames"]
    tail_samples = source.samples[:, tail_start:tail_start + tail_frames]
    if tail_samples.shape[1] != tail_frames:
        raise ValueError("frozen tail interval exceeds source WAV")
    row = _write_derived(root, "tail120to140_4", tail_samples, tail_start)
    artifacts[row["name"]] = row
    rows.append(row)
    probe = full_start[:, :RATE]
    row = _write_derived(root, "source_probe_1s", probe, source_start)
    artifacts[row["name"]] = row
    rows.append(row)
    silence_path = root / "inputs" / "true_silence_16384.wav"
    silence = np.zeros((2, 16_384), dtype=np.float32)
    row = _write_derived(root, "true_silence_16384", silence, 0)
    artifacts[row["name"]] = row
    rows.append(row)
    geometry = {"schema": "real-coherence-derived-inputs-v1", "sample_rate_hz": RATE,
                "source_sha256": protocol["source"]["sha256"],
                "source_selection_start_sample": source_start,
                "source_selection_frames": views["full_frames"],
                "views": rows[:2], "tail": artifacts["tail120to140_4"],
                "probe": artifacts["source_probe_1s"],
                "silence": artifacts["true_silence_16384"]}
    packed = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
    geometry["derived_geometry_sha256"] = hashlib.sha256(packed).hexdigest()
    return geometry, artifacts


def _write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def _verify_pins(pins: list[dict], stage: str) -> None:
    for pin in pins:
        try:
            actual = sha256(pin["path"])
        except OSError as exc:
            raise GateStop(f"{stage}: pinned file missing: {pin['name']}: {exc}") from exc
        if actual != pin["sha256"]:
            raise GateStop(f"{stage}: pinned file changed: {pin['name']}")


def _command(binary: Path, model: Path, job: dict, out: Path,
             cond: dict, backend: str) -> list[str]:
    argv = [str(binary), "rich-mid-restore", "--model", str(model),
            "--input", str(job["input"]), "--backend", backend,
            "--mid-cutoff", str(cond["mid_cutoff"]), "--mid-ceiling", str(cond["mid_ceiling"]),
            "--steps", str(cond["steps"]), "--strength", str(cond["strength"]),
            "--chunk-seconds", str(cond["chunk_seconds"]),
            "--overlap-seconds", str(cond["overlap_seconds"])]
    if job.get("quiet_abstention", False):
        argv.append("--quiet-abstention")
    if job["arm"] == "continuous_with_final_trusted_band_lock":
        argv.extend(["--continuous-excitation", "--final-band-lock",
                     "--input-origin-samples", str(job["origin"])])
    return argv + ["--out", str(out)]


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_child(name: str, command: list[str], log_path: Path, root: Path,
               pins: list[dict], timeout: float = TIMEOUT_SECONDS,
               limit_kib: int = MEMORY_LIMIT_KIB) -> dict:
    if (root / "HALT").exists():
        raise GateStop("manual HALT marker was present before child launch")
    _verify_pins(pins, f"before {name}")
    if log_path.exists():
        raise FileExistsError(log_path)
    env = os.environ.copy()
    for key in ("ROACH_PROFILE_MID", "ROACH_FOREGROUND_CONFIRMED", "ROACH_MID_CACHE_MIB",
                "ROACH_MID_CACHE_BUILD", "ROACH_MID_BATCH_ROWS"):
        env.pop(key, None)
    if "--backend" in command and command[command.index("--backend") + 1] == "opencl":
        env["OCL_ICD_ASSUME_ICD_EXTENSION"] = "1"
    peak_rss = peak_hwm = 0
    observed = False
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while True:
                counters = memory(process.pid)
                if counters:
                    observed = True
                peak_rss = max(peak_rss, counters.get("VmRSS", 0))
                peak_hwm = max(peak_hwm, counters.get("VmHWM", 0))
                if max(peak_rss, peak_hwm) > limit_kib:
                    raise GateStop(f"{name}: child RSS/HWM exceeded {limit_kib // 1024} MiB")
                if (root / "HALT").exists():
                    raise GateStop("manual HALT marker appeared during child")
                if time.monotonic() - started > timeout:
                    raise GateStop(f"{name}: child exceeded {timeout:g} second timeout")
                status = process.poll()
                if status is not None:
                    break
                time.sleep(POLL_SECONDS)
            if not observed:
                raise GateStop(f"{name}: child RSS/HWM could not be observed")
            if status != 0:
                raise GateStop(f"{name}: native child exited {status}")
        except BaseException as exc:
            _stop(process)
            log_path.with_suffix(".stopped.json").write_text(json.dumps({
                "schema": "real-coherence-child-stopped-v1", "name": name,
                "reason": str(exc), "returncode_after_stop": process.poll(),
                "sampled_peak_rss_kib": peak_rss, "sampled_process_hwm_kib": peak_hwm,
                "memory_observed": observed, "argv": command,
            }, indent=2) + "\n", encoding="utf-8")
            raise
    return {"name": name, "argv": command, "exit_code": status,
            "log": str(log_path), "log_sha256": sha256(log_path),
            "sampled_peak_rss_mib": peak_rss / 1024,
            "sampled_process_hwm_mib": peak_hwm / 1024}


def _expected_chunk_starts(samples: int, chunk_seconds: float = 10.0,
                           overlap_seconds: float = 2.0) -> list[int]:
    chunk, overlap = round(chunk_seconds * RATE), round(overlap_seconds * RATE)
    step = chunk - overlap
    if chunk <= overlap or samples <= 0:
        raise ValueError("invalid chunk geometry")
    starts, pos = [], 0
    while pos + chunk <= samples:
        starts.append(pos)
        pos += step
    if pos < samples:
        final_start = max(0, samples - chunk)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
    if not starts:
        starts.append(0)
    return starts


def _check_render(job: dict, output: Path, cond: dict) -> tuple[dict, np.ndarray]:
    wav = output / "restored.wav"
    meta_path = output / "restoration.json"
    if not wav.is_file() or not meta_path.is_file():
        raise GateStop(f"{job['name']}: missing restored.wav/restoration.json")
    audio = read_wav(wav)
    samples = np.asarray(audio.samples, dtype=np.float32)
    if audio.sample_rate != RATE or audio.channels != 2 or audio.num_samples != job["frames"]:
        raise GateStop(f"{job['name']}: restored geometry mismatch")
    if not np.isfinite(samples).all():
        raise GateStop(f"{job['name']}: nonfinite restored audio")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    continuous = job["arm"] == "continuous_with_final_trusted_band_lock"
    expected = {
        "steps": cond["steps"], "strength": float(np.float32(cond["strength"])),
        "mid_cutoff": float(np.float32(cond["mid_cutoff"])), "mid_ceiling": float(np.float32(cond["mid_ceiling"])),
        "chunk_seconds": cond["chunk_seconds"], "overlap_seconds": cond["overlap_seconds"],
        "controlled": False, "denoise": False, "auto_eq": False,
    }
    historical_baseline = job["name"] == "baseline857-plain-probe"
    if historical_baseline:
        # This pinned build predates these receipt fields. Its exact invocation
        # is preserved separately; missing historical metadata is not fabricated.
        expected.pop("chunk_seconds")
        expected.pop("overlap_seconds")
    if not historical_baseline:
        expected.update({
            "quiet_abstention": job.get("quiet_abstention", False),
            "excitation_mode": "continuous" if continuous else "restart_per_chunk",
            "input_origin_samples": job["origin"] if continuous else 0,
            "whole_track_trusted_projection": continuous,
            "chunk_start_samples": _expected_chunk_starts(
                job["frames"], cond["chunk_seconds"], cond["overlap_seconds"]),
        })
    for key, value in expected.items():
        if meta.get(key) != value:
            raise GateStop(f"{job['name']}: restoration.json {key} mismatch: {meta.get(key)!r}")
    return {"restored_wav": str(wav), "restored_sha256": sha256(wav),
            "restoration_json": str(meta_path), "restoration_json_sha256": sha256(meta_path),
            "settings": {key: meta.get(key) for key in expected},
            "output_geometry": {"rate": audio.sample_rate, "channels": audio.channels,
                                "frames": audio.num_samples}}, samples


def _one_sided_energy_rms(values: np.ndarray, rate: int, mask: np.ndarray | None = None) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Parseval metric requires a finite nonempty mono vector")
    n = x.size
    spec = np.fft.rfft(x) / n
    weights = np.full(spec.size, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    if mask is None:
        mask = np.ones(spec.size, dtype=bool)
    if mask.shape != spec.shape:
        raise ValueError("frequency mask geometry mismatch")
    return float(np.sqrt(np.sum(weights[mask] * np.abs(spec[mask]) ** 2, dtype=np.float64)))


def protected_error_metrics(input_lr: np.ndarray, output_lr: np.ndarray,
                            low_hz: float = 1500.0, high_hz: float = 8000.0,
                            rate: int = RATE) -> dict:
    x = np.asarray(input_lr, dtype=np.float32)
    y = np.asarray(output_lr, dtype=np.float32)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] != 2 or x.shape[1] == 0:
        raise ValueError("protected metric requires aligned stereo frames")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("protected metric refuses nonfinite samples")
    n = x.shape[1]
    freq = np.fft.rfftfreq(n, 1.0 / rate)
    mask = (freq <= low_hz) | (freq >= high_hz)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    xviews = {"left": x[0], "right": x[1],
              "mid_orthonormal": (x[0].astype(np.float64) + x[1].astype(np.float64)) * inv_sqrt2,
              "side_orthonormal": (x[0].astype(np.float64) - x[1].astype(np.float64)) * inv_sqrt2}
    yviews = {"left": y[0], "right": y[1],
              "mid_orthonormal": (y[0].astype(np.float64) + y[1].astype(np.float64)) * inv_sqrt2,
              "side_orthonormal": (y[0].astype(np.float64) - y[1].astype(np.float64)) * inv_sqrt2}
    errors = {}
    for name in xviews:
        errors[name] = _one_sided_energy_rms(
            np.asarray(yviews[name], dtype=np.float64) - np.asarray(xviews[name], dtype=np.float64),
            rate, mask,
        )
    return {"closed_protected_bands_hz": [[0.0, low_hz], [high_hz, rate / 2]],
            "parseval_one_sided_weights": "DC/Nyquist=1; interior=2; odd-length has no Nyquist bin",
            "error_rms": errors, "max_error_rms": max(errors.values())}


def _rms(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a, dtype=np.float64)))


def aligned_layout_rms(view0: np.ndarray, view1: np.ndarray, starts: list[int],
                       common_start: int, common_end: int) -> float:
    if len(starts) != 2 or common_start < max(starts) or common_end <= common_start:
        raise ValueError("invalid common full-source interval")
    stop0, stop1 = common_end - starts[0], common_end - starts[1]
    begin0, begin1 = common_start - starts[0], common_start - starts[1]
    if min(begin0, begin1) < 0 or stop0 > view0.shape[1] or stop1 > view1.shape[1]:
        raise ValueError("common comparison interval exceeds a view")
    a = view0[:, begin0:stop0].astype(np.float64)
    b = view1[:, begin1:stop1].astype(np.float64)
    if a.shape != b.shape or a.shape[1] == 0:
        raise ValueError("aligned view comparison has empty/mismatched frames")
    return _rms(a - b)


def residual_growth(restart: np.ndarray, candidate: np.ndarray, source: np.ndarray,
                    floor: float = 1e-12) -> dict:
    if restart.shape != candidate.shape or restart.shape != source.shape:
        raise ValueError("residual growth requires same geometry")
    base, changed = _rms(restart.astype(np.float64) - source), _rms(candidate.astype(np.float64) - source)
    ratio = changed / base if base > floor else None
    growth = ratio - 1.0 if ratio is not None else None
    return {"restart_residual_rms": base, "candidate_residual_rms": changed,
            "ratio": ratio, "growth_fraction": growth,
            "defined_floor_rms": floor, "defined": ratio is not None}


def first_boundary_residual_differences(restart: np.ndarray, candidate: np.ndarray,
                                       tail_origin: int, boundaries: list[int],
                                       radius_samples: int = RATE // 4) -> list[dict]:
    if restart.shape != candidate.shape or restart.ndim != 2 or restart.shape[0] != 2:
        raise ValueError("boundary comparison requires aligned stereo renders")
    delta = candidate.astype(np.float64) - restart.astype(np.float64)
    rows = []
    for boundary in boundaries:
        lo, hi = max(0, boundary - radius_samples), min(delta.shape[1], boundary + radius_samples)
        changed = np.flatnonzero(np.any(delta[:, lo:hi] != 0.0, axis=0))
        sample = int(lo + changed[0]) if changed.size else None
        rows.append({"boundary_local_sample": boundary, "boundary_global_sample": tail_origin + boundary,
                     "search_window_samples": [lo, hi],
                     "first_differing_residual_sample_local": sample,
                     "first_differing_residual_sample_global": tail_origin + sample if sample is not None else None,
                     "distance_to_boundary_samples": sample - boundary if sample is not None else None,
                     "descriptive_only": True})
    return rows


def _commands(plan: dict, protocol: dict, inputs: dict[str, dict], root: Path) -> list[dict]:
    c = protocol["conditioning"]
    views = protocol["views"]
    tail = protocol.get("tail_probe", protocol.get("tail", protocol.get("tail_control")))
    candidate = Path(plan["candidate_binary"]["path"]).resolve()
    baseline = Path(plan["baseline_binary"]["path"]).resolve()
    model = Path(plan["model"]["path"]).resolve()
    rows = []

    def add(name: str, binary: Path, input_name: str, frames: int, origin: int,
            arm: str, backend: str, quiet: bool, group: str) -> None:
        rows.append({"name": name, "binary": str(binary), "binary_sha256": sha256(binary),
                     "input": inputs[input_name]["path"],
                     "frames": frames, "origin": origin, "arm": arm,
                     "backend": backend, "quiet_abstention": quiet, "group": group})

    for i, start in enumerate(views["view_starts_samples"]):
        origin = views["source_start_sample"] + start
        for arm in ("restart", "continuous_with_final_trusted_band_lock"):
            add(f"view{i}-{arm}", candidate, f"view{i}", views["view_frames"], origin,
                arm, "opencl", True, "paired_view")
    for arm in ("restart", "continuous_with_final_trusted_band_lock"):
        add(f"tail120-{arm}", candidate, "tail120to140_4", tail["frames"],
            tail["source_start_sample"], arm, "opencl", True, "unpaired_tail")
    probe_origin = views["source_start_sample"]
    add("baseline857-plain-probe", baseline, "source_probe_1s", RATE, probe_origin,
        "restart", "cpu", False, "baseline_parity")
    add("candidate-plain-probe", candidate, "source_probe_1s", RATE, probe_origin,
        "restart", "cpu", False, "baseline_parity")
    for backend in ("cpu", "opencl"):
        add(f"candidate-combined-probe-{backend}", candidate, "source_probe_1s", RATE,
            probe_origin, "continuous_with_final_trusted_band_lock", backend, True,
            "backend_parity")
    add("candidate-combined-silence", candidate, "true_silence_16384", 16_384, 0,
        "continuous_with_final_trusted_band_lock", "cpu", True, "silence")
    return rows


def _child_command(plan: dict, protocol: dict, job: dict, render_dir: Path) -> list[str]:
    return _command(Path(job["binary"]).resolve(), Path(plan["model"]["path"]).resolve(),
                    job, render_dir, protocol["conditioning"], job["backend"])


def _latch(root: Path, reason: str, children: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "HALTED.json"
    if marker.exists():
        return
    _write_json_new(marker, {"schema": "real-coherence-gate-halted-v1", "reason": reason,
                            "completed_children": [row["name"] for row in children],
                            "partial_child_receipts": children,
                            "note": "partial evidence preserved; no eligibility result was published"})


def _run_all(plan_path: Path, out: Path) -> dict:
    out = out.resolve()
    plan, protocol, pins = preflight(plan_path, out)
    children, renders = [], {}
    try:
        out.mkdir(parents=True, exist_ok=True)
        if (out / "HALT").exists():
            raise GateStop("manual HALT marker was present before input preparation")
        (out / "inputs").mkdir()
        (out / "renders").mkdir()
        (out / "logs").mkdir()
        geometry, inputs = derive_inputs(Path(plan["source_wav"]["path"]).resolve(), protocol, out)
        derived_pins = [{"name": "derived_" + row["name"], "path": row["path"],
                         "sha256": row["sha256"]} for row in inputs.values()]
        pins.extend(derived_pins)
        _write_json_new(out / "plan.json", plan)
        _write_json_new(out / "derived_geometry.json", geometry)
        if (out / "HALT").exists():
            raise GateStop("manual HALT marker appeared before native sequence")
        jobs = _commands(plan, protocol, inputs, out)
        if len(jobs) != 11:
            raise GateStop(f"expected 11 pinned native calls, derived {len(jobs)}")
        for job in jobs:
            _verify_pins(pins, f"before {job['name']}")
            render = out / "renders" / job["name"]
            render.mkdir()
            command = _child_command(plan, protocol, job, render)
            child = _run_child(job["name"], command, out / "logs" / f"{job['name']}.log", out, pins)
            child.update({"input_sha256": inputs[Path(job["input"]).stem]["sha256"],
                          "render_dir": str(render), "backend": job["backend"],
                          "arm": job["arm"], "group": job["group"]})
            children.append(child)
            validated, samples = _check_render(job, render, protocol["conditioning"])
            child.update(validated)
            renders[job["name"]] = samples
            _verify_pins(pins, f"after {job['name']}")
        if (out / "HALT").exists():
            raise GateStop("manual HALT marker appeared before offline analysis")
        analysis = analyze_renders(protocol, inputs, jobs, renders, out / "HALT")
        gates = evaluate_gates(protocol, analysis)
        complete = {
            "schema": "real-coherence-target-free-gate-complete-v1",
            "scope": "short real-song Mid coherence screen; not a full mastering or production-quality qualification",
            "eligible_for_bounded_listening": all(
                item["pass"] is True for item in gates.values() if item.get("pass") is not None
            ),
            "production_eligible": False,
            "baseline_identity": {"binary_sha256": BASELINE_SHA256,
                                  "source_commit": BASELINE_SOURCE_COMMIT,
                                  "artifact_path": str(BASELINE_PATH)},
            "runner_process_memory_kib": memory(os.getpid()),
            "target_metrics": None,
            "target_metrics_note": "No true target/source reconstruction is available for this real song.",
            "post_finalizer_true_peak": None,
            "post_finalizer_note": "The sequence invokes rich-mid-restore only; final mastering/true-peak validation is a separate later gate.",
            "mastered_output_or_auto_master_claim": None,
            "protocol_sha256": next(p["sha256"] for p in pins if p["name"] == "protocol"),
            "derived_geometry_sha256": geometry["derived_geometry_sha256"],
            "pins": pins,
            "children": children,
            "analysis": analysis,
            "gates": gates,
            "interpretation": "Lower layout residual can make wrong residuals more consistent; this gate grants only bounded listening eligibility, not recovery or preference.",
        }
        if (out / "HALT").exists():
            raise GateStop("manual HALT marker appeared before completion publication")
        _write_json_new(out / "COMPLETE.json", complete)
        return complete
    except BaseException as exc:
        _latch(out, str(exc), children)
        raise


def analyze_renders(protocol: dict, inputs: dict[str, dict], jobs: list[dict],
                    renders: dict[str, np.ndarray], halt_path: Path | None = None) -> dict:
    cond, views = protocol["conditioning"], protocol["views"]
    tail = protocol.get("tail_probe", protocol.get("tail", protocol.get("tail_control")))
    source_arrays = {key: _read_stereo_float(Path(row["path"]), row["frames"])
                     for key, row in inputs.items()}
    paired = {}
    protected_rows = []
    growth_rows = []
    for view_index in (0, 1):
        if halt_path is not None and halt_path.exists():
            raise GateStop("manual HALT marker appeared during offline view analysis")
        input_audio = source_arrays[f"view{view_index}"]
        for arm in ("restart", "continuous_with_final_trusted_band_lock"):
            name = f"view{view_index}-{arm}"
            output = renders[name]
            metric = protected_error_metrics(input_audio, output,
                                             cond["mid_cutoff"], cond["mid_ceiling"])
            protected_rows.append({"job": name, **metric})
            paired.setdefault(arm, {})[view_index] = output
        growth = residual_growth(paired["restart"][view_index],
                                 paired["continuous_with_final_trusted_band_lock"][view_index],
                                 input_audio)
        growth_rows.append({"view": view_index, **growth})
    common_start, common_end = views["comparison_global_samples"]
    starts = views["view_starts_samples"]
    layout = {}
    for arm, results in paired.items():
        baseline_delta = aligned_layout_rms(results[0], results[1], starts, common_start, common_end)
        layout[arm] = {"aligned_common_source_samples": [common_start, common_end],
                       "view0_local_samples": [common_start - starts[0], common_end - starts[0]],
                       "view1_local_samples": [common_start - starts[1], common_end - starts[1]],
                       "layout_difference_rms": baseline_delta}
    restart_delta = layout["restart"]["layout_difference_rms"]
    continuous_delta = layout["continuous_with_final_trusted_band_lock"]["layout_difference_rms"]
    layout_reduction = (1.0 - continuous_delta / restart_delta) if restart_delta > 1e-12 else None
    layout["reduction_fraction"] = layout_reduction
    tail_input = source_arrays["tail120to140_4"]
    tail_rows = []
    for arm in ("restart", "continuous_with_final_trusted_band_lock"):
        if halt_path is not None and halt_path.exists():
            raise GateStop("manual HALT marker appeared during offline tail analysis")
        name = f"tail120-{arm}"
        output = renders[name]
        protected_rows.append({"job": name, **protected_error_metrics(
            tail_input, output, cond["mid_cutoff"], cond["mid_ceiling"])})
        if arm == "restart":
            tail_restart = output
        else:
            tail_combined = output
    tail_growth = residual_growth(tail_restart, tail_combined, tail_input)
    tail_chunk_starts = tail["local_chunk_starts_samples"]
    tail_rows.append({"source_interval_start_sample": tail["source_start_sample"],
                      "frames": tail["frames"],
                      "expected_chunk_starts_samples": tail_chunk_starts,
                      "residual_rms_growth": tail_growth,
                      "layout_reduction": None,
                      "layout_reduction_note": "Unpaired tail has no boundary-shifted comparator.",
                      "first_residual_differences_near_boundaries": first_boundary_residual_differences(
                          tail_restart, tail_combined, tail["source_start_sample"], tail_chunk_starts[1:])})
    probe = renders["candidate-plain-probe"]
    baseline_probe = renders["baseline857-plain-probe"]
    backend_cpu = renders["candidate-combined-probe-cpu"]
    backend_opencl = renders["candidate-combined-probe-opencl"]
    silence = renders["candidate-combined-silence"]
    backend_abs = float(np.max(np.abs(backend_cpu.astype(np.float64) - backend_opencl.astype(np.float64))))
    silence_max = float(np.max(np.abs(silence)))
    protected_gate_rows = [r["max_error_rms"] for r in protected_rows
                           if r["job"].startswith("view") or r["job"].startswith("tail")]
    growth_values = [r["growth_fraction"] for r in growth_rows] + [tail_growth["growth_fraction"]]
    growth_max = max(growth_values) if all(v is not None for v in growth_values) else None
    return {
        "view_protected_error": protected_rows,
        "max_protected_error_rms_all_arms_views_and_tail": max(protected_gate_rows),
        "paired_view_layout": layout,
        "per_view_residual_growth_vs_restart": growth_rows,
        "tail_control": tail_rows,
        "tail_residual_growth_vs_restart": tail_growth,
        "tail_layout_reduction": None,
        "tail_unpaired_boundary_differences_are_descriptive_only": True,
        "baseline857_vs_candidate_plain_probe": {
            "frames": baseline_probe.shape[1],
            "sample_bits_exact": bool(np.array_equal(baseline_probe.view(np.uint32), probe.view(np.uint32))),
            "baseline_sha256": next(j for j in jobs if j["name"] == "baseline857-plain-probe")["binary_sha256"],
        },
        "candidate_cpu_opencl_combined_probe_max_abs": backend_abs,
        "combined_true_silence_max_abs": silence_max,
        "maximum_residual_growth_fraction_views_and_tail": growth_max,
        "metrics_are_target_free": True,
    }


def evaluate_gates(protocol: dict, analysis: dict) -> dict:
    gates = protocol["eligibility_gates"]
    layout_min = float(gates["layout_rms_reduction_min"])
    protected_max = float(gates["max_whole_view_protected_error_rms"])
    growth_max = float(gates["residual_rms_growth_vs_restart_max"])
    backend_max = float(gates["backend_sample_max_abs"])
    layout_reduction = analysis["paired_view_layout"]["reduction_fraction"]
    protected_values = [r["max_error_rms"] for r in analysis["view_protected_error"]
                        if r["job"].startswith("view") or r["job"].startswith("tail")]
    candidate_protected = [r["max_error_rms"] for r in analysis["view_protected_error"]
                           if r["job"].startswith("view") and "continuous_with_final" in r["job"]]
    candidate_tail_protected = [r["max_error_rms"] for r in analysis["view_protected_error"]
                                if r["job"].startswith("tail") and "continuous_with_final" in r["job"]]
    growths = [r["growth_fraction"] for r in analysis["per_view_residual_growth_vs_restart"]]
    growths.append(analysis["tail_residual_growth_vs_restart"]["growth_fraction"])
    all_finite = all(math.isfinite(x) for x in protected_values + [analysis["candidate_cpu_opencl_combined_probe_max_abs"]]
                     + [analysis["combined_true_silence_max_abs"]]
                     + [analysis["paired_view_layout"]["restart"]["layout_difference_rms"],
                        analysis["paired_view_layout"]["continuous_with_final_trusted_band_lock"]["layout_difference_rms"]]
                     + [x for x in growths if x is not None])
    exact = analysis["baseline857_vs_candidate_plain_probe"]["sample_bits_exact"]
    return {
        "full_view_layout_difference_reduction_at_least_threshold": {
            "observed": layout_reduction, "minimum": layout_min,
            "pass": layout_reduction is not None and math.isfinite(layout_reduction) and layout_reduction >= layout_min},
        "combined_whole_view_protected_error_rms": {
            "observed_max": max(candidate_protected) if candidate_protected else None,
            "limit": protected_max,
            "pass": bool(candidate_protected) and max(candidate_protected) <= protected_max},
        "combined_tail_protected_error_rms": {
            "observed_max": max(candidate_tail_protected) if candidate_tail_protected else None,
            "limit": protected_max,
            "pass": bool(candidate_tail_protected) and max(candidate_tail_protected) <= protected_max},
        "residual_rms_growth_each_view_and_tail": {
            "observed": growths, "maximum": growth_max,
            "pass": all(g is not None and math.isfinite(g) and g <= growth_max for g in growths)},
        "baseline_restart_disabled_path_sample_bits_exact": {"pass": exact},
        "combined_cpu_opencl_short_probe_max_abs": {
            "observed": analysis["candidate_cpu_opencl_combined_probe_max_abs"],
            "limit": backend_max,
            "pass": analysis["candidate_cpu_opencl_combined_probe_max_abs"] <= backend_max},
        "combined_true_silence_exact_zero": {
            "observed_max_abs": analysis["combined_true_silence_max_abs"],
            "pass": analysis["combined_true_silence_max_abs"] == 0.0},
        "finite_metrics_and_geometry": {"pass": all_finite},
        "fixed_native_settings_and_chunk_starts": {"pass": True},
        "final_auto_master_true_peak": {
            "observed_dbtp": None, "pass": None,
            "status": "not_run; this gate covers rich-mid outputs only"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = _run_all(args.plan, args.out)
    except GateStop as exc:
        print(f"HALTED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"eligible_for_bounded_listening": result["eligible_for_bounded_listening"],
                      "complete": True}, indent=2))
    return 0 if result["eligible_for_bounded_listening"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
