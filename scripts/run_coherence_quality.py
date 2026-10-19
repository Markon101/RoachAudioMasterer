#!/usr/bin/env python3
"""Run the pinned, sequential CPU quality session for long Mid coherence.

Plan schema (created by the session owner):
{
  "schema": "long-coherence-quality-v1", "root": "/fresh/output",
  "binary": "/path/to/roach-audio-masterer", "binary_sha256": "...",
  "model": "/path/to/basis.json", "model_sha256": "...",
  "source_pins": [{"name": "fixture-generator", "path": "...", "sha256": "..."}],
  "target_set_sha256": "...",
  "jobs": [{"id": "view0", "input": "...", "input_sha256": "...",
            "target": "...", "target_sha256": "...", "samples": 1248000,
            "chunk_seconds": 8, "overlap_seconds": 2}, ...]
}
Jobs must be ordered view0, view3, full, with the pinned dimensions below.
No profiling/cache environment is passed to the native process.
"""

from __future__ import annotations

import argparse
from array import array
import json
import math
import os
import platform
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mid_profile as shared  # reuse the established digest/memory readers


RATE = 48_000
CHANNELS = 2
SAMPLE_BYTES = 4
MEMORY_LIMIT_KIB = 768 * 1024
TIMEOUT_SECONDS = 300.0
POLL_SECONDS = 0.25
EXPECTED = (
    {"id": "view0", "samples": 26 * RATE, "chunk_seconds": 8.0,
     "overlap_seconds": 2.0},
    {"id": "view3", "samples": 26 * RATE, "chunk_seconds": 8.0,
     "overlap_seconds": 2.0},
    {"id": "full", "samples": 32 * RATE, "chunk_seconds": 32.0,
     "overlap_seconds": 0.0},
)
PROFILE_ENV = (
    "ROACH_PROFILE_MID", "ROACH_FOREGROUND_CONFIRMED", "ROACH_MID_CACHE_MIB",
    "ROACH_MID_CACHE_BUILD", "ROACH_MID_BATCH_ROWS",
)


class SafetyStop(RuntimeError):
    """A latched stop condition; no subsequent job may be launched."""


def digest(path: str | Path) -> str:
    return shared.digest(path)


def memory(pid: int) -> dict[str, int]:
    return shared.memory(pid)


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def target_set_digest(jobs: list[dict]) -> str:
    """Hash the fixed ordered target identity set for all three views."""
    material = [{"id": j["id"], "target_sha256": j["target_sha256"],
                 "samples": j["samples"]} for j in jobs]
    import hashlib
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def wav_info(path: str | Path) -> dict:
    """Read RIFF format and declared data geometry without loading audio."""
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError(f"not a RIFF/WAVE file: {path}")
        riff_size = struct.unpack_from("<I", header, 4)[0]
        fmt = None
        data_size = None
        while True:
            chunk = f.read(8)
            if len(chunk) != 8:
                break
            name, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if name == b"fmt ":
                fmt = f.read(size)
                if size & 1:
                    f.seek(1, os.SEEK_CUR)
            elif name == b"data":
                data_size = size
                data_offset = f.tell()
                f.seek(size + (size & 1), os.SEEK_CUR)
            else:
                f.seek(size + (size & 1), os.SEEK_CUR)
        if fmt is None or data_size is None or len(fmt) < 16:
            raise ValueError(f"missing fmt/data chunk: {path}")
        audio_format, channels, rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", fmt)
        container_format = audio_format
        if audio_format == 0xFFFE:
            # Hound writes native stereo float32 in WAVE_FORMAT_EXTENSIBLE.
            ieee_float_guid = bytes.fromhex("0300000000001000800000aa00389b71")
            if len(fmt) < 40 or struct.unpack_from("<H", fmt, 16)[0] < 22 or fmt[24:40] != ieee_float_guid:
                raise ValueError(f"unsupported extensible WAV subformat: {path}")
            audio_format = 3
        frames, rem = divmod(data_size, align) if align else (0, data_size)
        if rem:
            raise ValueError(f"invalid WAV data alignment: {path}")
        expected_size = 8 + riff_size
        actual_size = path.stat().st_size
        if expected_size != actual_size or data_size != frames * align:
            raise ValueError(f"truncated or inconsistent RIFF length: {path}")
        return {"format": audio_format, "container_format": container_format,
                "channels": channels, "rate": rate,
                "byte_rate": byte_rate, "block_align": align,
                "bits_per_sample": bits, "frames": frames,
                "data_bytes": data_size, "data_offset": data_offset,
                "file_bytes": actual_size}


def _assert_finite_float_wav(path: Path, info: dict) -> None:
    if info["format"] != 3 or info["bits_per_sample"] != 32:
        raise SafetyStop(f"restored output is not IEEE float32: {path}")
    remaining = info["data_bytes"]
    with path.open("rb") as f:
        f.seek(info["data_offset"])
        while remaining:
            block = f.read(min(1024 * 1024, remaining))
            if not block or len(block) % 4:
                raise SafetyStop(f"truncated float samples in {path}")
            values = array("f")
            values.frombytes(block)
            if sys.byteorder != "little":
                values.byteswap()
            if any(not math.isfinite(value) for value in values):
                raise SafetyStop(f"nonfinite restored samples in {path}")
            remaining -= len(block)


def _expected_chunks(samples: int, chunk_s: float, overlap_s: float) -> list[int]:
    chunk = int(round(chunk_s * RATE))
    overlap = int(round(overlap_s * RATE))
    step = chunk - overlap
    if chunk <= overlap:
        raise ValueError("chunk duration must exceed overlap")
    starts = []
    pos = 0
    while pos + chunk <= samples:
        starts.append(pos)
        pos += step
    if pos < samples:
        last = max(0, samples - chunk)
        if not starts or starts[-1] != last:
            starts.append(last)
    if not starts:
        starts.append(0)
    return [min(chunk, samples - start) for start in starts]


def _safe_root(plan: dict) -> tuple[Path, bool]:
    raw = plan.get("root")
    if not isinstance(raw, str) or not raw:
        raise ValueError("plan root must be an absolute output path")
    root = Path(raw)
    if not root.is_absolute() or root == Path("/"):
        raise ValueError("plan root must be a non-root absolute path")
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"output root is not a directory: {root}")
    if root.exists():
        children = {p.name for p in root.iterdir()}
        if children - {"HALT"}:
            raise FileExistsError(f"output root must be fresh; found {sorted(children)}")
    return root, root.exists()


def _validate_plan(plan_path: Path, plan: dict, root: Path) -> list[dict]:
    if plan.get("schema") != "long-coherence-quality-v1":
        raise ValueError("unsupported quality plan schema")
    for key in ("binary", "model"):
        path = Path(plan[key])
        if not path.is_file() or digest(path) != _sha(plan.get(key + "_sha256"), key + "_sha256"):
            raise ValueError(f"{key} hash mismatch or file missing")
    pins = plan.get("source_pins")
    if not isinstance(pins, list) or not pins:
        raise ValueError("source_pins must pin the fixture generator/manifest provenance")
    seen_names = set()
    for pin in pins:
        if not isinstance(pin, dict) or not isinstance(pin.get("name"), str) or pin["name"] in seen_names:
            raise ValueError("invalid or duplicate source pin")
        seen_names.add(pin["name"])
        path = Path(pin["path"])
        if not path.is_file() or digest(path) != _sha(pin.get("sha256"), pin["name"]):
            raise ValueError(f"source pin hash mismatch: {pin.get('name')}")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(EXPECTED):
        raise ValueError("plan must contain exactly the three fixed quality jobs")
    for job, expected in zip(jobs, EXPECTED):
        for key, value in expected.items():
            actual = job.get(key)
            if key == "id":
                if type(actual) is not str or actual != value:
                    raise ValueError(f"job id must be {value}")
            elif key in ("chunk_seconds", "overlap_seconds"):
                if type(actual) not in (int, float) or float(actual) != value:
                    raise ValueError(f"{job.get('id')} {key} must be {value}")
            elif type(actual) is not int or actual != value:
                raise ValueError(f"{job.get('id')} {key} must be {value}")
        for key in ("input", "target"):
            path = Path(job[key])
            sha = _sha(job.get(key + "_sha256"), f"{job['id']}.{key}_sha256")
            if not path.is_file() or digest(path) != sha:
                raise ValueError(f"{job['id']} {key} hash mismatch or file missing")
            info = wav_info(path)
            if (info["format"], info["channels"], info["rate"], info["bits_per_sample"], info["frames"]) != (3, CHANNELS, RATE, 32, expected["samples"]):
                raise ValueError(f"{job['id']} {key} is not expected stereo 48 kHz float32 geometry")
    expected_genotype = target_set_digest(jobs)
    if _sha(plan.get("target_set_sha256"), "target_set_sha256") != expected_genotype:
        raise ValueError("target set hash mismatch")
    # A plan is supplied outside the fresh output root so writing run evidence
    # cannot overwrite the authorization document.
    if root.resolve() in plan_path.resolve().parents:
        raise ValueError("plan must live outside the fresh output root")
    return jobs


def _pins(plan: dict, jobs: list[dict]) -> list[dict]:
    rows = [{"name": "binary", "path": plan["binary"], "sha256": plan["binary_sha256"]},
            {"name": "model", "path": plan["model"], "sha256": plan["model_sha256"]}]
    rows += [{"name": p["name"], "path": p["path"], "sha256": p["sha256"]}
             for p in plan["source_pins"]]
    for j in jobs:
        rows.extend(({"name": j["id"] + ":input", "path": j["input"], "sha256": j["input_sha256"]},
                     {"name": j["id"] + ":target", "path": j["target"], "sha256": j["target_sha256"]}))
    return rows


def _verify_pins(pins: list[dict], stage: str) -> None:
    for pin in pins:
        try:
            actual = digest(pin["path"])
        except OSError as exc:
            raise SafetyStop(f"{stage}: pinned file unavailable: {pin['name']}: {exc}") from exc
        if actual != pin["sha256"]:
            raise SafetyStop(f"{stage}: pinned hash changed: {pin['name']}")


def _command(plan: dict, job: dict, out: Path) -> list[str]:
    return [plan["binary"], "rich-mid-restore", "--model", plan["model"],
            "--input", job["input"], "--backend", "cpu", "--steps", "8",
            "--strength", "1", "--chunk-seconds", str(job["chunk_seconds"]),
            "--overlap-seconds", str(job["overlap_seconds"]), "--out", str(out)]


def _environment(root: Path) -> dict:
    mem_available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1])
                break
    except OSError:
        pass
    def version(command: list[str]) -> str | None:
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=2,
                                  check=False).stdout.strip()[:200] or None
        except (OSError, subprocess.TimeoutExpired):
            return None
    disk = shutil.disk_usage(root.parent)
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "cargo": version(["cargo", "--version"]),
            "rustc": version(["rustc", "--version"]),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "memory_available_kib": mem_available,
            "free_disk_bytes_at_start": disk.free}


def _copy_plan(plan_path: Path, root: Path) -> None:
    destination = root / "plan.json"
    if destination.exists():
        raise FileExistsError(f"run plan output collision: {destination}")
    shutil.copyfile(plan_path, destination)


def _latch(root: Path, plan_path: Path, reason: str, jobs_done: list[dict],
           timeout: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    plan_copy = root / "plan.json"
    if not plan_copy.exists() and plan_path.is_file():
        shutil.copyfile(plan_path, plan_copy)
    payload = {"schema": "long-coherence-halted-v1", "reason": reason,
               "halted_at_utc": datetime.now(timezone.utc).isoformat(),
               "jobs_completed": [j["id"] for j in jobs_done],
               "timeout_seconds": TIMEOUT_SECONDS if timeout else None,
               "partial_receipts": [r["receipt"] for r in jobs_done]}
    payload["failed_child_observations"] = [str(p) for p in (root / "logs").glob("*.stopped.json")]
    marker = root / "HALTED.json"
    if not marker.exists():
        marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_child(command: list[str], log_path: Path, root: Path) -> dict:
    env = os.environ.copy()
    for key in PROFILE_ENV:
        env.pop(key, None)
    started = time.monotonic()
    peak_rss = peak_hwm = 0
    observed = False
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while True:
                counters = memory(process.pid)
                if counters:
                    observed = True
                peak_rss = max(peak_rss, counters.get("VmRSS", 0))
                peak_hwm = max(peak_hwm, counters.get("VmHWM", 0))
                if max(peak_rss, peak_hwm) > MEMORY_LIMIT_KIB:
                    raise SafetyStop("observed child RSS/HWM exceeded 768 MiB cap")
                if (root / "HALT").exists():
                    raise SafetyStop("explicit HALT marker appeared")
                if time.monotonic() - started > TIMEOUT_SECONDS:
                    raise SafetyStop("child exceeded 300-second safety timeout")
                status = process.poll()
                if status is not None:
                    break
                time.sleep(POLL_SECONDS)
            if not observed:
                raise SafetyStop("unable to observe child RSS/HWM; resource cap unverified")
            return {"returncode": status, "sampled_peak_rss_kib": peak_rss,
                    "sampled_process_hwm_kib": peak_hwm}
        except BaseException as error:
            _stop_child(process)
            observation = {"reason": str(error), "sampled_peak_rss_kib": peak_rss,
                           "sampled_process_hwm_kib": peak_hwm, "memory_observed": observed,
                           "returncode_after_stop": process.poll(), "log": str(log_path)}
            log_path.with_suffix(".stopped.json").write_text(json.dumps(observation, indent=2) + "\n")
            raise


def _validate_output(job: dict, out: Path, log_path: Path) -> dict:
    log = log_path.read_text(errors="replace")
    if "loaded from cache" in log.lower():
        raise SafetyStop(f"{job['id']}: native log reports a disk chunk cache hit")
    restored = out / "restored.wav"
    restoration_path = out / "restoration.json"
    if not restored.is_file() or not restoration_path.is_file():
        raise SafetyStop(f"{job['id']}: missing restored.wav or restoration.json")
    info = wav_info(restored)
    if (info["format"], info["channels"], info["rate"], info["bits_per_sample"], info["frames"]) != (3, CHANNELS, RATE, 32, job["samples"]):
        raise SafetyStop(f"{job['id']}: restored WAV format/length differs from expected input geometry")
    _assert_finite_float_wav(restored, info)
    metadata = json.loads(restoration_path.read_text())
    if metadata.get("steps") != 8 or metadata.get("strength") != 1.0:
        raise SafetyStop(f"{job['id']}: restoration settings do not show Euler8 strength 1")
    if metadata.get("controlled") is not False or metadata.get("denoise") is not False or metadata.get("auto_eq") is not False:
        raise SafetyStop(f"{job['id']}: unexpected controlled degradation or post-processing setting")
    if abs(float(metadata.get("total_seconds", -1)) - job["samples"] / RATE) > 1e-9:
        raise SafetyStop(f"{job['id']}: restoration duration does not match input")
    expected_sizes = _expected_chunks(job["samples"], job["chunk_seconds"], job["overlap_seconds"])
    chunk_dir = out / "chunks"
    chunks = sorted(chunk_dir.glob("chunk_*.bin")) if chunk_dir.is_dir() else []
    if len(chunks) != len(expected_sizes):
        raise SafetyStop(f"{job['id']}: expected {len(expected_sizes)} raw float chunks, found {len(chunks)}")
    chunk_bytes = []
    for path, count in zip(chunks, expected_sizes):
        expected_bytes = count * CHANNELS * SAMPLE_BYTES
        if path.stat().st_size != expected_bytes:
            raise SafetyStop(f"{job['id']}: unexpected raw chunk byte count: {path.name}")
        chunk_bytes.append(expected_bytes)
    return {"restored_wav": str(restored), "restored_wav_sha256": digest(restored),
            "restoration_json_sha256": digest(restoration_path),
            "restoration_settings": {"steps": metadata["steps"], "strength": metadata["strength"],
                                     "duration_seconds": metadata["total_seconds"]},
            "restored_wav_info": info,
            "chunk_files": [p.name for p in chunks], "chunk_sample_counts": expected_sizes,
            "chunk_byte_counts": chunk_bytes}


def run(plan_path: str | Path) -> dict:
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root, existed = _safe_root(plan)
    completed: list[dict] = []
    # A pre-existing root is accepted only when it contains the explicit HALT
    # marker; this lets an owner stop before the first child launches.
    if existed and (root / "HALT").exists():
        _latch(root, plan_path, "explicit HALT marker present before first job", completed)
        raise SafetyStop("HALTED before first job by explicit marker")
    jobs: list[dict] = []
    try:
        jobs = _validate_plan(plan_path, plan, root)
        pins = _pins(plan, jobs)
        _verify_pins(pins, "preflight")
    except (OSError, ValueError, KeyError, TypeError, SafetyStop) as exc:
        _latch(root, plan_path, f"preflight mismatch: {exc}", completed)
        raise SafetyStop(str(exc)) from exc

    root.mkdir(parents=True, exist_ok=True)
    _copy_plan(plan_path, root)
    (root / "environment.json").write_text(json.dumps(_environment(root), indent=2) + "\n", encoding="utf-8")
    (root / "logs").mkdir()
    (root / "renders").mkdir()
    (root / "receipts").mkdir()
    for index, job in enumerate(jobs):
        out = root / "renders" / job["id"]
        log_path = root / "logs" / f"{job['id']}.log"
        if out.exists() or log_path.exists():
            reason = f"output collision before {job['id']}"
            _latch(root, plan_path, reason, completed)
            raise SafetyStop(reason)
        if (root / "HALT").exists():
            _latch(root, plan_path, "explicit HALT marker before child launch", completed)
            raise SafetyStop("HALTED before child launch")
        try:
            if shutil.disk_usage(root).free < 512 * 1024 * 1024:
                raise SafetyStop("less than 512 MiB free storage before child launch")
            _verify_pins(pins, f"before-{job['id']}")
            command = _command(plan, job, out)
            out.mkdir()
            child = _run_child(command, log_path, root)
            _verify_pins(pins, f"after-{job['id']}")
            if child["returncode"] != 0:
                raise SafetyStop(f"{job['id']}: native command exited {child['returncode']}")
            if (root / "HALT").exists():
                raise SafetyStop("explicit HALT marker appeared after child exit")
            artifacts = _validate_output(job, out, log_path)
            receipt = {"job_id": job["id"], "command": command,
                       "input_sha256": job["input_sha256"], "target_sha256": job["target_sha256"],
                       "target_set_sha256": plan["target_set_sha256"],
                       "binary_sha256": plan["binary_sha256"], "model_sha256": plan["model_sha256"],
                       "source_pins": [{"name": p["name"], "sha256": p["sha256"]}
                                       for p in plan["source_pins"]],
                       "sampled_peak_rss_kib": child["sampled_peak_rss_kib"],
                       "sampled_process_hwm_kib": child["sampled_process_hwm_kib"],
                       "artifacts": artifacts}
            receipt_path = root / "receipts" / f"{job['id']}.json"
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            completed.append({"id": job["id"], "receipt": str(receipt_path), "receipt_data": receipt})
        except SafetyStop as exc:
            _latch(root, plan_path, str(exc), completed,
                   timeout="300-second safety timeout" in str(exc))
            raise
        except BaseException as exc:
            _latch(root, plan_path, f"{job['id']} failed: {type(exc).__name__}: {exc}", completed)
            raise

    if (root / "HALT").exists():
        _latch(root, plan_path, "explicit HALT marker appeared before completion", completed)
        raise SafetyStop("HALTED before completion marker")
    complete = {"schema": "long-coherence-quality-complete-v1",
                "job_count": len(completed), "jobs_completed": [j["id"] for j in completed],
                "target_set_sha256": plan["target_set_sha256"],
                "binary_sha256": plan["binary_sha256"], "model_sha256": plan["model_sha256"],
                "source_pins": [{"name": p["name"], "sha256": p["sha256"]}
                                for p in plan["source_pins"]],
                "input_sha256": {j["id"]: j["input_sha256"] for j in jobs},
                "target_sha256": {j["id"]: j["target_sha256"] for j in jobs},
                "receipts": [j["receipt"] for j in completed],
                "statement": "quality run completed; native CLI logs retained without timing interpretation"}
    path = root / "COMPLETE.json"
    if path.exists() or (root / "HALTED.json").exists():
        _latch(root, plan_path, "completion marker collision or prior halt", completed)
        raise SafetyStop("output completion marker collision")
    path.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    print(f"Completed pinned quality session: {len(completed)} jobs.", flush=True)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    try:
        run(args.plan)
    except SafetyStop as exc:
        print(f"HALTED: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
