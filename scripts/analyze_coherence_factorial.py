#!/usr/bin/env python3
"""Bounded offline comparison of restart/continuous and projected/unprojected arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from . import analyze_long_coherence as base
    from .roach_ears.audio_io import read_wav
else:
    import analyze_long_coherence as base
    from roach_ears.audio_io import read_wav


RATE = base.RATE
ARMS = ("restart", "restart_projected", "continuous", "continuous_projected")
VIEWS = ("view0", "view3", "full")
def digest(path: str | Path) -> str:
    return base.digest(path)


def rms(x) -> float:
    return base.rms(x)


def waveform_difference(a, b) -> dict:
    return base.waveform_difference(a, b)


def _absolute(path: str | Path, relative_to: Path | None = None) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute() and relative_to is not None:
        p = relative_to / p
    return p.resolve()


def _audio(path: Path, name: str, frames: int | None = None) -> np.ndarray:
    raw = read_wav(path)
    if raw.sample_rate != RATE or raw.channels != 2:
        raise ValueError(f"{name}: expected stereo {RATE} Hz WAV")
    # `base.channels` validates finite, nonempty stereo and returns an L/R/M/S
    # dictionary for metrics; keep the physical L/R matrix for alignment and
    # waveform-difference calculations.
    base.channels(raw.samples)
    samples = np.asarray(raw.samples, dtype=np.float64)
    if samples.shape[1] > 48 * RATE:
        raise ValueError(f"{name}: exceeds 48-second analysis bound")
    if frames is not None and samples.shape[1] != frames:
        raise ValueError(f"{name}: expected {frames} frames, found {samples.shape[1]}")
    return samples


def _verify_crop(full: np.ndarray, view: np.ndarray, origin: int, name: str) -> None:
    if origin < 0 or origin + view.shape[1] > full.shape[1]:
        raise ValueError(f"{name}: view exceeds full physical source")
    expected = full[:, origin:origin + view.shape[1]]
    if not np.array_equal(expected.view(np.uint64), view.view(np.uint64)):
        raise ValueError(f"{name}: view is not an exact crop of the full source")


def _stereo_correlation(audio: np.ndarray) -> float | None:
    left, right = audio[0], audio[1]
    l = left - np.mean(left, dtype=np.float64)
    r = right - np.mean(right, dtype=np.float64)
    denom = float(np.linalg.norm(l) * np.linalg.norm(r))
    return float(np.dot(l, r) / denom) if denom > 1e-30 else None


def _protected_gate(comparison: dict, limit: float) -> dict:
    """Use the protocol's absolute whole-view protected-band RMS threshold."""
    rows = {}
    all_errors = []
    for channel in ("left", "right", "mid", "side"):
        rows[channel] = {}
        for band in ("low", "high"):
            metric = comparison[channel][band]
            error = float(metric["error_rms"])
            power = float(metric["reference_power"])
            relative = error / math.sqrt(power) if power > 1e-14 else None
            rows[channel][band] = {"absolute_error_rms": error,
                                   "relative_error_ratio": relative,
                                   "reference_power": power,
                                   "pass": error <= limit}
            all_errors.append(error)
    maximum = max(all_errors, default=0.0)
    return {"threshold_absolute_rms": limit, "maximum_absolute_error_rms": maximum,
            "pass": maximum <= limit,
            "definition": "unwindowed whole-view output-minus-input error RMS in closed protected bands",
            "bands_hz": {"low": [0.0, 1500.0], "high": [8000.0, 24000.0]},
            "channels": rows}


def _factorial_layout_gates(layouts: dict, gates: dict) -> dict:
    restart = layouts["restart"]["rms"]
    restart_projected = layouts["restart_projected"]["rms"]
    continuous = layouts["continuous"]["rms"]
    continuous_projected = layouts["continuous_projected"]["rms"]

    def reduction(reference, candidate):
        if reference is None or candidate is None or reference <= 1e-12:
            return None
        return 1.0 - candidate / reference

    projection_only = reduction(restart, restart_projected)
    excitation = reduction(restart, continuous)
    excitation_projected = reduction(restart_projected, continuous_projected)
    projection_limit = float(gates["projection_only_layout_rms_reduction_maximum_for_excitation_attribution"])
    excitation_minimum = float(gates["layout_rms_reduction_vs_restart_minimum"])
    return {
        "projection_only_layout_rms_reduction": projection_only,
        "projection_only_attribution_limit": projection_limit,
        "projection_only_attribution_gate_pass": projection_only is not None and projection_only <= projection_limit,
        "continuous_layout_rms_reduction_vs_restart": excitation,
        "continuous_projected_layout_rms_reduction_vs_restart_projected": excitation_projected,
        "minimum_excitation_reduction": excitation_minimum,
        "continuous_layout_gate_pass": excitation is not None and excitation >= excitation_minimum,
        "continuous_projected_layout_gate_pass": excitation_projected is not None and excitation_projected >= excitation_minimum,
        "excitation_attribution_supported_by_this_gate": (
            projection_only is not None and projection_only <= projection_limit
            and excitation is not None and excitation >= excitation_minimum),
        "qualification": "single-phrase mechanistic comparison; gate does not establish audible or target-fidelity improvement",
    }


def _session_records(session: Path, name: str, fixture: Path,
                     manifest: dict) -> dict[str, dict]:
    if (session / "HALTED.json").exists():
        raise ValueError(f"{name} session is halted")
    complete_path = session / "COMPLETE.json"
    plan_path = session / "plan.json"
    if not complete_path.is_file() or not plan_path.is_file():
        raise ValueError(f"{name} session is incomplete")
    complete = json.loads(complete_path.read_text())
    plan = json.loads(plan_path.read_text())
    if complete.get("jobs_completed") != list(VIEWS):
        raise ValueError(f"{name} session does not contain all fixed views in order")
    jobs = {job["id"]: job for job in plan.get("jobs", [])}
    if set(jobs) != set(VIEWS):
        raise ValueError(f"{name} plan does not contain the fixed view set")
    specs = {v["id"]: v for v in manifest["views"]}
    records = {}
    for view in VIEWS:
        spec = specs[view] if view != "full" else {
            "input": "input-full.wav", "target": "target-full.wav",
            "origin_sample": 0, "samples": manifest["samples"]}
        job = jobs[view]
        input_path = fixture / spec["input"]
        target_path = fixture / spec["target"]
        if _absolute(job["input"]) != input_path.resolve() or _absolute(job["target"]) != target_path.resolve():
            raise ValueError(f"{name}/{view}: plan paths do not match fixture manifest")
        expected_input_sha = manifest["source_sha256"][spec["input"].replace("-", "_").replace(".wav", "")]
        expected_target_sha = manifest["source_sha256"][spec["target"].replace("-", "_").replace(".wav", "")]
        if digest(input_path) != expected_input_sha or digest(target_path) != expected_target_sha:
            raise ValueError(f"{name}/{view}: fixture audio hash changed")
        if job.get("input_sha256") != expected_input_sha or job.get("target_sha256") != expected_target_sha:
            raise ValueError(f"{name}/{view}: session plan hashes differ from fixture")
        if job.get("samples") != spec["samples"]:
            raise ValueError(f"{name}/{view}: session sample count differs from fixture")
        expected_chunk, expected_overlap = (32.0, 0.0) if view == "full" else (8.0, 2.0)
        if (job.get("chunk_seconds") != expected_chunk
                or job.get("overlap_seconds") != expected_overlap):
            raise ValueError(f"{name}/{view}: session chunk layout differs from the fixed protocol")
        receipt_path = session / "receipts" / f"{view}.json"
        receipt = json.loads(receipt_path.read_text())
        artifact = receipt["artifacts"]
        output_path = session / "renders" / view / "restored.wav"
        if _absolute(artifact["restored_wav"]) != output_path.resolve():
            raise ValueError(f"{name}/{view}: receipt path mismatch")
        output_sha = digest(output_path)
        if output_sha != artifact["restored_wav_sha256"]:
            raise ValueError(f"{name}/{view}: output changed since session receipt")
        if (receipt.get("input_sha256") != expected_input_sha
                or receipt.get("target_sha256") != expected_target_sha
                or receipt.get("binary_sha256") != complete.get("binary_sha256")
                or receipt.get("model_sha256") != complete.get("model_sha256")):
            raise ValueError(f"{name}/{view}: receipt provenance differs from COMPLETE")
        if (complete.get("input_sha256", {}).get(view) != expected_input_sha
                or complete.get("target_sha256", {}).get(view) != expected_target_sha):
            raise ValueError(f"{name}/{view}: COMPLETE audio hashes differ from fixture")
        if receipt.get("source_pins", []) != complete.get("source_pins", []):
            raise ValueError(f"{name}/{view}: receipt source pins differ from COMPLETE")
        metadata = json.loads((output_path.parent / "restoration.json").read_text())
        if metadata.get("steps") != 8 or metadata.get("strength") != 1.0:
            raise ValueError(f"{name}/{view}: expected fixed Euler8 strength 1 settings")
        if name == "continuous" and metadata.get("excitation_mode") != "continuous":
            raise ValueError(f"{name}/{view}: output is not continuous-excitation mode")
        if name == "restart" and metadata.get("excitation_mode", "restart_per_chunk") != "restart_per_chunk":
            raise ValueError(f"{name}/{view}: output is not restart-per-chunk mode")
        if metadata.get("input_origin_samples", 0) != (spec["origin_sample"] if name == "continuous" else 0):
            raise ValueError(f"{name}/{view}: native sample origin differs from fixture view origin")
        records[view] = {"path": output_path, "sha256": output_sha,
                         "input_sha256": expected_input_sha, "target_sha256": expected_target_sha,
                         "binary_sha256": complete["binary_sha256"],
                         "model_sha256": complete["model_sha256"],
                         "source_pins": complete.get("source_pins", []),
                         "metadata": metadata}
    return records


def _projection_records(projection_root: Path, fixture: Path, protocol_path: Path,
                        protocol: dict, source_records: dict) -> tuple[dict, dict]:
    complete_path = projection_root / "COMPLETE.json"
    if not complete_path.is_file():
        raise ValueError("projection root has no COMPLETE.json")
    complete = json.loads(complete_path.read_text())
    if complete.get("protocol_sha256") != digest(protocol_path):
        raise ValueError("projection COMPLETE does not pin the supplied protocol")
    binary = Path(complete["binary"]).resolve()
    binary_sha = complete["binary_sha256"]
    if not binary.is_file() or digest(binary) != binary_sha:
        raise ValueError("projection binary is missing or its hash changed")
    source_pins = complete.get("source_pins")
    if not isinstance(source_pins, list) or not source_pins:
        raise ValueError("projection COMPLETE must pin source files")
    pins_by_path = {}
    for pin in source_pins:
        path = Path(pin["path"]).resolve()
        if digest(path) != pin["sha256"]:
            raise ValueError(f"projection source pin changed: {path}")
        pins_by_path[path] = pin["sha256"]
    for relpath, sha in protocol.get("source_sha256", {}).items():
        path = (Path(__file__).resolve().parents[1] / relpath).resolve()
        if not path.is_file() or digest(path) != sha or pins_by_path.get(path) != sha:
            raise ValueError(f"projection protocol source pin mismatch: {relpath}")
    jobs = complete.get("jobs")
    expected_pairs = {(mode, view) for mode in ("restart", "continuous") for view in VIEWS}
    if not isinstance(jobs, list) or len(jobs) != 6:
        raise ValueError("projection COMPLETE must contain six projection jobs")
    by_key = {}
    for job in jobs:
        view = job.get("view", job.get("id"))
        key = (job.get("mode"), view)
        if key not in expected_pairs or key in by_key:
            raise ValueError(f"unexpected/duplicate projection job identity: {key}")
        by_key[key] = job
    if set(by_key) != expected_pairs:
        raise ValueError("projection COMPLETE lacks one or more factorial jobs")

    result = {"restart_projected": {}, "continuous_projected": {}}
    specs = {v["id"]: v for v in json.loads((fixture / "manifest.json").read_text())["views"]}
    specs["full"] = {"input": "input-full.wav", "origin_sample": 0,
                      "samples": json.loads((fixture / "manifest.json").read_text())["samples"]}
    for mode in ("restart", "continuous"):
        for view in VIEWS:
            job = by_key[(mode, view)]
            source = source_records[mode][view]
            spec = specs[view]
            input_path = fixture / spec["input"]
            output_path = projection_root / mode / view / "projected.wav"
            metadata_path = output_path.parent / "projection.json"
            if not output_path.is_file() or not metadata_path.is_file():
                raise ValueError(f"missing projected artifact for {mode}/{view}")
            metadata = json.loads(metadata_path.read_text())
            if (_absolute(metadata["input"]) != input_path.resolve()
                    or _absolute(metadata["proposal"]) != source["path"].resolve()):
                raise ValueError(f"{mode}/{view}: projection metadata paths do not match input/proposal")
            if metadata.get("frames") != spec["samples"] or metadata.get("cutoff_hz") != 1500.0 or metadata.get("ceiling_hz") != 8000.0:
                raise ValueError(f"{mode}/{view}: unexpected projection geometry or bands")
            hashes = {"input_sha256": digest(input_path), "proposal_sha256": source["sha256"],
                      "output_sha256": digest(output_path),
                      "projection_metadata_sha256": digest(metadata_path)}
            for key, value in hashes.items():
                if job.get(key) != value:
                    raise ValueError(f"{mode}/{view}: projection COMPLETE {key} mismatch")
            result[f"{mode}_projected"][view] = {"path": output_path,
                                                  "sha256": hashes["output_sha256"],
                                                  "metadata_sha256": hashes["projection_metadata_sha256"],
                                                  "input_sha256": hashes["input_sha256"],
                                                  "proposal_sha256": hashes["proposal_sha256"],
                                                  "binary_sha256": binary_sha,
                                                  "metadata": metadata}
    provenance = {"complete_sha256": digest(complete_path), "binary": str(binary),
                  "binary_sha256": binary_sha,
                  "source_pins": [{"path": str(p.resolve()), "sha256": sha}
                                  for p, sha in sorted(pins_by_path.items(), key=lambda x: str(x[0]))]}
    return result, provenance


def _window_stats(audio: np.ndarray, start: int, stop: int) -> dict:
    segment = audio[..., start:stop]
    if not segment.shape[-1]:
        return {"rms": None, "sample_peak": None}
    return {"rms": rms(segment), "sample_peak": float(np.max(np.abs(segment)))}


def _onset_rows(audio: np.ndarray, input_audio: np.ndarray, target: np.ndarray,
                events: list[dict]) -> list[dict]:
    half = round(0.020 * RATE)
    rows = []
    out_channels = base.channels(audio)
    input_channels = base.channels(input_audio)
    target_channels = base.channels(target)
    for event in events:
        sample = int(event.get("view_sample", event["sample"]))
        start, stop = max(0, sample - half), min(audio.shape[1], sample + half)
        out_stats = {key: _window_stats(val, start, stop)
                     for key, val in out_channels.items()}
        input_stats = {key: _window_stats(val[None, :], start, stop)
                       for key, val in input_channels.items()}
        target_stats = {key: _window_stats(val[None, :], start, stop)
                        for key, val in target_channels.items()}
        ratios = {}
        for channel in ("left", "right", "mid"):
            op, ip, tp = (out_stats[channel]["sample_peak"], input_stats[channel]["sample_peak"],
                          target_stats[channel]["sample_peak"])
            ratios[channel] = {
                "output_to_target_peak_ratio": op / tp if tp and tp > 1e-12 else None,
                "output_to_target_rms_ratio": (out_stats[channel]["rms"] / target_stats[channel]["rms"]
                                                if target_stats[channel]["rms"] and target_stats[channel]["rms"] > 1e-12 else None),
                "output_peak_minus_input": op - ip if ip is not None else None,
            }
        rows.append({"event_global_sample": event["sample"], "event_view_sample": sample,
                     "window_samples": [start, stop], "window_rule": "event +/-20ms, clipped to view",
                     "output": out_stats, "input": input_stats, "target": target_stats,
                     "relative": ratios})
    return rows


def analyze(fixture: Path, restart_session: Path, continuous_session: Path,
            projection_root: Path, protocol_path: Path) -> dict:
    fixture = fixture.resolve()
    restart_session, continuous_session = restart_session.resolve(), continuous_session.resolve()
    projection_root, protocol_path = projection_root.resolve(), protocol_path.resolve()
    manifest_path = fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "long-coherence-procedural-fixture-v1" or manifest.get("sample_rate") != RATE or manifest.get("channels") != 2:
        raise ValueError("fixture manifest geometry/schema mismatch")
    if manifest.get("samples", 0) > 48 * RATE:
        raise ValueError("fixture exceeds the 48-second bound")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "coherence-projection-factorial-protocol-v1" or protocol.get("arms") != list(ARMS):
        raise ValueError("projection protocol schema/arm order mismatch")
    if protocol.get("source_sessions", {}).get("restart") != str(restart_session) and Path(protocol.get("source_sessions", {}).get("restart", "")).resolve() != restart_session:
        raise ValueError("protocol restart session path mismatch")
    if protocol.get("source_sessions", {}).get("continuous") != str(continuous_session) and Path(protocol.get("source_sessions", {}).get("continuous", "")).resolve() != continuous_session:
        raise ValueError("protocol continuous session path mismatch")
    if not any("no target scaling" in item for item in protocol.get("controls", [])):
        raise ValueError("protocol must state that targets are not used for scaling")
    for name, sha in protocol.get("source_sha256", {}).items():
        path = (Path(__file__).resolve().parents[1] / name).resolve()
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"projection protocol source changed: {name}")

    specs = {v["id"]: dict(v) for v in manifest.get("views", [])}
    if set(specs) != {"view0", "view3"}:
        raise ValueError("fixture must contain view0 and view3")
    specs["full"] = {"id": "full", "input": "input-full.wav", "target": "target-full.wav",
                      "origin_sample": 0, "samples": manifest["samples"], "events": manifest.get("events", [])}
    input_audio, target_audio = {}, {}
    for view in VIEWS:
        spec = specs[view]
        input_path, target_path = fixture / spec["input"], fixture / spec["target"]
        for role, path in (("input", input_path), ("target", target_path)):
            key = path.name.replace("-", "_").replace(".wav", "")
            if digest(path) != manifest["source_sha256"].get(key):
                raise ValueError(f"fixture {role} hash mismatch: {path.name}")
        input_audio[view] = _audio(input_path, f"fixture/{view}/input", spec["samples"])
        target_audio[view] = _audio(target_path, f"fixture/{view}/target", spec["samples"])
    _verify_crop(input_audio["full"], input_audio["view0"], specs["view0"]["origin_sample"], "view0 input")
    _verify_crop(input_audio["full"], input_audio["view3"], specs["view3"]["origin_sample"], "view3 input")
    _verify_crop(target_audio["full"], target_audio["view0"], specs["view0"]["origin_sample"], "view0 target")
    _verify_crop(target_audio["full"], target_audio["view3"], specs["view3"]["origin_sample"], "view3 target")

    interior = manifest["comparison_interior_global_samples"]
    if len(interior) != 2 or interior[1] - interior[0] != 13 * RATE:
        raise ValueError("factorial protocol requires the manifest's 13-second common interior")
    begin, end = map(int, interior)
    for view in ("view0", "view3"):
        origin = int(specs[view]["origin_sample"])
        if begin < origin or end > origin + specs[view]["samples"]:
            raise ValueError(f"common interior exceeds {view}")
    i0 = input_audio["view0"][:, begin - specs["view0"]["origin_sample"]:end - specs["view0"]["origin_sample"]]
    i3 = input_audio["view3"][:, begin - specs["view3"]["origin_sample"]:end - specs["view3"]["origin_sample"]]
    if not np.array_equal(i0, i3):
        raise ValueError("aligned view inputs differ on manifest common interior")

    sources = {"restart": _session_records(restart_session, "restart", fixture, manifest),
               "continuous": _session_records(continuous_session, "continuous", fixture, manifest)}
    restart_first = sources["restart"]["view0"]
    continuous_first = sources["continuous"]["view0"]
    if restart_first["model_sha256"] != continuous_first["model_sha256"]:
        raise ValueError("restart/continuous sessions used different model checkpoints")
    restart_complete = json.loads((restart_session / "COMPLETE.json").read_text())
    continuous_complete = json.loads((continuous_session / "COMPLETE.json").read_text())
    if restart_complete.get("target_set_sha256") != continuous_complete.get("target_set_sha256"):
        raise ValueError("restart/continuous sessions do not share the same target set")
    generator_sha = manifest.get("generator_sha256")
    for name, source_records in sources.items():
        pinned_generators = [p.get("sha256") for p in source_records["view0"]["source_pins"]
                             if p.get("name", "").endswith("scripts/long_coherence_fixture.py")]
        if generator_sha is None or pinned_generators != [generator_sha]:
            raise ValueError(f"{name} source pins do not match fixture generator hash")
    projected, projection_provenance = _projection_records(
        projection_root, fixture, protocol_path, protocol, sources)
    records = {"restart": sources["restart"], "continuous": sources["continuous"], **projected}
    limit = float(protocol["gates"]["max_whole_view_protected_error_rms"])
    views = {}
    layout_audio = {arm: {} for arm in ARMS if arm != "full"}
    for view in VIEWS:
        spec = specs[view]
        x, target = input_audio[view], target_audio[view]
        identity = base.compare(target, x)
        arms = {}
        unprojected_audio = {}
        for mode in ("restart", "continuous"):
            source_record = sources[mode][view]
            proposal = _audio(source_record["path"], f"{mode}/{view}", spec["samples"])
            unprojected_audio[mode] = proposal
            projected_record = projected[f"{mode}_projected"][view]
            projected_audio = _audio(projected_record["path"], f"{mode}_projected/{view}", spec["samples"])
            for arm, y, record in ((mode, proposal, source_record),
                                   (f"{mode}_projected", projected_audio, projected_record)):
                input_metrics = base.compare(x, y)
                target_metrics = base.compare(target, y)
                gate = _protected_gate(input_metrics, limit)
                sample_stats = {"rms": rms(y), "sample_peak": float(np.max(np.abs(y))),
                                "stereo_correlation": _stereo_correlation(y)}
                row = {"output_sha256": record["sha256"], "sample_stats": sample_stats,
                       "vs_input": input_metrics, "vs_target": target_metrics,
                       "protected_band_gate": gate}
                if arm.endswith("_projected"):
                    source_y = unprojected_audio[mode]
                    delta = waveform_difference(source_y, y)
                    row["projection_delta_vs_unprojected"] = {
                        **delta,
                        "peak_delta": float(np.max(np.abs(y)) - np.max(np.abs(source_y))),
                        "stereo_correlation_before": _stereo_correlation(source_y),
                        "stereo_correlation_after": _stereo_correlation(y),
                    }
                events = spec.get("events", []) if view != "full" else manifest.get("events", [])
                row["onsets"] = _onset_rows(y, x, target, events)
                arms[arm] = row
                if view != "full":
                    origin = int(spec["origin_sample"])
                    layout_audio[arm][view] = y[:, begin - origin:end - origin].copy()
        views[view] = {"origin_sample": spec["origin_sample"], "samples": spec["samples"],
                       "input_sha256": manifest["source_sha256"][spec["input"].replace("-", "_").replace(".wav", "")],
                       "target_sha256": manifest["source_sha256"][spec["target"].replace("-", "_").replace(".wav", "")],
                       "identity_vs_target": identity, "arms": arms}

    layout = {}
    for arm in layout_audio:
        layout[arm] = waveform_difference(layout_audio[arm]["view0"], layout_audio[arm]["view3"])
    factorial_gates = _factorial_layout_gates(layout, protocol["gates"])
    return {
        "schema": "coherence-projection-factorial-analysis-v1",
        "scope": "one procedural phrase; no inference, target-selected gain, or perceptual claim",
        "provenance": {
            "fixture_manifest_sha256": digest(manifest_path),
            "restart_complete_sha256": digest(restart_session / "COMPLETE.json"),
            "continuous_complete_sha256": digest(continuous_session / "COMPLETE.json"),
            "projection_complete_sha256": projection_provenance["complete_sha256"],
            "projection_protocol_sha256": digest(protocol_path),
            "analysis_source_sha256": digest(Path(__file__)),
            "sessions": {name: {"binary_sha256": next(iter(records.values()))["binary_sha256"],
                                "model_sha256": next(iter(records.values()))["model_sha256"],
                                "source_pins": next(iter(records.values()))["source_pins"]}
                         for name, records in sources.items()},
            "projection": projection_provenance,
            "audio_hashes": {view: {"input": views[view]["input_sha256"],
                                     "target": views[view]["target_sha256"],
                                     "outputs": {arm: views[view]["arms"][arm]["output_sha256"] for arm in ARMS}}
                             for view in VIEWS},
        },
        "definitions": {
            "rate": RATE, "arms": list(ARMS), "views": list(VIEWS),
            "bands_hz": {"low": [0.0, 1500.0], "mid": [1500.0, 8000.0], "high": [8000.0, 24000.0]},
            "protected_gate": "absolute output-minus-input error RMS over each whole view; also report error RMS/sqrt(input band power) when input power >1e-14",
            "layout_scope": "13-second aligned common interior from fixture manifest; cropped FFT metrics are diagnostic and do not act as the whole-view gate",
            "onset_window": "local +/-20 ms sample RMS and peak, relative to target without gain selection",
            "sample_peak_note": "sample peak only; no true-peak or loudness claim",
        },
        "protocol_gates": protocol["gates"],
        "views": views,
        "common_interior_global_samples": [begin, end],
        "aligned_input_identical": True,
        "layout_waveform_differences": layout,
        "factorial_layout_gates": factorial_gates,
        "promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--restart-session", type=Path, required=True)
    parser.add_argument("--continuous-session", type=Path, required=True)
    parser.add_argument("--projection-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing existing output: {args.out}")
    result = analyze(args.fixture, args.restart_session, args.continuous_session,
                     args.projection_root, args.protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
