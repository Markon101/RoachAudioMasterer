#!/usr/bin/env python3
"""Offline stopped-mid negative-control analysis; no inference or target scoring."""
import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import analyze_long_coherence as base
    from .roach_ears.audio_io import read_wav
else:
    import analyze_long_coherence as base
    from roach_ears.audio_io import read_wav

RATE = base.RATE
VIEWS = ("view0", "view3")
ARMS = ("restart", "restart_projected", "continuous", "continuous_projected")


def digest(path):
    return base.digest(path)


def waveform_difference(a, b):
    return base.waveform_difference(a, b)


def _audio(path, frames=None):
    wav = read_wav(path)
    if wav.sample_rate != RATE or wav.channels != 2:
        raise ValueError(f"expected stereo {RATE} Hz audio: {path}")
    base.channels(wav.samples)  # validates finite, nonempty stereo data
    if frames is not None and wav.samples.shape != (2, frames):
        raise ValueError(f"unaligned audio geometry: {path}")
    return wav.samples


def _same(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def _false_residual(x, y):
    d = y.astype(np.float64) - x.astype(np.float64)
    e, signal = base.rms(d), base.rms(x)
    return {"rms": e, "max_abs": float(np.max(np.abs(d))),
            "relative_rms_to_input": e / signal if signal > 1e-14 else None}


def _growth(reference_rms, candidate_rms):
    power_floor = 1e-14
    if reference_rms * reference_rms <= power_floor:
        return {"ratio": None, "denominator_defined": False, "pass": False,
                "null_power_floor": power_floor, "null_rms_floor": power_floor ** 0.5}
    return {"ratio": (candidate_rms - reference_rms) / reference_rms,
            "denominator_defined": True, "null_power_floor": power_floor}


def _layout_reduction(restart_rms, candidate_rms):
    return 1.0 - candidate_rms / restart_rms if restart_rms > 1e-14 else None


def _assess_pair(layout_restart, layout_candidate, residual_restart, residual_candidate, gates):
    reduction = _layout_reduction(layout_restart, layout_candidate)
    growth = _growth(residual_restart, residual_candidate)
    growth["pass"] = (growth["denominator_defined"] and
                      growth["ratio"] <= gates["continuous_projected_vs_restart_projected_false_residual_rms_growth_max"])
    return {"layout_rms_reduction": reduction,
            "layout_pass": reduction is not None and reduction >= gates["layout_rms_reduction_minimum_for_mechanistic_claim"],
            "false_residual_rms_growth": growth}


def _protected(x, y, limit):
    c = base.compare(x.astype(np.float64), y.astype(np.float64))
    rows = {ch: {band: {"error_rms": c[ch][band]["error_rms"],
                        "reference_power": c[ch][band]["reference_power"],
                        "pass": c[ch][band]["error_rms"] <= limit}
                  for band in ("low", "high")}
            for ch in ("left", "right", "mid", "side")}
    maximum = max(v[b]["error_rms"] for v in rows.values() for b in ("low", "high"))
    return {"maximum_error_rms": maximum, "threshold": limit,
            "pass": maximum <= limit, "channels": rows,
            "scope": "whole-view absolute output-minus-input error; protected bands 0-1500 and 8000-24000 Hz"}


def _missing_band_power(audio):
    zero = np.zeros(audio.shape[1], dtype=np.float32)
    return {ch: base.band_metrics(samples, zero)["mid"]["reference_power"]
            for ch, samples in base.channels(audio).items()}


def _verify_crop(full, view, origin, label):
    if origin < 0 or origin + view.shape[1] > full.shape[1] or not _same(full[:, origin:origin + view.shape[1]], view):
        raise ValueError(f"{label} is not a bit-identical full-source crop")


def _fixture_data(root, protocol):
    manifest_path = root / "manifest.json"
    m = json.loads(manifest_path.read_text())
    if (m.get("source_mode") != "stopped_mid" or m.get("sample_rate") != RATE
            or m.get("channels") != 2 or m.get("samples") != 32 * RATE):
        raise ValueError("expected 32s stereo stopped_mid fixture")
    if m.get("recipe", {}).get("damage", {}).get("type") != "identity" or m["recipe"]["damage"].get("applications") != 0:
        raise ValueError("stopped-mid input must declare identity damage")
    if (m.get("source_construction", {}).get("mode") != "stopped_mid"
            or m["source_construction"]["parent_recipe"].get("independent_stems") is not False
            or m["source_construction"]["transform"].get("renormalization_after_transform") is not False):
        raise ValueError("stopped-mid parent/transform provenance is incomplete")
    specs = {v["id"]: v for v in m["views"]}
    if set(specs) != set(VIEWS) or any(specs[v]["samples"] != 26 * RATE for v in VIEWS):
        raise ValueError("expected two 26s views")
    specs["full"] = {"input": "input-full.wav", "target": "target-full.wav", "samples": m["samples"], "origin_sample": 0}
    audio, targets = {}, {}
    for view, s in specs.items():
        for role, dest in (("input", audio), ("target", targets)):
            p = root / s[role]
            key = s[role].replace("-", "_").replace(".wav", "")
            if digest(p) != m["source_sha256"][key]:
                raise ValueError(f"fixture hash mismatch: {s[role]}")
            dest[view] = _audio(p, s["samples"])
        if not _same(audio[view], targets[view]):
            raise ValueError(f"{view} input is not a bit-identical target copy")
    for v in VIEWS:
        _verify_crop(audio["full"], audio[v], specs[v]["origin_sample"], f"{v} input")
        _verify_crop(targets["full"], targets[v], specs[v]["origin_sample"], f"{v} target")
    powers = _missing_band_power(targets["full"])
    threshold = float(protocol["gates"]["stopped_target_full_source_missing_band_power_max"])
    max_power = max(powers.values())
    if max_power > threshold:
        raise ValueError("full-source missing-mid power exceeds frozen null gate")
    interior = m["comparison_interior_global_samples"]
    if interior[1] - interior[0] != 13 * RATE:
        raise ValueError("expected frozen 13-second common interior")
    return m, specs, audio, targets, {"per_channel": powers, "maximum": max_power,
                                      "threshold": threshold, "samples": m["samples"],
                                      "scope": "32s full-source FFT, not a cropped-view FFT"}


def _session(path, mode, fixture, manifest, specs, audio, protocol):
    cp, pp = path / "COMPLETE.json", path / "plan.json"
    if (path / "HALTED.json").exists() or not cp.is_file() or not pp.is_file():
        raise ValueError(f"{mode} session is halted or incomplete")
    c, p = json.loads(cp.read_text()), json.loads(pp.read_text())
    if c.get("jobs_completed") != list(VIEWS) or p.get("backend") != "opencl":
        raise ValueError(f"{mode} session must be the two-view pinned OpenCL run")
    if c.get("binary_sha256") != p.get("binary_sha256") or c.get("model_sha256") != p.get("model_sha256"):
        raise ValueError(f"{mode} plan and completion pins differ")
    for name in ("binary", "model"):
        fp = Path(p[name])
        if not fp.is_file() or digest(fp) != p[name + "_sha256"]:
            raise ValueError(f"{mode} {name} hash mismatch")
    jobs, records = {j["id"]: j for j in p["jobs"]}, {}
    if set(jobs) != set(VIEWS):
        raise ValueError(f"{mode} plan view set mismatch")
    for view in VIEWS:
        s, j = specs[view], jobs[view]
        inp, tgt = fixture / s["input"], fixture / s["target"]
        ih = manifest["source_sha256"][s["input"].replace("-", "_").replace(".wav", "")]
        th = manifest["source_sha256"][s["target"].replace("-", "_").replace(".wav", "")]
        if (Path(j["input"]).resolve() != inp.resolve() or Path(j["target"]).resolve() != tgt.resolve()
                or j.get("input_sha256") != ih
                or j.get("target_sha256") != th or j.get("samples") != s["samples"]
                or j.get("chunk_seconds") != 8 or j.get("overlap_seconds") != 2):
            raise ValueError(f"{mode}/{view} plan differs from fixture or fixed layout")
        receipt = json.loads((path / "receipts" / f"{view}.json").read_text())
        out = path / "renders" / view / "restored.wav"
        oh = digest(out)
        if (oh != receipt["artifacts"]["restored_wav_sha256"]
                or receipt.get("input_sha256") != ih or receipt.get("target_sha256") != th
                or receipt.get("binary_sha256") != c.get("binary_sha256")
                or receipt.get("model_sha256") != c.get("model_sha256")
                or receipt.get("source_pins", []) != c.get("source_pins", [])):
            raise ValueError(f"{mode}/{view} receipt mismatch")
        log = (path / "logs" / f"{view}.log").read_text(errors="replace")
        if "loaded from cache" in log.lower():
            raise ValueError(f"{mode}/{view} log reports a chunk-cache hit")
        y = _audio(out, s["samples"])
        meta = json.loads((out.parent / "restoration.json").read_text())
        want = "continuous" if mode == "continuous" else "restart_per_chunk"
        if (meta.get("excitation_mode", "restart_per_chunk") != want
                or meta.get("steps") != 8 or meta.get("strength") != 1.0
                or meta.get("mid_cutoff") != 1500.0 or meta.get("mid_ceiling") != 8000.0
                or meta.get("controlled") is not False or meta.get("denoise") is not False
                or meta.get("auto_eq") is not False or meta.get("total_seconds") != 26.0
                or (mode == "continuous" and meta.get("input_origin_samples") != s["origin_sample"])):
            raise ValueError(f"{mode}/{view} restoration settings/origin mismatch")
        x = audio[view]
        false, protected = _false_residual(x, y), _protected(x, y, protocol["gates"]["max_protected_absolute_error_rms"])
        stitched, _, overlaps = base.inspect_chunks(out.parent, x.astype(np.float64), s["origin_sample"], 8 * RATE, 2 * RATE)
        parity = waveform_difference(y.astype(np.float64), stitched)
        if parity["max_abs"] > 5e-7:
            raise ValueError(f"{mode}/{view} WOLA reconstruction exceeds 5e-7")
        records[view] = {"path": out, "sha256": oh, "audio": y, "false_residual": false,
                         "protected": protected, "wola_parity": parity,
                         "overlap_cosines": [q["weighted_uncentered_correlation"] for q in overlaps]}
    return {"complete_sha256": digest(cp), "binary_sha256": c["binary_sha256"],
            "model_sha256": c["model_sha256"], "source_pins": c.get("source_pins", []), "views": records}


def _projection(root, protocol_path, fixture, specs, manifest, sources):
    cp = root / "COMPLETE.json"
    c = json.loads(cp.read_text())
    if c.get("protocol_sha256") != digest(protocol_path):
        raise ValueError("projection COMPLETE protocol hash mismatch")
    binary = Path(c["binary"])
    if not binary.is_file() or digest(binary) != c["binary_sha256"]:
        raise ValueError("projection binary hash mismatch")
    for pin in c.get("source_pins", []):
        path = Path(pin["path"])
        if not path.is_file() or digest(path) != pin["sha256"]:
            raise ValueError(f"projection source pin mismatch: {path}")
    jobs = {(j["mode"], j.get("view", j.get("id"))): j for j in c["jobs"]}
    result = {"restart_projected": {}, "continuous_projected": {}}
    for mode in ("restart", "continuous"):
        for view in VIEWS:
            j, src, s = jobs[(mode, view)], sources[mode]["views"][view], specs[view]
            inp, out = fixture / s["input"], root / mode / view / "projected.wav"
            mp = out.parent / "projection.json"
            meta = json.loads(mp.read_text())
            if Path(meta["input"]).resolve() != inp.resolve() or Path(meta["proposal"]).resolve() != src["path"].resolve():
                raise ValueError(f"{mode}/{view} projection metadata path mismatch")
            if (meta.get("frames") != s["samples"] or meta.get("cutoff_hz") != 1500.0
                    or meta.get("ceiling_hz") != 8000.0):
                raise ValueError(f"{mode}/{view} projection geometry/bands mismatch")
            hashes = {"input_sha256": digest(inp), "proposal_sha256": src["sha256"],
                      "output_sha256": digest(out), "metadata_sha256": digest(mp)}
            if any(j.get(k) != v for k, v in hashes.items()):
                raise ValueError(f"{mode}/{view} projection receipt hash mismatch")
            y = _audio(out, s["samples"])
            x = _audio(inp, s["samples"])
            result[f"{mode}_projected"][view] = {"path": out, "sha256": hashes["output_sha256"],
                "metadata_sha256": hashes["metadata_sha256"],
                "audio": y, "false_residual": _false_residual(x, y),
                "protected": _protected(x, y, json.loads(protocol_path.read_text())["gates"]["max_protected_absolute_error_rms"])}
    return {"complete_sha256": digest(cp), "binary_sha256": c["binary_sha256"],
            "source_pins": c.get("source_pins", []), "views": result}


def _layout(arms, specs, interior):
    begin, end = interior
    result = {}
    for arm in ARMS:
        a, b = arms[arm]["view0"]["audio"], arms[arm]["view3"]["audio"]
        o0, o3 = specs["view0"]["origin_sample"], specs["view3"]["origin_sample"]
        result[arm] = waveform_difference(a[:, begin-o0:end-o0], b[:, begin-o3:end-o3])
    return result


def _layout_reductions(layout):
    r = layout["restart"]["rms"]
    rp = layout["restart_projected"]["rms"]
    return {"continuous_vs_restart": _layout_reduction(r, layout["continuous"]["rms"]),
            "continuous_projected_vs_restart_projected": _layout_reduction(rp, layout["continuous_projected"]["rms"])}


def analyze(fixture, restart_session, continuous_session, projection_root, protocol_path):
    fixture, restart_session, continuous_session, projection_root, protocol_path = map(Path.resolve,
        (fixture, restart_session, continuous_session, projection_root, protocol_path))
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "adverse-mid-development-protocol-v1":
        raise ValueError("unexpected adverse protocol schema")
    manifest, specs, audio, targets, null = _fixture_data(fixture, protocol)
    restart = _session(restart_session, "restart", fixture, manifest, specs, audio, protocol)
    continuous = _session(continuous_session, "continuous", fixture, manifest, specs, audio, protocol)
    if restart["binary_sha256"] != continuous["binary_sha256"] or restart["model_sha256"] != continuous["model_sha256"]:
        raise ValueError("restart and continuous controls must share binary and model")
    sources = {"restart": restart, "continuous": continuous}
    projection = _projection(projection_root, protocol_path, fixture, specs, manifest, sources)
    arms = {"restart": restart["views"], "continuous": continuous["views"], **projection["views"]}
    layout = _layout(arms, specs, manifest["comparison_interior_global_samples"])
    reductions = _layout_reductions(layout)
    growth = {}
    gmax = protocol["gates"]["continuous_projected_vs_restart_projected_false_residual_rms_growth_max"]
    for v in VIEWS:
        growth[v] = _assess_pair(layout["restart_projected"]["rms"],
            layout["continuous_projected"]["rms"],
            arms["restart_projected"][v]["false_residual"]["rms"],
            arms["continuous_projected"][v]["false_residual"]["rms"], protocol["gates"])
    growth_pass = all(row["false_residual_rms_growth"]["pass"] for row in growth.values())
    layout_min = protocol["gates"]["layout_rms_reduction_minimum_for_mechanistic_claim"]
    layout_pass = reductions["continuous_projected_vs_restart_projected"] is not None and reductions["continuous_projected_vs_restart_projected"] >= layout_min
    protected_pass = all(arms[a][v]["protected"]["pass"] for a in ARMS for v in VIEWS)
    return {"schema": "stopped-mid-coherence-analysis-v1",
        "scope": "forced missing-mid negative control; no autonomous gating, target scoring, or audio-quality claim",
        "provenance": {"analysis_source_sha256": digest(Path(__file__)),
            "shared_analysis_source_sha256": digest(Path(base.__file__)),
            "fixture_manifest_sha256": digest(fixture / "manifest.json"),
            "protocol_sha256": digest(protocol_path), "restart_complete_sha256": restart["complete_sha256"],
            "continuous_complete_sha256": continuous["complete_sha256"],
            "projection_complete_sha256": projection["complete_sha256"],
            "source_mode": manifest["source_mode"], "generator_sha256": manifest["generator_sha256"],
            "restart_binary_sha256": restart["binary_sha256"], "continuous_binary_sha256": continuous["binary_sha256"],
            "model_sha256": restart["model_sha256"], "projection_binary_sha256": projection["binary_sha256"],
            "source_pins": {"restart": restart["source_pins"], "continuous": continuous["source_pins"],
                            "projection": projection["source_pins"]}},
        "fixture_null": null, "identity_oracle": {v: {"false_residual_rms": 0.0, "max_abs": 0.0,
            "bitwise_output_equals_input": True, "scope": "mathematical identity oracle, not model output"} for v in VIEWS},
        "common_interior_global_samples": manifest["comparison_interior_global_samples"],
        "aligned_input_identical": True,
        "views": {v: {"origin_sample": specs[v]["origin_sample"], "samples": specs[v]["samples"],
            "input_sha256": manifest["source_sha256"][specs[v]["input"].replace("-", "_").replace(".wav", "")],
            "target_sha256": manifest["source_sha256"][specs[v]["target"].replace("-", "_").replace(".wav", "")],
            "arms": {a: {"output_sha256": arms[a][v]["sha256"],
                "false_residual": arms[a][v]["false_residual"], "whole_view_protected": arms[a][v]["protected"],
                "raw_wola_parity": arms[a][v].get("wola_parity"), "raw_overlap_cosines": arms[a][v].get("overlap_cosines")}
                for a in ARMS}} for v in VIEWS},
        "layout_waveform_differences": layout, "layout_rms_reductions": reductions,
        "projected_negative_control_gates_by_view": growth,
        "gates": {"false_residual_growth_max": gmax, "false_residual_growth_pass": growth_pass,
            "minimum_layout_rms_reduction": layout_min, "layout_reduction_pass": layout_pass,
            "max_protected_absolute_error_rms": protocol["gates"]["max_protected_absolute_error_rms"],
            "all_whole_view_protected_pass": protected_pass,
            "stopped_target_full_source_missing_band_power_max": null["maximum"],
            "stopped_target_full_source_missing_band_pass": null["maximum"] <= null["threshold"]},
        "interpretation": "Nonzero residual on bit-identical stopped-mid input is a false residual under this forced policy; coherent residual is not quality evidence.",
        "promotion": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--restart-session", type=Path, required=True)
    p.add_argument("--continuous-session", type=Path, required=True)
    p.add_argument("--projection-root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.fixture, a.restart_session, a.continuous_session, a.projection_root, a.protocol)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
