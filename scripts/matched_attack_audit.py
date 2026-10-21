#!/usr/bin/env python3
"""Source-selected paired attack diagnostics; NumPy only, bounded WAV reads.

This is an offline diagnostic of existing final masters, not a punch estimator
or perceptual validation. Definitions live in a create-first design.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import struct
import sys

import numpy as np


RATIO_WINDOWS = {
    "attack_to_sustain_rms_db": ("attack", "sustain"),
    "attack_to_sustain_peak_db": ("attack", "sustain"),
    "attack_peak_to_sustain_rms_db": ("attack", "sustain"),
    "attack_to_pre_rms_db": ("attack", "pre"),
    "attack_crest_minus_pre_db": ("attack", "pre"),
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def check_latches(output):
    present = [name for name in ("HALT", "HALT.json", "STOP", "STOP.json")
               if (Path(output) / name).exists()]
    if present:
        raise RuntimeError(f"artifact-local stop latch present: {present}")


def check_pins(pins):
    for path, expected in pins.items():
        if sha256(path) != expected:
            raise ValueError(f"input/tool pin changed: {path}")


def memory_receipt(cap_mib=None):
    result = {"ru_maxrss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _ = line.split()
                result[key.rstrip(":") + "_kib"] = int(value)
    except OSError:
        pass
    result["observed_hwm_mib"] = max(result["ru_maxrss_kib_linux"],
                                      result.get("VmHWM_kib", 0)) / 1024
    if cap_mib is not None and result["observed_hwm_mib"] > cap_mib:
        raise MemoryError(f"observed process HWM exceeds {cap_mib} MiB: {result}")
    return result


class WavReader:
    """Strict little-endian PCM16 or IEEE-float32 WAV with bounded float64 reads."""

    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        try:
            header = self.file.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
                raise ValueError("expected little-endian RIFF/WAVE")
            declared_size = struct.unpack("<I", header[4:8])[0] + 8
            if declared_size != self.path.stat().st_size:
                raise ValueError("RIFF size differs from file length")
            fmt = None
            data = None
            while self.file.tell() + 8 <= declared_size:
                tag, size = struct.unpack("<4sI", self.file.read(8))
                offset = self.file.tell()
                if offset + size > declared_size:
                    raise ValueError("truncated WAV chunk")
                if tag == b"fmt ":
                    if fmt is not None:
                        raise ValueError("duplicate fmt chunk")
                    fmt = self.file.read(size)
                elif tag == b"data":
                    if data is not None:
                        raise ValueError("duplicate data chunk")
                    data = (offset, size)
                self.file.seek(offset + size + (size & 1))
            if fmt is None or len(fmt) < 16 or data is None:
                raise ValueError("missing WAV format/data")
            tag, self.channels, self.rate, byte_rate, align, bits = struct.unpack("<HHIIHH", fmt[:16])
            if tag == 0xFFFE:
                if len(fmt) < 40 or struct.unpack("<H", fmt[16:18])[0] < 22:
                    raise ValueError("invalid extensible WAV")
                if fmt[28:40] != bytes.fromhex("00001000800000aa00389b71"):
                    raise ValueError("unsupported extensible WAV GUID")
                tag = struct.unpack("<I", fmt[24:28])[0]
                valid_bits = struct.unpack("<H", fmt[18:20])[0]
                if valid_bits not in (0, bits):
                    raise ValueError("unsupported valid-bit count")
            if (tag, bits) == (1, 16):
                self.dtype, self.scale = np.dtype("<i2"), 1 / 32768
                self.encoding = "PCM16"
            elif (tag, bits) == (3, 32):
                self.dtype, self.scale = np.dtype("<f4"), 1.0
                self.encoding = "IEEE-float32"
            else:
                raise ValueError(f"unsupported WAV format {tag}/{bits}")
            self.frame_bytes = self.channels * self.dtype.itemsize
            if self.channels != 2 or align != self.frame_bytes or byte_rate != self.rate * align:
                raise ValueError("expected well-formed stereo WAV")
            self.offset, size = data
            if size % align:
                raise ValueError("partial WAV frame")
            self.frames = size // align
        except Exception:
            self.file.close()
            raise

    def close(self):
        self.file.close()

    def metadata(self):
        return {"path": str(self.path), "rate": self.rate, "channels": self.channels,
                "frames": self.frames, "encoding": self.encoding,
                "data_offset": self.offset, "frame_bytes": self.frame_bytes}

    def read(self, start, end):
        if not 0 <= start < end <= self.frames or end - start > 65536:
            raise ValueError(f"invalid or unbounded read [{start}, {end})")
        self.file.seek(self.offset + start * self.frame_bytes)
        data = self.file.read((end - start) * self.frame_bytes)
        if len(data) != (end - start) * self.frame_bytes:
            raise ValueError("short WAV read")
        samples = np.frombuffer(data, dtype=self.dtype).astype(np.float64).reshape(-1, self.channels)
        samples *= self.scale
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"nonfinite audio in {self.path} at {start}")
        return samples


def verify_geometry(readers, geometry):
    for name, reader in readers.items():
        if (reader.rate, reader.channels, reader.frames) != (
                geometry["sample_rate"], geometry["channels"], geometry["frames"]):
            raise ValueError(f"geometry mismatch for {name}: {reader.metadata()}")


def validate_audio(reader):
    energy, peak = 0.0, 0.0
    for start in range(0, reader.frames, 65536):
        x = reader.read(start, min(start + 65536, reader.frames))
        energy += float(np.sum(x * x))
        peak = max(peak, float(np.max(np.abs(x))))
    return {"all_samples_finite": True, "sample_peak": peak,
            "rms": math.sqrt(energy / (reader.frames * reader.channels))}


def ms_samples(milliseconds, rate):
    return int(round(milliseconds * rate / 1000))


def make_bandpass(rate, band, taps):
    if taps % 2 != 1 or taps < 3:
        raise ValueError("FIR length must be odd and >=3")
    lo, hi = band
    if not 0 < lo < hi < rate / 2:
        raise ValueError("invalid band edges")
    q = np.arange(taps, dtype=np.float64) - taps // 2
    h = (2 * hi / rate * np.sinc(2 * hi / rate * q)
         - 2 * lo / rate * np.sinc(2 * lo / rate * q)) * np.hamming(taps)
    response = float(np.sum(h * np.cos(2 * np.pi * ((lo + hi) / 2) / rate * q)))
    if response <= 0:
        raise ValueError("nonpositive FIR midpoint response")
    return h / response


def filtered_patch(reader, start, end, coefficients):
    half = len(coefficients) // 2
    patch = reader.read(start - half, end + half)
    nfft = 1 << (len(patch) + len(coefficients) - 2).bit_length()
    kernel = np.fft.rfft(coefficients, n=nfft)
    output = np.empty((end - start, reader.channels), dtype=np.float64)
    for channel in range(reader.channels):
        spectrum = np.fft.rfft(patch[:, channel], n=nfft)
        spectrum *= kernel
        convolved = np.fft.irfft(spectrum, n=nfft)
        output[:, channel] = convolved[2 * half:2 * half + end - start]
    return output


def normalized_lag_search(reference, extended, max_lag, outside_neighborhood):
    """Positive lag means the second recording is delayed relative to reference."""
    n = len(reference)
    if extended.shape != (n + 2 * max_lag, reference.shape[1]):
        raise ValueError("bad alignment patch geometry")
    nfft = 1 << (len(extended) + n - 2).bit_length()
    dots = np.zeros(2 * max_lag + 1, dtype=np.float64)
    energy_y = np.zeros_like(dots)
    ref = reference - np.mean(reference, axis=0)
    energy_x = float(np.sum(ref * ref))
    for channel in range(reference.shape[1]):
        conv = np.fft.irfft(np.fft.rfft(extended[:, channel], n=nfft)
                            * np.fft.rfft(ref[::-1, channel], n=nfft), n=nfft)
        dots += conv[n - 1:n + 2 * max_lag]
        y = extended[:, channel]
        cumulative = np.concatenate(([0.0], np.cumsum(y)))
        square_cumulative = np.concatenate(([0.0], np.cumsum(y * y)))
        channel_sum = cumulative[n:] - cumulative[:-n]
        energy_y += square_cumulative[n:] - square_cumulative[:-n] - channel_sum * channel_sum / n
    if energy_x <= 0 or np.any(energy_y <= 0):
        raise ValueError("silent or degenerate alignment anchor")
    correlations = dots / np.sqrt(energy_x * np.maximum(energy_y, 0))
    best_index = int(np.argmax(correlations))
    lags = np.arange(-max_lag, max_lag + 1)
    outside = np.abs(lags) > outside_neighborhood
    return {"best_lag_samples": best_index - max_lag,
            "best_correlation": float(correlations[best_index]),
            "zero_lag_correlation": float(correlations[max_lag]),
            "zero_peak_margin_outside_neighborhood": float(correlations[max_lag] - np.max(correlations[outside]))}


def alignment_audit(readers, design):
    cfg = design["alignment"]
    rate = design["geometry"]["sample_rate"]
    frames = design["geometry"]["frames"]
    coefficients = make_bandpass(rate, cfg["band_hz"], design["filter"]["taps"])
    length = ms_samples(cfg["anchor_seconds"] * 1000, rate)
    lag = ms_samples(cfg["max_lag_ms"], rate)
    neighborhood = ms_samples(cfg["outside_zero_neighborhood_ms"], rate)
    records = []
    for fraction in cfg["anchor_fractions"]:
        start = int(round(fraction * frames)) - length // 2
        for pair in cfg["pairs"]:
            first, second = pair.split("/")
            reference = filtered_patch(readers[first], start, start + length, coefficients)
            extended = filtered_patch(readers[second], start - lag, start + length + lag, coefficients)
            record = normalized_lag_search(reference, extended, lag, neighborhood)
            record.update(pair=pair, anchor_fraction=fraction, start_sample=start, end_sample=start + length)
            record["passed"] = (
                record["best_lag_samples"] == cfg["required_best_lag_samples"]
                and record["zero_lag_correlation"] >= cfg["minimum_correlation"]
                and record["zero_peak_margin_outside_neighborhood"] >= cfg["minimum_peak_margin_outside_neighborhood"])
            records.append(record)
    return {"passed": all(r["passed"] for r in records), "anchors": records,
            "shift_applied_samples": 0, "scope": "integer-lag anchors only; no sub-sample or every-event guarantee"}


def input_block_rms(reader, frame_length):
    count = reader.frames // frame_length
    rms = np.empty(count, dtype=np.float64)
    frames_per_read = max(1, 65536 // frame_length)
    for index in range(0, count, frames_per_read):
        stop = min(index + frames_per_read, count)
        x = reader.read(index * frame_length, stop * frame_length)
        blocks = x.reshape(stop - index, frame_length, reader.channels)
        rms[index:stop] = np.sqrt(np.mean(blocks * blocks, axis=(1, 2)))
    return rms


def select_events(block_rms, rate, frames, detector, windows_ms, half_support):
    frame = ms_samples(detector["frame_ms"], rate)
    history = ms_samples(detector["history_ms"], rate) // frame
    radius = detector["local_peak_radius_frames"]
    if history < 1 or len(block_rms) <= history + 2 * radius:
        raise ValueError("insufficient input for onset detector")
    rms = np.asarray(block_rms, dtype=np.float64)
    score = np.full(len(rms), -np.inf)
    cumulative = np.concatenate(([0.0], np.cumsum(rms * rms)))
    past_rms = np.sqrt(np.maximum((cumulative[history:-1] - cumulative[:-history - 1]) / history, 0))
    score[history:] = 20 * np.log10(np.maximum(rms[history:], 1e-15)
                                   / np.maximum(past_rms, 1e-15))
    activity_floor = float(np.max(rms)) * 10 ** (detector["activity_relative_to_max_block_rms_db"] / 20)
    first_offset = min(ms_samples(v[0], rate) for v in windows_ms.values())
    last_offset = max(ms_samples(v[1], rate) for v in windows_ms.values())
    candidates = []
    above = 0
    for index in range(max(history, radius), len(rms) - radius):
        if rms[index] < activity_floor or rms[index] <= 0 or score[index] < detector["rise_threshold_db"]:
            continue
        above += 1
        if not (np.all(score[index] > score[index - radius:index])
                and np.all(score[index] >= score[index + 1:index + radius + 1])):
            continue
        sample = index * frame
        reason = "pending"
        if sample + first_offset - half_support < 0 or sample + last_offset + half_support > frames:
            reason = "insufficient_window_or_FIR_boundary_support"
        candidates.append({"frame_index": index, "onset_sample": sample,
                           "onset_seconds": sample / rate, "rise_db": float(score[index]),
                           "input_frame_rms": float(rms[index]), "status": reason})
    accepted = []
    spacing = ms_samples(detector["minimum_spacing_ms"], rate)
    for candidate in sorted(candidates, key=lambda row: (-row["rise_db"], row["onset_sample"])):
        if candidate["status"] != "pending":
            continue
        blockers = [row for row in accepted if abs(row["onset_sample"] - candidate["onset_sample"]) < spacing]
        if blockers:
            candidate["status"] = "suppressed_by_stronger_source_event"
            candidate["suppressor_onset_sample"] = blockers[0]["onset_sample"]
        else:
            candidate["status"] = "accepted"
            accepted.append(candidate)
    accepted.sort(key=lambda row: row["onset_sample"])
    candidates.sort(key=lambda row: row["onset_sample"])
    for index, row in enumerate(accepted):
        row["event_id"] = f"E{index + 1:04d}"
    return {"source_only": True, "frame_samples": frame, "history_frames": history,
            "complete_input_frames": len(rms), "discarded_tail_samples": frames % frame,
            "input_max_block_rms": float(np.max(rms)), "detector_activity_floor": activity_floor,
            "above_threshold_active_frames_with_history_and_neighbors": above,
            "local_maximum_candidates": len(candidates), "accepted_count": len(accepted),
            "exclusion_counts": {reason: sum(row["status"] == reason for row in candidates)
                                 for reason in ("insufficient_window_or_FIR_boundary_support",
                                                "suppressed_by_stronger_source_event")},
            "candidates": candidates, "accepted": accepted}


def amplitude_db(value):
    return 20 * math.log10(value) if value > 0 else None


def window_metrics(audio):
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio * audio)))
    return {"peak": peak, "rms": rms, "peak_dbfs": amplitude_db(peak),
            "rms_dbfs": amplitude_db(rms),
            "crest_db": 20 * math.log10(peak / rms) if peak > 0 and rms > 0 else None}


def measure_event(reader, onset, design, filters):
    rate = reader.rate
    windows = {key: [onset + ms_samples(value, rate) for value in interval]
               for key, interval in design["windows_ms"].items()}
    start = min(interval[0] for interval in windows.values())
    end = max(interval[1] for interval in windows.values())
    result = {}
    for name, coefficients in filters.items():
        audio = reader.read(start, end) if coefficients is None else filtered_patch(reader, start, end, coefficients)
        result[name] = {key: window_metrics(audio[lo - start:hi - start])
                        for key, (lo, hi) in windows.items()}
    return result


def difference(first, second):
    return first - second if first is not None and second is not None else None


def ratios(windows, source_windows, source_floor):
    values, reasons = {}, {}
    for name, (attack, other) in RATIO_WINDOWS.items():
        if source_windows[attack]["rms"] <= source_floor or source_windows[other]["rms"] <= source_floor:
            values[name], reasons[name] = None, "shared_source_window_below_activity_floor"
            continue
        if name == "attack_crest_minus_pre_db":
            value = difference(windows[attack]["crest_db"], windows[other]["crest_db"])
        else:
            numerator = windows[attack]["peak" if "peak" in name else "rms"]
            denominator = windows[other]["peak" if name == "attack_to_sustain_peak_db" else "rms"]
            value = 20 * math.log10(numerator / denominator) if numerator > 0 and denominator > 0 else None
        values[name] = value
        if value is None:
            reasons[name] = "output_or_source_exact_zero_metric"
    return values, reasons


def pair_event(event, observations, source_floor, design):
    onset = event["onset_sample"]
    rate = design["geometry"]["sample_rate"]
    result = {key: event[key] for key in ("event_id", "onset_sample", "onset_seconds", "rise_db")}
    result["windows_samples"] = {key: [onset + ms_samples(t, rate) for t in interval]
                                 for key, interval in design["windows_ms"].items()}
    result["bands"] = {}
    for band in design["bands_hz"]:
        source_windows = observations["source"][band]
        row = {"source_window_active": {key: value["rms"] > source_floor
                                         for key, value in source_windows.items()}}
        for name, values in observations.items():
            ratio_values, exclusions = ratios(values[band], source_windows, source_floor)
            row[name] = {"windows": values[band], "ratios_db": ratio_values,
                         "ratio_null_reasons": exclusions}
        paired = {}
        for window in design["windows_ms"]:
            for metric in ("peak_dbfs", "rms_dbfs", "crest_db"):
                paired[f"{window}_{metric}"] = difference(row["candidate"]["windows"][window][metric],
                                                          row["control"]["windows"][window][metric])
        for metric in RATIO_WINDOWS:
            paired[metric] = difference(row["candidate"]["ratios_db"][metric], row["control"]["ratios_db"][metric])
        row["candidate_minus_control_db"] = paired
        for name in ("control", "candidate"):
            row[f"{name}_minus_source_ratios_db"] = {
                metric: difference(row[name]["ratios_db"][metric], row["source"]["ratios_db"][metric])
                for metric in RATIO_WINDOWS}
        result["bands"][band] = row
    return result


def robust_summary(values, total):
    x = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(x):
        return {"count": 0, "null_count": total}
    quantiles = np.percentile(x, [10, 25, 50, 75, 90])
    return {"count": len(x), "null_count": total - len(x), "mean": float(np.mean(x)),
            "median": float(quantiles[2]), "p10": float(quantiles[0]), "p25": float(quantiles[1]),
            "p75": float(quantiles[3]), "p90": float(quantiles[4]), "min": float(np.min(x)),
            "max": float(np.max(x)), "median_absolute_delta": float(np.median(np.abs(x))),
            "negative_count": int(np.sum(x < -1e-9)), "positive_count": int(np.sum(x > 1e-9)),
            "near_zero_count_at_1e-9_db": int(np.sum(np.abs(x) <= 1e-9))}


def summarize(records, design):
    result = {}
    subsets = {"all": records,
               "prior_listening_excerpt_48_78s": [row for row in records if 48 <= row["onset_seconds"] < 78]}
    for name, rows in subsets.items():
        result[name] = {"paired_event_count": len(rows), "bands": {}}
        for band in design["bands_hz"]:
            band_result = {}
            for group in ("candidate_minus_control_db", "control_minus_source_ratios_db",
                          "candidate_minus_source_ratios_db"):
                metrics = (next(iter(rows))["bands"][band][group].keys() if rows
                           else RATIO_WINDOWS.keys())
                band_result[group] = {metric: robust_summary([row["bands"][band][group][metric] for row in rows], len(rows))
                                      for metric in metrics}
            result[name]["bands"][band] = band_result
    return result


def run(design_path):
    design_path = Path(design_path).resolve()
    output = design_path.parent
    check_latches(output)
    design = json.loads(design_path.read_text())
    invocation = {"schema": "roach-matched-attack-invocation-v1", "started_utc": utc_now(),
                  "argv": sys.argv, "cwd": str(Path.cwd()), "design_path": str(design_path),
                  "design_sha256": sha256(design_path), "script_sha256": sha256(__file__),
                  "test_script_sha256": sha256(Path(__file__).with_name("test_matched_attack_audit.py")),
                  "design_mtime_ns_before_run": design_path.stat().st_mtime_ns,
                  "python": platform.python_version(), "numpy": np.__version__,
                  "scipy_available": importlib.util.find_spec("scipy") is not None,
                  "environment": {key: os.environ.get(key) for key in ("LD_PRELOAD", "OPENBLAS_NUM_THREADS")},
                  "initial_memory": memory_receipt(design["resources"]["process_hwm_aim_mib"]),
                  "analysis_is_speed_experiment": False}
    write_json(output / "invocation.json", invocation)
    readers = {}
    try:
        check_latches(output)
        if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
            raise ValueError("run with OPENBLAS_NUM_THREADS=1")
        pins = {}
        immutable_pins = {str(design_path): invocation["design_sha256"],
                          str(Path(__file__).resolve()): invocation["script_sha256"],
                          str(Path(__file__).with_name("test_matched_attack_audit.py").resolve()): invocation["test_script_sha256"]}
        for name, spec in design["inputs"].items():
            path = Path(spec["path"]).resolve()
            actual = sha256(path)
            if actual != spec["sha256"]:
                raise ValueError(f"hash mismatch for {name}")
            immutable_pins[str(path)] = actual
            readers[name] = WavReader(path)
            pins[name] = {"path": str(path), "sha256": actual,
                          "size_bytes": path.stat().st_size, **readers[name].metadata()}
        verify_geometry(readers, design["geometry"])
        for name, reader in readers.items():
            pins[name]["global_validation"] = validate_audio(reader)
        write_json(output / "source_hashes.json", pins)
        cap = design["resources"]["process_hwm_aim_mib"]
        memory_receipt(cap)
        check_latches(output)
        check_pins(immutable_pins)
        alignment = alignment_audit(readers, design)
        write_json(output / "alignment.json", alignment)
        if not alignment["passed"]:
            raise ValueError("uncertain alignment: predeclared zero-lag anchor gate failed")
        rate = design["geometry"]["sample_rate"]
        taps = design["filter"]["taps"]
        detector = design["detector"]
        rms = input_block_rms(readers["source"], ms_samples(detector["frame_ms"], rate))
        selection = select_events(rms, rate, readers["source"].frames, detector,
                                  design["windows_ms"], taps // 2)
        del rms
        write_json(output / "selection.json", selection)
        if not selection["accepted"]:
            raise ValueError("no accepted source events under the fixed detector")
        source_floor = selection["input_max_block_rms"] * 1e-5
        filters = {name: None if band is None else make_bandpass(rate, band, taps)
                   for name, band in design["bands_hz"].items()}
        records = []
        with (output / "paired_events.jsonl").open("x", encoding="utf-8") as f:
            for index, event in enumerate(selection["accepted"]):
                check_latches(output)
                if index % 25 == 0:
                    check_pins(immutable_pins)
                observations = {name: measure_event(reader, event["onset_sample"], design, filters)
                                for name, reader in readers.items()}
                record = pair_event(event, observations, source_floor, design)
                f.write(json.dumps(record, allow_nan=False) + "\n")
                f.flush()
                records.append(record)
                if (index + 1) % 25 == 0:
                    print(f"matched events retained: {index + 1}; memory={memory_receipt(cap)['observed_hwm_mib']:.1f} MiB", flush=True)
        summary = {"schema": "roach-matched-attack-summary-v1", "design_sha256": invocation["design_sha256"],
                   "paired_event_count": len(records), "identical_event_mask_across_recordings": True,
                   "source_window_activity_floor": source_floor,
                   "delta_direction": "candidate minus control; all values in dB",
                   "level_invariant_fields": list(RATIO_WINDOWS) + [f"{window}_crest_db" for window in design["windows_ms"]],
                   "level_dependent_fields": [f"{window}_{metric}_dbfs" for window in design["windows_ms"] for metric in ("peak", "rms")],
                   "subsets": summarize(records, design), "memory": memory_receipt(cap),
                   "limitations": design["limitations"] + [design["summaries"]["inference"],
                       "Gain invariance applies to ratios with the frozen source mask and nonzero output windows; absolute levels remain reported separately."]}
        write_json(output / "summary.json", summary)
        check_pins(immutable_pins)
        files = ("design.json", "invocation.json", "source_hashes.json", "alignment.json", "selection.json",
                 "paired_events.jsonl", "summary.json")
        check_latches(output)
        write_json(output / "COMPLETE.json", {"schema": "roach-matched-attack-complete-v1",
                    "completed_utc": utc_now(), "paired_event_count": len(records),
                    "artifacts_sha256": {name: sha256(output / name) for name in files},
                    "memory": memory_receipt(cap), "audio_rehashed_unchanged": True,
                    "tools_rehashed_unchanged": True,
                    "frozen_binary_invoked_or_modified": False, "real_pair_runs": 1})
        print(json.dumps({"complete": str(output / "COMPLETE.json"), "paired_events": len(records),
                          "observed_hwm_mib": summary["memory"]["observed_hwm_mib"]}))
    except Exception as error:
        if not (output / "STOP.json").exists():
            write_json(output / "STOP.json", {"stopped_utc": utc_now(), "error_type": type(error).__name__,
                        "error": str(error), "memory": memory_receipt(), "retry_or_retuning_authorized": False})
        raise
    finally:
        for reader in readers.values():
            reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    args = parser.parse_args()
    run(args.design)


if __name__ == "__main__":
    main()
