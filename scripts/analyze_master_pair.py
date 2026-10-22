#!/usr/bin/env python3
"""Bounded, local, descriptive analysis of a same-geometry source/master pair.

No waveform shift fitting, resampling, perceptual score, LUFS or true-peak
estimation. The two recordings must share rate/channels/frame count. Analysis
constants below are the preregistration; hash this file before analyzing songs.
Welch-style PSD: Peter D. Welch, "The Use of Fast Fourier Transform for the
Estimation of Power Spectra: A Method Based on Time Averaging Over Short,
Modified Periodograms" (1967), https://doi.org/10.1109/TAU.1967.1161901 .
This implements ordinary averaged periodograms, not the paper's experiments.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

try:
    from scripts.matched_attack_audit import WavReader, check_latches, memory_receipt, sha256
except ModuleNotFoundError:
    from matched_attack_audit import WavReader, check_latches, memory_receipt, sha256


DEFINITIONS = {
    "version": 1,
    "amplitude_floor": 1e-12,
    "near_fullscale_absolute_threshold": 0.999,
    "envelope_ms": 10,
    "transient_window_ms": 100,
    "contrast_window_ms": 1000,
    "grid": "round(k * sample_rate * seconds), origin sample zero; final partial retained",
    "percentiles": "NumPy percentile method=linear, descriptive time-window distribution",
    "contrast": "p90 minus p10 of dBFS RMS of complete 1 s windows above amplitude floor",
    "transient_proxy": "peak/RMS crest in source-fixed 100 ms windows; never an attack fidelity score",
    "excerpt_selection": "up to 3 largest positive successive complete 100 ms source RMS rises; "
                         "both above floor; descending rise then earliest index; >=2 s separation; "
                         "fallback first eligible complete window; +/-50 ms excerpt",
    "spectrum": {"nfft": 4096, "hop": 2048, "window": "np.hanning(4096), symmetric Hann",
                 "detrend": False, "padding": False, "channels": "mean of channel powers",
                 "normalization": "one-sided PSD, |FFT|^2/(sample_rate*sum(window^2)); "
                                  "double interior bins; mean over complete windows",
                 "bands_hz": [[20, 120], [120, 500], [500, 2000], [2000, 8000], [8000, 20000]],
                 "band_edges": "lower inclusive, upper exclusive; truncate at Nyquist",
                 "bass_ms_band_hz": [20, 80],
                 "bass_ms": "M=(L+R)/2 and S=(L-R)/2 complex FFT; same one-sided PSD scaling; "
                            "M power plus S power equals pooled channel power; coarse energy diagnostic only",
                 "broadband": "all available one-sided bins including DC and Nyquist"},
    "spectral_shape_normalization": "band/bin raw power dB shift minus whole-track pooled RMS dB gain; never least-squares gain",
    "gain": "g=max(0,sum(source*master)/sum(source^2)), pooled channels, no intercept; "
            "zero-source denominator -> null; zero is constrained boundary if dot<=0",
    "residual": "master-g*source, no alignment fit; RMS uses explicit second streaming pass",
    "correlation": "pooled Pearson after separate per-channel DC centering, original sample grid",
    "mid_side": "M=(L+R)/2, S=(L-R)/2; side fraction=S_energy/(M_energy+S_energy)",
    "nulls": "dB and ratios null if any amplitude operand<=1e-12 (power<=1e-24); "
             "no epsilon inserted into ratios; null windows remain in raw data",
    "memory_cap_mib": 384,
}
FLOOR = DEFINITIONS["amplitude_floor"]
WARNINGS = [
    "Descriptive change only: residual magnitude and waveform correlation do not measure distortion or fidelity.",
    "Same sample grid is assumed; no fitted alignment or strict alignment rejection. Phase, filtering, "
    "delay and other legitimate processing can reduce correlation and increase residuals.",
    "RMS dBFS is not LUFS. No true-peak estimate or continuous-time peak guarantee is supplied here; "
    "use the separate native/FFmpeg metering receipts.",
    "Time windows within one song are dependent observations, not independent replications or confidence intervals.",
    "Crest and energy-rise are waveform proxies; greater brightness, level or crest does not establish audible improvement.",
    "Near-fullscale counts are an amplitude-threshold proxy, not proof of clipping or clipped-waveform recovery.",
    "20-80 Hz mid/side power uses coarse 4096-point FFT bins (about 11.7 Hz, five bins at 48 kHz); "
    "this is not true coherence or a mono guarantee. Reduced side energy is not automatically an improvement; "
    "bass width may be intentional.",
]


def db_amp(value):
    return 20 * math.log10(value) if value > FLOOR else None


def db_ratio(numerator, denominator, power=False):
    floor = FLOOR ** 2 if power else FLOOR
    if numerator <= floor or denominator <= floor:
        return None
    return (10 if power else 20) * math.log10(numerator / denominator)


def delta(after, before):
    return None if after is None or before is None else after - before


def distribution(values):
    values = list(values)
    good = np.asarray([v for v in values if v is not None], dtype=np.float64)
    result = {"count": len(values), "valid_count": len(good), "null_count": len(values) - len(good)}
    for label, q in (("min", 0), ("p10", 10), ("median", 50), ("p90", 90), ("max", 100)):
        result[label] = float(np.percentile(good, q, method="linear")) if len(good) else None
    return result


def moments(x):
    return {"n": len(x), "sum": np.sum(x, axis=0), "squares": np.sum(x * x, axis=0),
            "cross": float(np.sum(x[:, 0] * x[:, 1])),
            "peak": float(np.max(np.abs(x))),
            "near_fullscale": np.sum(np.abs(x) >= DEFINITIONS["near_fullscale_absolute_threshold"], axis=0)}


def empty_moments():
    return {"n": 0, "sum": np.zeros(2), "squares": np.zeros(2), "cross": 0., "peak": 0.,
            "near_fullscale": np.zeros(2, dtype=np.int64)}


def add_moments(total, part):
    for key in total:
        total[key] = max(total[key], part[key]) if key == "peak" else total[key] + part[key]


def level(m):
    rms = math.sqrt(float(np.sum(m["squares"])) / (2 * m["n"]))
    return {"rms": rms, "rms_dbfs": db_amp(rms), "sample_peak": m["peak"],
            "sample_peak_dbfs": db_amp(m["peak"]), "crest_db": db_ratio(m["peak"], rms)}


def global_metrics(m):
    result = level(m)
    sums, squares, n = m["sum"], m["squares"], m["n"]
    centered = np.maximum(squares - sums * sums / n, 0)
    denominator = math.sqrt(float(centered[0] * centered[1]))
    lr = (m["cross"] - float(sums[0] * sums[1]) / n) / denominator if denominator > FLOOR ** 2 * n else None
    mid = max(0., float(np.sum(squares) + 2 * m["cross"]) / (4 * n))
    side = max(0., float(np.sum(squares) - 2 * m["cross"]) / (4 * n))
    result.update(dc_per_channel=(sums / n).tolist(),
                  near_fullscale_counts_per_channel=m["near_fullscale"].tolist(),
                  lr_correlation=None if lr is None else float(np.clip(lr, -1, 1)),
                  mid_mean_square=mid, side_mean_square=side,
                  side_to_mid_db=db_ratio(side, mid, power=True),
                  side_energy_fraction=side / (mid + side) if mid + side > FLOOR ** 2 else None)
    return result


def sample_grid(frames, rate, milliseconds):
    k, start = 0, 0
    while start < frames:
        k += 1
        nominal_end = round(k * rate * milliseconds / 1000)
        end = min(frames, nominal_end)
        if end <= start:
            raise ValueError("sample rate too low for analysis grid")
        yield start, end, nominal_end <= frames
        start = end


def read_bounded(reader, start, end):
    # Excerpts and coarse windows may exceed WavReader's 65536-frame bound.
    if end - start <= 65536:
        return reader.read(start, end)
    raise ValueError("analysis window exceeds bounded read; rates above 655360 Hz unsupported")


def select_excerpts(rows):
    candidates = []
    for i in range(1, len(rows)):
        row, previous = rows[i], rows[i - 1]
        rise = db_ratio(row["source_rms"], previous["source_rms"])
        if row["complete"] and previous["complete"] and rise is not None and rise > 0:
            candidates.append((rise, i))
    selected = []
    for rise, i in sorted(candidates, key=lambda v: (-v[0], v[1])):
        if all(abs(rows[i]["start_seconds"] - rows[j]["start_seconds"]) >= 2 for _, j in selected):
            selected.append((rise, i))
        if len(selected) == 3:
            break
    if not selected:
        eligible = [i for i, row in enumerate(rows) if row["complete"] and row["source_rms"] > FLOOR]
        if eligible:
            selected = [(None, eligible[0])]
    return [{"window_index": i, "start_seconds": rows[i]["start_seconds"], "source_rise_db": rise}
            for rise, i in selected]


def spectrum(source, master, rms_gain_db, latch=lambda: None):
    nfft, hop = 4096, 2048
    window = np.hanning(nfft)
    accum = [np.zeros(nfft // 2 + 1), np.zeros(nfft // 2 + 1)]
    accum_ms = [[np.zeros(nfft // 2 + 1), np.zeros(nfft // 2 + 1)] for _ in range(2)]
    count = 0
    # Batch complete windows in <=65536 source frames; never a whole-song array.
    batch_windows = 24
    starts = range(0, source.frames - nfft + 1, hop)
    for base in range(0, len(starts), batch_windows):
        latch()
        count_batch = min(batch_windows, len(starts) - base)
        lo = starts[base]
        hi = lo + (count_batch - 1) * hop + nfft
        for index, reader in enumerate((source, master)):
            audio = read_bounded(reader, lo, hi)
            view = np.lib.stride_tricks.sliding_window_view(audio, nfft, axis=0)[::hop]
            transformed = np.fft.rfft(view * window, axis=-1)
            power = np.mean(np.abs(transformed) ** 2, axis=1)
            accum[index] += np.sum(power, axis=0)
            mid = (transformed[:, 0, :] + transformed[:, 1, :]) / 2
            side = (transformed[:, 0, :] - transformed[:, 1, :]) / 2
            accum_ms[index][0] += np.sum(np.abs(mid) ** 2, axis=0)
            accum_ms[index][1] += np.sum(np.abs(side) ** 2, axis=0)
        count += count_batch
    factor = np.full(nfft // 2 + 1, 2. / (source.rate * np.sum(window ** 2)))
    factor[[0, -1]] /= 2
    curves = [a * factor / count if count else np.zeros_like(a) for a in accum]
    freq = np.fft.rfftfreq(nfft, 1 / source.rate)
    total = [float(np.sum(a) * source.rate / nfft) for a in curves]
    bass_mask = (freq >= 20) & (freq < 80)
    bass_ms = {"low_hz": 20, "high_hz": 80, "frequency_resolution_hz": source.rate / nfft,
               "available_bin_count": int(np.sum(bass_mask)),
               "scope": "coarse bin-integrated energy, not coherence or a mono guarantee; "
                        "reduced side energy is not automatically better; intentional width may be affected"}
    for index, label in enumerate(("source", "master")):
        mid_energy, side_energy = [float(np.sum(a[bass_mask] * factor[bass_mask]) / count * source.rate / nfft)
                                   if count else None for a in accum_ms[index]]
        pooled_energy = float(np.sum(curves[index][bass_mask]) * source.rate / nfft) if count else None
        has_energy = count and mid_energy + side_energy > FLOOR ** 2
        bass_ms[label] = {"pooled_mean_square": pooled_energy, "mid_mean_square": mid_energy,
                          "side_mean_square": side_energy,
                          "side_energy_fraction": side_energy / (mid_energy + side_energy) if has_energy else None,
                          "side_to_mid_db": db_ratio(side_energy, mid_energy, power=True) if count else None}
    bands = []
    for low, high in DEFINITIONS["spectrum"]["bands_hz"]:
        mask = (freq >= low) & (freq < high)
        energy = [float(np.sum(a[mask]) * source.rate / nfft) for a in curves]
        shift = db_ratio(energy[1], energy[0], power=True) if count else None
        bands.append({"low_hz": low, "high_hz": high, "available_bin_count": int(np.sum(mask)),
                      "source_mean_square": energy[0] if count else None,
                      "master_mean_square": energy[1] if count else None,
                      "raw_energy_shift_db": shift,
                      "broadband_normalized_shift_db": delta(shift, rms_gain_db)})
    rows = []
    for i, hz in enumerate(freq):
        shift = db_ratio(curves[1][i], curves[0][i], power=True) if count else None
        rows.append({"frequency_hz": float(hz), "source_psd": float(curves[0][i]) if count else None,
                     "master_psd": float(curves[1][i]) if count else None, "raw_shift_db": shift,
                     "broadband_normalized_shift_db": delta(shift, rms_gain_db)})
    return {"window_count": count, "trailing_unanalyzed_frames": source.frames - (starts[-1] + nfft) if count else source.frames,
            "source_broadband_mean_square": total[0] if count else None,
            "master_broadband_mean_square": total[1] if count else None,
            "normalization_pooled_rms_gain_db": rms_gain_db,
            "bass_20_80_ms": bass_ms,
            "bands": bands}, rows


def analyze(source, master, latch=lambda: None):
    if (source.rate, source.channels, source.frames) != (master.rate, master.channels, master.frames):
        raise ValueError("source/master must have identical sample rate, channels and frame count; no alignment fitting")
    if source.channels != 2 or source.frames == 0 or source.rate < 100:
        raise ValueError("nonempty stereo audio and sample rate >=100 required")
    totals = [empty_moments(), empty_moments()]
    envelope, transient, blocks = [], [], []
    dot = 0.
    # Full-track moments and exact fixed 10 ms/100 ms/1 s boundaries share the
    # 10 ms grid. Require integer 10 ms sample count so coarser sums are exact.
    if source.rate % 100:
        raise ValueError("sample rate must be divisible by 100 for exact nested window grids")
    partial100, partial1000 = [[empty_moments(), empty_moments()] for _ in range(2)]
    raw_difference_energy = 0.
    for k, (start, end, complete) in enumerate(sample_grid(source.frames, source.rate, 10)):
        if k % 100 == 0:
            latch()
        x, y = (read_bounded(r, start, end) for r in (source, master))
        parts = [moments(x), moments(y)]
        dot += float(np.sum(x * y))
        raw_difference_energy += float(np.sum((y - x) ** 2))
        levels = [level(p) for p in parts]
        envelope.append({"start_sample": start, "end_sample": end, "start_seconds": start / source.rate,
                         "complete": complete,
                         **{f"{label}_{key}": lev[key] for label, lev in zip(("source", "master"), levels)
                            for key in ("rms", "sample_peak", "crest_db")}})
        for i, part in enumerate(parts):
            add_moments(totals[i], part)
            add_moments(partial100[i], part)
            add_moments(partial1000[i], part)
        for span, accum, rows in ((10, partial100, transient), (100, partial1000, blocks)):
            if (k + 1) % span == 0 or end == source.frames:
                first = end - accum[0]["n"]
                row = {"start_sample": first, "end_sample": end, "start_seconds": first / source.rate,
                       "complete": accum[0]["n"] == source.rate * span // 100}
                for label, a in zip(("source", "master"), accum):
                    row.update({f"{label}_{key}": value for key, value in level(a).items()})
                row.update(raw_rms_delta_db=delta(row["master_rms_dbfs"], row["source_rms_dbfs"]),
                           crest_delta_db=delta(row["master_crest_db"], row["source_crest_db"]))
                rows.append(row)
                accum[:] = [empty_moments(), empty_moments()]
    metrics = [global_metrics(total) for total in totals]
    source_energy = float(np.sum(totals[0]["squares"]))
    gain = max(0., dot / source_energy) if source_energy > source.frames * 2 * FLOOR ** 2 else None
    # Explicit subtraction avoids catastrophic cancellation for identity/gain controls.
    residual_energy = 0.
    if gain is not None:
        for start in range(0, source.frames, 65536):
            latch()
            end = min(start + 65536, source.frames)
            x, y = (read_bounded(r, start, end) for r in (source, master))
            residual_energy += float(np.sum((y - gain * x) ** 2))
    centered_dot = dot - float(np.sum(totals[0]["sum"] * totals[1]["sum"])) / source.frames
    centered_energy = [max(0., float(np.sum(t["squares"] - t["sum"] ** 2 / t["n"]))) for t in totals]
    corr_den = math.sqrt(centered_energy[0] * centered_energy[1])
    corr = float(np.clip(centered_dot / corr_den, -1, 1)) if corr_den > source.frames * 2 * FLOOR ** 2 else None
    residual = math.sqrt(residual_energy / (2 * source.frames)) if gain is not None else None
    paired = {"best_nonnegative_global_gain": gain, "gain_constraint_boundary": gain == 0 if gain is not None else None,
              "unconstrained_gain": dot / source_energy if gain is not None else None,
              "gain_db": db_amp(gain) if gain is not None else None,
              "waveform_correlation_per_channel_dc_removed": corr,
              "raw_difference_rms": math.sqrt(raw_difference_energy / (2 * source.frames)),
              "gain_adjusted_residual_rms": residual,
              "residual_to_master_rms_db": db_ratio(residual, metrics[1]["rms"]) if residual is not None else None,
              "global_rms_delta_db": delta(metrics[1]["rms_dbfs"], metrics[0]["rms_dbfs"]),
              "global_crest_delta_db": delta(metrics[1]["crest_db"], metrics[0]["crest_db"]),
              "transient_crest_delta_db_distribution": distribution(r["crest_delta_db"] for r in transient if r["complete"])}
    for label in ("source", "master"):
        contrast = distribution(r[f"{label}_rms_dbfs"] for r in blocks if r["complete"])
        contrast["p90_minus_p10_db"] = delta(contrast["p90"], contrast["p10"])
        paired[f"{label}_one_second_rms_dbfs_distribution"] = contrast
        paired[f"{label}_100ms_crest_db_distribution"] = distribution(r[f"{label}_crest_db"] for r in transient if r["complete"])
    spectral, curve = spectrum(source, master, paired["global_rms_delta_db"], latch)
    summary = {"geometry": {"sample_rate": source.rate, "channels": 2, "frames": source.frames,
                             "duration_seconds": source.frames / source.rate},
               "source": metrics[0], "master": metrics[1], "paired": paired,
               "spectrum": spectral, "excerpts": select_excerpts(transient), "warnings": WARNINGS}
    return summary, {"envelope_10ms": envelope, "transient_100ms": transient, "blocks_1s": blocks, "spectrum": curve}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def svg_plot(title, series, xlabel, ylabel, width=1000, height=250):
    points = [(float(x), float(y)) for _, _, values in series for x, y in values if y is not None and math.isfinite(y)]
    if not points:
        return f"<h3>{html.escape(title)}</h3><p>No defined values.</p>"
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    def coordinate(x, y):
        return f"{65+(x-xmin)/(xmax-xmin)*(width-90):.2f},{height-40-(y-ymin)/(ymax-ymin)*(height-75):.2f}"
    pieces = [f'<h3>{html.escape(title)}</h3><svg role="img" aria-label="{html.escape(title)}" '
              f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/>',
              f'<path d="M65,35 V{height-40} H{width-25}" stroke="#333" fill="none"/>']
    for name, color, values in series:
        # Preserve gaps: never join across null spectrum bins or silent windows.
        segments, current = [], []
        for x, y in values:
            if y is None or not math.isfinite(y):
                if current:
                    segments.append(current)
                current = []
            else:
                current.append(coordinate(x, y))
        if current:
            segments.append(current)
        for segment in segments:
            pieces.append(f'<polyline points="{" ".join(segment)}" stroke="{color}" stroke-width="1" fill="none"/>')
    pieces += [f'<text x="65" y="20">{ymin:.4g} to {ymax:.4g} {html.escape(ylabel)}</text>',
               f'<text x="65" y="{height-10}">{xmin:.4g} to {xmax:.4g} {html.escape(xlabel)}</text>', '</svg>',
               '<p>' + ' / '.join(f'<span style="color:{color}">{html.escape(name)}</span>' for name, color, _ in series) + '</p>']
    return "".join(pieces)


def make_report(summary, raw, source, master):
    out = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Local source/master analysis</title>',
           '<style>body{font:16px sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}svg{width:100%;border:1px solid #ddd}pre{white-space:pre-wrap}</style>',
           '<h1>Local source/master analysis</h1>',
           '<p>NumPy analysis; standalone SVG plots. Matplotlib unavailable or unused. No network resources. '
           'All metrics describe signal changes; listen to assess preference.</p>']
    out += [f'<p>{html.escape(warning)}</p>' for warning in WARNINGS]
    colors = (("source", "#2464ac"), ("master", "#c33d32"))
    for metric in ("sample_peak", "rms"):
        series = [(label, color, [(r["start_seconds"], db_amp(r[f"{label}_{metric}"])) for r in raw["envelope_10ms"]])
                  for label, color in colors]
        out.append(svg_plot(f"Full-track 10 ms {metric} envelope (raw level)", series, "seconds", "dBFS"))
    out.append(svg_plot("Broadband-normalized spectral energy change", [("master minus source", "#833a9a",
        [(r["frequency_hz"], r["broadband_normalized_shift_db"]) for r in raw["spectrum"] if 20 <= r["frequency_hz"] <= 20000])], "Hz (linear)", "dB"))
    series = []
    for label, color in colors:
        values = sorted(r[f"{label}_crest_db"] for r in raw["transient_100ms"] if r["complete"] and r[f"{label}_crest_db"] is not None)
        series.append((label, color, [(v, (i + 1) / len(values)) for i, v in enumerate(values)]))
    out.append(svg_plot("Complete 100 ms crest distribution (empirical CDF)", series, "crest dB", "cumulative fraction"))
    for excerpt in summary["excerpts"]:
        center = round(excerpt["start_seconds"] * source.rate)
        start, end = max(0, center - round(.05 * source.rate)), min(source.frames, center + round(.05 * source.rate))
        series = []
        for label, color, reader in (("source", "#2464ac", source), ("master", "#c33d32", master)):
            audio = read_bounded(reader, start, end)
            # Every sample is retained. Separate channel plots prevent mono cancellation.
            for channel in range(2):
                series.append((f"{label} {'LR'[channel]}", color if channel == 0 else ("#6a99cd" if label == "source" else "#dc8a7d"),
                               [(i / source.rate, float(value)) for i, value in enumerate(audio[:, channel], start)]))
        out.append(svg_plot(f"Source-selected excerpt at {excerpt['start_seconds']:.3f} s, raw amplitude", series, "seconds", "amplitude"))
    out += ['<h2>Summary and definitions</h2><pre>', html.escape(json.dumps(summary, indent=2, allow_nan=False)), '</pre></html>']
    return "\n".join(out)


def compare_summaries(previous, current):
    """Only descriptive scalar distributions; no cross-version sample pairing."""
    result = {"scope": "descriptive independent recording summaries; no sample correlation, subtraction, "
                       "alignment or significance test between versions; different lengths/content may confound differences",
              "previous_geometry": previous["geometry"], "current_geometry": current["geometry"], "differences": {}}
    for label in ("source", "master"):
        result["differences"][label] = {key: delta(current[label][key], previous[label][key]) for key in
                                       ("rms_dbfs", "sample_peak_dbfs", "crest_db", "side_energy_fraction", "lr_correlation")}
    for key in ("global_rms_delta_db", "global_crest_delta_db"):
        result["differences"][f"difference_in_within_pair_{key}"] = delta(current["paired"][key], previous["paired"][key])
    for label in ("source", "master"):
        for suffix, metric in (("one_second_rms_dbfs_distribution", "p90_minus_p10_db"),
                               ("100ms_crest_db_distribution", "median")):
            key = f"{label}_{suffix}"
            result["differences"][f"{key}_{metric}"] = delta(current["paired"][key][metric], previous["paired"][key][metric])
    result["difference_in_within_pair_band_shifts"] = [
        {"low_hz": new["low_hz"], "high_hz": new["high_hz"],
         "raw_shift_difference_db": delta(new["raw_energy_shift_db"], old["raw_energy_shift_db"]),
         "normalized_shift_difference_db": delta(new["broadband_normalized_shift_db"], old["broadband_normalized_shift_db"])}
        for old, new in zip(previous["spectrum"]["bands"], current["spectrum"]["bands"])
        if (old["low_hz"], old["high_hz"]) == (new["low_hz"], new["high_hz"])]
    return result


def check_output_latches(out):
    check_latches(out)
    check_latches(Path(out).parent)


def prepare_source(path, out, latch):
    """Freeze excerpt IDs without opening or inspecting any master."""
    source = WavReader(path)
    try:
        if source.frames == 0 or source.rate % 100:
            raise ValueError("nonempty source and sample rate divisible by 100 required")
        source_hash = sha256(path)
        rows = []
        for start, end, complete in sample_grid(source.frames, source.rate, 100):
            latch()
            value = level(moments(read_bounded(source, start, end)))
            rows.append({"start_seconds": start / source.rate, "complete": complete, "source_rms": value["rms"]})
        if sha256(path) != source_hash:
            raise ValueError("source changed during excerpt preparation")
        write_json(out / "excerpts.json", {"source_sha256": source_hash, "source": source.metadata(),
                   "excerpts": select_excerpts(rows), "selection": DEFINITIONS["excerpt_selection"],
                   "analyzer_sha256": sha256(__file__), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        print(json.dumps({"source_only_excerpt_plan": str(out / "excerpts.json")}))
    finally:
        source.close()


def run(args):
    out = Path(args.out)
    check_output_latches(out)
    out.mkdir(parents=True, exist_ok=False)
    latch = lambda: check_output_latches(out)
    source = master = None
    try:
        latch()
        write_json(out / "definitions.json", DEFINITIONS)
        if args.prepare_source:
            prepare_source(args.source, out, latch)
            return
        if not args.master:
            raise ValueError("--master is required unless --prepare-source")
        source = WavReader(args.source)
        master = WavReader(args.master)
        pins = {str(Path(path).resolve()): sha256(path) for path in (args.source, args.master, __file__, Path(__file__).with_name("matched_attack_audit.py"))}
        summary, raw = analyze(source, master, latch)
        if args.excerpt_plan:
            plan = json.loads(Path(args.excerpt_plan).read_text())
            if (plan["source_sha256"] != pins[str(Path(args.source).resolve())]
                    or plan["excerpts"] != summary["excerpts"]
                    or plan["analyzer_sha256"] != pins[str(Path(__file__).resolve())]):
                raise ValueError("source excerpt plan differs from the source hash or deterministic selection")
            summary["excerpt_plan"] = {"path": str(Path(args.excerpt_plan).resolve()), "sha256": sha256(args.excerpt_plan)}
        summary.update(definitions=DEFINITIONS, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       provenance={"source": source.metadata(), "master": master.metadata(), "sha256": pins,
                                   "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                                   "matplotlib_available": importlib.util.find_spec("matplotlib") is not None,
                                   "plot_backend": "local standalone SVG/HTML", "argv": sys.argv})
        for name, rows in raw.items():
            latch()
            write_csv(out / f"{name}.csv", rows)
        latch()
        report = make_report(summary, raw, source, master)
        with (out / "report.html").open("x", encoding="utf-8") as handle:
            handle.write(report)
        if args.compare_summary:
            previous = json.loads(Path(args.compare_summary).read_text())
            write_json(out / "cross_version_descriptive.json", compare_summaries(previous, summary))
        latch()
        for path, expected in pins.items():
            if sha256(path) != expected:
                raise ValueError(f"input/tool changed during analysis: {path}")
        summary["memory"] = memory_receipt(DEFINITIONS["memory_cap_mib"])
        write_json(out / "summary.json", summary)
        print(json.dumps({"summary": str(out / "summary.json"), "report": str(out / "report.html"), "memory": summary["memory"]}))
    except Exception as exc:
        if not (out / "FAILED.json").exists():
            write_json(out / "FAILED.json", {"error": str(exc), "type": type(exc).__name__})
        raise
    finally:
        if source is not None:
            source.close()
        if master is not None:
            master.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--master")
    parser.add_argument("--out", required=True, help="fresh local artifact directory; existing paths are rejected")
    parser.add_argument("--compare-summary", help="optional earlier version summary for descriptive scalar differences only")
    parser.add_argument("--prepare-source", action="store_true", help="source-only excerpt selection; no master opened; fresh --out required")
    parser.add_argument("--excerpt-plan", help="source-only excerpts.json to verify during paired analysis")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
