#!/usr/bin/env python3
"""Deterministic, model-independent controls for rich_mid normalized WOLA.

This is a Python/NumPy logical model of the source windowing, not an emulator
of Rust f32 operation ordering or deployed audio behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def chunk_starts(total_samples: int, chunk_samples: int, overlap_samples: int) -> list[int]:
    """Mirror rich_mid.rs: regular step starts, then right-anchor final chunk."""
    if total_samples < 1 or chunk_samples < 1 or not 0 <= overlap_samples < chunk_samples:
        raise ValueError("require positive lengths and 0 <= overlap < chunk")
    step = chunk_samples - overlap_samples
    starts: list[int] = []
    pos = 0
    while pos + chunk_samples <= total_samples:
        starts.append(pos)
        pos += step
    if pos < total_samples:
        final_start = max(0, total_samples - chunk_samples)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
    if not starts:
        starts.append(0)
    return starts


def source_weights(actual_chunk_len: int, overlap_samples: int, idx: int, n_chunks: int) -> np.ndarray:
    """Mirror the two independent fades from rich_mid.rs, in float64 math."""
    w = np.ones(actual_chunk_len, dtype=np.float64)
    if idx > 0:
        fade_in_len = min(overlap_samples, actual_chunk_len)
        for j in range(fade_in_len):
            theta = (math.pi / 2.0) * (j / fade_in_len)
            w[j] = math.sin(theta) ** 2
    if idx < n_chunks - 1 and actual_chunk_len > overlap_samples:
        fade_out_len = overlap_samples
        start_fade = actual_chunk_len - fade_out_len
        for j in range(fade_out_len):
            theta = (math.pi / 2.0) * (j / fade_out_len)
            w[start_fade + j] = math.cos(theta) ** 2
    return w


def normalized_wola(chunks: list[np.ndarray], starts: list[int], total_samples: int,
                    overlap_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Blend 1-D or [channels,samples] chunks and return signal, normalization."""
    if len(chunks) != len(starts) or not chunks:
        raise ValueError("chunks and starts must be nonempty and have equal lengths")
    first = np.asarray(chunks[0], dtype=np.float64)
    stereo = first.ndim == 2
    channels = first.shape[0] if stereo else None
    out = np.zeros((channels, total_samples) if stereo else total_samples, dtype=np.float64)
    denom = np.zeros(total_samples, dtype=np.float64)
    for idx, (chunk, start) in enumerate(zip(chunks, starts)):
        x = np.asarray(chunk, dtype=np.float64)
        length = x.shape[-1]
        if start < 0 or start + length > total_samples:
            raise ValueError("chunk extends outside output")
        if x.ndim != first.ndim or (stereo and x.shape[0] != channels):
            raise ValueError("all chunks must have consistent channel shape")
        w = source_weights(length, overlap_samples, idx, len(chunks))
        if stereo:
            out[:, start:start + length] += x * w[None, :]
        else:
            out[start:start + length] += x * w
        denom[start:start + length] += w
    nz = denom > 1e-8
    if stereo:
        out[:, nz] /= denom[nz]
    else:
        out[nz] /= denom[nz]
    return out, denom


def spectral_project(x: np.ndarray, rate: int, keep_low_hz: float,
                     keep_high_hz: float) -> np.ndarray:
    """Periodic finite-clip projection retaining only strict low < f < high."""
    x = np.asarray(x, dtype=np.float64)
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / rate)
    mask = (f <= keep_low_hz) | (f >= keep_high_hz)
    spec = np.fft.rfft(x, axis=-1)
    spec[..., mask] = 0.0
    return np.fft.irfft(spec, n=x.shape[-1], axis=-1)


def forbidden_energy_ratio(x: np.ndarray, rate: int, low_hz: float,
                           high_hz: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    spec = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / rate)
    # rFFT bins represent conjugate pairs; DC and (for even N) Nyquist have
    # multiplicity one. This also handles odd N, which has no Nyquist bin.
    multiplicity = np.full(f.shape, 2.0, dtype=np.float64)
    multiplicity[0] = 1.0
    if x.shape[-1] % 2 == 0:
        multiplicity[-1] = 1.0
    power = np.abs(spec) ** 2 * multiplicity
    forbidden = (f <= low_hz) | (f >= high_hz)
    denom = float(np.sum(power))
    return float(np.sum(power[..., forbidden]) / denom) if denom else 0.0


def run_controls(source_path: Path, seed: int = 20261018) -> dict:
    rng = np.random.default_rng(seed)
    # Identity and same-residual controls with deliberately irregular terminal anchor.
    total, chunk, overlap = 1013, 401, 101
    starts = chunk_starts(total, chunk, overlap)
    signal = rng.normal(size=(2, total)).astype(np.float64)
    views = [signal[:, s:min(s + chunk, total)] for s in starts]
    identity, weights = normalized_wola(views, starts, total, overlap)
    identity_err = float(np.max(np.abs(identity - signal)))
    coherent_resid = rng.normal(scale=0.03, size=(2, total))
    coherent_views = [coherent_resid[:, s:min(s + chunk, total)] for s in starts]
    coherent, _ = normalized_wola(coherent_views, starts, total, overlap)
    coherent_err = float(np.max(np.abs(coherent - coherent_resid)))

    # Equal-weight variance result: deterministic orthogonal, centered, equal-energy vectors.
    n = 4096
    a = np.linspace(-1.0, 1.0, n, endpoint=False)
    a -= a.mean()
    b = np.roll(a, n // 4)
    b -= b.mean()
    b -= a * (np.dot(a, b) / np.dot(a, a))
    b *= math.sqrt(np.dot(a, a) / np.dot(b, b))
    midpoint = 0.5 * (a + b)
    variance_ratio = float(np.var(midpoint) / np.var(a))

    # Band-locked finite chunks; after WOLA, inspect a different whole-track DFT grid.
    rate, n_total, c_len, o_len = 24000, 12037, 4096, 1024
    c_starts = chunk_starts(n_total, c_len, o_len)
    raw = rng.normal(size=n_total)
    chunk_resids = []
    chunk_ratios = []
    for s in c_starts:
        seg = raw[s:min(s + c_len, n_total)]
        resid = spectral_project(seg, rate, 1500.0, 8000.0)
        chunk_resids.append(resid)
        chunk_ratios.append(forbidden_energy_ratio(resid, rate, 1500.0, 8000.0))
    stitched, stitch_weights = normalized_wola(chunk_resids, c_starts, n_total, o_len)
    whole_ratio = forbidden_energy_ratio(stitched, rate, 1500.0, 8000.0)
    reprojected = spectral_project(stitched, rate, 1500.0, 8000.0)
    projected_ratio = forbidden_energy_ratio(reprojected, rate, 1500.0, 8000.0)
    projection_change = float(np.linalg.norm(reprojected - stitched) /
                              max(np.linalg.norm(stitched), 1e-30))

    source_bytes = source_path.read_bytes()
    return {
        "schema": "wola-controls-v1",
        "classification": "deterministic mathematical control; not audio-quality evidence",
        "provenance": {
            "seed": seed,
            "numpy_version": np.__version__,
            "python_logical_control": True,
            "rust_f32_rounding_emulated": False,
            "source_path": str(source_path),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "source_window_mirror": {
            "rule": "regular starts by chunk-overlap; if next regular position is before total end, append max(0,total-chunk) unless duplicate",
            "weight_rule": "per-chunk ones; nonfirst prefix sin^2(pi/2*j/fade_len); nonlast suffix cos^2(pi/2*j/overlap); divide accumulated signal by accumulated weights",
            "identity_geometry": {"total": total, "chunk": chunk, "overlap": overlap,
                                  "starts": starts,
                                  "intersections": [[max(starts[i], starts[i+1]),
                                                     min(starts[i] + chunk, starts[i+1] + chunk)]
                                                    for i in range(len(starts)-1)],
                                  "weight_min": float(weights.min()),
                                  "weight_max": float(weights.max())},
        },
        "identity": {"max_abs_error": identity_err, "tolerance": 2e-14,
                     "pass": identity_err <= 2e-14},
        "identical_coherent_residual": {"max_abs_error": coherent_err,
                                         "tolerance": 2e-14,
                                         "pass": coherent_err <= 2e-14},
        "orthogonal_equal_variance_midpoint": {
            "correlation": float(np.corrcoef(a, b)[0, 1]),
            "variance_ratio": variance_ratio,
            "variance_ratio_expected": 0.5,
            "level_delta_db": 10.0 * math.log10(variance_ratio),
            "level_delta_db_expected": 10.0 * math.log10(0.5),
            "interpretation": "toy equal-weight uncorrelated residual midpoint; zero covariance is sufficient for this variance identity, not a claim of statistical independence or a measured audio artifact",
        },
        "chunk_lock_vs_whole_track": {
            "seed": seed,
            "sample_rate_hz": rate,
            "protected_bands_hz": [[0.0, 1500.0], [8000.0, rate / 2]],
            "protected_cutoff_rule": "closed protected bands; residual retained only for strict 1500 < f < 8000 Hz",
            "total_samples": n_total,
            "chunk_samples": c_len,
            "overlap_samples": o_len,
            "starts": c_starts,
            "chunk_error_max_forbidden_energy_ratio": max(chunk_ratios),
            "chunk_error_max_forbidden_energy_db": 10.0 * math.log10(max(max(chunk_ratios), 1e-300)),
            "whole_track_error_forbidden_energy_ratio": whole_ratio,
            "whole_track_error_forbidden_energy_db": 10.0 * math.log10(max(whole_ratio, 1e-300)),
            "whole_track_error_after_global_projection_ratio": projected_ratio,
            "whole_track_error_after_global_projection_db": 10.0 * math.log10(max(projected_ratio, 1e-300)),
            "global_projection_relative_waveform_change": projection_change,
            "interpretation": "per-chunk Fourier locks can leak on the whole-track grid after time-window stitching; global projection is only a numerical counterexample and may alter temporal behavior",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh output directory; existing paths are rejected")
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "src/rich_mid.rs")
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = run_controls(args.source, args.seed)
    (args.out / "wola_controls.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
