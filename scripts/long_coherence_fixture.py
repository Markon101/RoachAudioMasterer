#!/usr/bin/env python3
"""Make a deterministic long procedural stereo fixture for coherence work.

The complete physical source is synthesized and degraded before any views are
cut. NumPy's float64 FFT is a close recipe match to Rust dsp::lowpass, not a
bitwise reproduction of its f32 rustfft implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

RATE = 48_000
DEFAULT_SEED = 881_000
DEFAULT_EVENTS = (9.5, 12.5, 15.5, 18.5)
_VIEW_SECONDS = 26.0
_VIEW_SHIFT_SAMPLES = 564 * 256  # 3.008s; same native STFT hop grid.


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_float_wav(path: Path, samples: np.ndarray) -> None:
    """Write canonical stereo IEEE-f32 RIFF/WAVE, interleaved little-endian."""
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("expected [samples, 2] stereo array")
    data = np.asarray(samples, dtype="<f4", order="C").tobytes(order="C")
    fmt = struct.pack("<HHIIHH", 3, 2, RATE, RATE * 8, 8, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def _lowpass_once(audio: np.ndarray, cutoff: float, transition: float,
                  power: float) -> np.ndarray:
    """Mirror dsp::gain/dsp::lowpass over the entire physical source."""
    n = audio.shape[0]
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    t = np.clip((f - cutoff) / transition, 0.0, 1.0)
    gain = np.maximum(0.5 + 0.5 * np.cos(np.pi * t), 0.0) ** power
    result = np.empty_like(audio, dtype=np.float64)
    for ch in range(2):
        spec = np.fft.rfft(audio[:, ch])
        spec *= gain
        result[:, ch] = np.fft.irfft(spec, n=n)
    return result


def _colored_noise(rng: np.random.Generator, n: int, low_hz: float,
                   high_hz: float) -> np.ndarray:
    """Globally filtered noise; no chunk-dependent state or edge restart."""
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    lo = np.clip(f / low_hz, 0.0, 1.0)
    hi = np.clip((RATE * 0.5 - f) / (RATE * 0.5 - high_hz), 0.0, 1.0)
    window = (0.5 - 0.5 * np.cos(np.pi * lo)) * (0.5 - 0.5 * np.cos(np.pi * hi))
    out = np.empty((n, 2), dtype=np.float64)
    for ch in range(2):
        x = rng.standard_normal(n)
        out[:, ch] = np.fft.irfft(np.fft.rfft(x) * window, n=n)
    return out


def _render_source(n: int, seed: int, events: Iterable[float]) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / RATE
    duration = n / RATE

    # Shared harmonic source, smoothly drifting in frequency and level. Phase
    # is the integral of a common slow frequency trajectory over the full take.
    drift = 0.0018 * np.sin(2 * np.pi * t / 8.7) + 0.0007 * np.sin(2 * np.pi * t / 3.9 + 0.4)
    level = 0.72 + 0.18 * np.sin(2 * np.pi * t / 7.2 + 0.2) + 0.07 * np.sin(2 * np.pi * t / 3.1)
    section = np.interp(t, [0, 8, 12, 18, 22, 48], [0.4, 0.4, 1.0, 1.0, 0.2, 0.2])
    level *= section
    base_hz = 146.8
    harmonic = np.zeros((n, 2), dtype=np.float64)
    for k in range(1, 19):
        inst_hz = base_hz * k * (1.0 + drift)
        phase = 2 * np.pi * np.cumsum(inst_hz, dtype=np.float64) / RATE
        amp = (1.0 / (k ** 1.13)) * (1.0 + 0.08 * np.sin(2 * np.pi * t / (5.4 + k * 0.07) + k))
        # Small shared stereo spread; channel phase progression remains linked.
        harmonic[:, 0] += amp * np.sin(phase)
        harmonic[:, 1] += amp * np.sin(phase + 0.035 * np.sin(2 * np.pi * t / 11.0))
    harmonic *= (0.20 * level)[:, None]

    # Continuous dense aspiration, with independent but bounded stereo fields.
    aspiration = _colored_noise(rng, n, 900.0, 18_500.0)
    aspiration *= (0.022 * (0.85 + 0.15 * np.sin(2 * np.pi * t / 6.3)))[:, None]

    # Four fixed attacks, followed by globally indexed early and late decay
    # fields. Their noise is generated once on the full timeline.
    impact = np.zeros((n, 2), dtype=np.float64)
    onset_rows = []
    decay_noise = rng.standard_normal((n, 2))
    for event_index, onset in enumerate(events):
        sample = int(round(float(onset) * RATE))
        if sample < 0 or sample >= n:
            continue
        q = np.arange(n - sample, dtype=np.float64) / RATE
        attack_n = min(int(0.045 * RATE), len(q))
        attack = np.zeros_like(q)
        attack[:attack_n] = np.exp(-q[:attack_n] / 0.010) * np.sin(2 * np.pi * (1200 + 180 * event_index) * q[:attack_n])
        impact[sample:, 0] += 0.12 * attack
        impact[sample:, 1] += 0.12 * attack
        # Actual delayed copies of the attack give deterministic stereo early
        # reflections. The later noise field is a diffuse-decay proxy.
        for left_delay, right_delay, gain in ((0.031, 0.041, 0.036),
                                               (0.067, 0.059, 0.022),
                                               (0.113, 0.129, 0.015)):
            for ch, delay_s in enumerate((left_delay, right_delay)):
                begin = sample + int(round(delay_s * RATE))
                count = min(attack_n, n - begin)
                if count > 0:
                    impact[begin:begin + count, ch] += gain * attack[:count]
        tail = np.arange(n - sample, dtype=np.float64) / RATE
        env = np.exp(-tail / 2.8)
        impact[sample:, 0] += 0.009 * env * decay_noise[sample:, 0]
        impact[sample:, 1] += 0.009 * env * decay_noise[sample:, 1]
        onset_rows.append({"time_seconds": float(onset), "sample": sample,
                           "kind": "fixed_attack_with_early_and_late_decay"})

    source = harmonic + aspiration + impact
    peak = float(np.max(np.abs(source)))
    common_gain = 0.62 / peak if peak > 0 else 1.0
    source *= common_gain
    return source.astype(np.float32), onset_rows


def prepare(out: Path, seed: int = DEFAULT_SEED, duration_seconds: float = 32.0,
            view_specs: list[tuple[str, float, float]] | None = None,
            event_times: Iterable[float] | None = None) -> dict:
    """Generate a full source, globally degrade it once, then write aligned views."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {out}")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    n = int(round(duration_seconds * RATE))
    if not 1 <= n <= 48 * RATE:
        raise ValueError("fixture duration must round to 1 sample through 48 seconds")
    if view_specs is None:
        view_len = min(int(round(_VIEW_SECONDS * RATE)), n)
        second_origin = min(_VIEW_SHIFT_SAMPLES, n - view_len)
        view_specs = [("view0", 0.0, view_len / RATE)]
        if second_origin > 0:
            view_specs.append(("view3", second_origin / RATE, view_len / RATE))
    normalized_views = []
    seen_views = set()
    for view_id, origin_seconds, length_seconds in view_specs:
        if not isinstance(view_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", view_id) or view_id in seen_views:
            raise ValueError("view identifiers must be unique local names")
        seen_views.add(view_id)
        origin = int(round(origin_seconds * RATE))
        length = int(round(length_seconds * RATE))
        if origin < 0 or length <= 0 or origin + length > n:
            raise ValueError(f"view {view_id} exceeds the physical source")
        normalized_views.append((view_id, origin, length))

    events = tuple(DEFAULT_EVENTS if event_times is None else event_times)
    target, event_rows = _render_source(n, seed, events)
    # Filter after the one common peak scale, over the entire source, before
    # any crop. Rust's f32 rustfft can differ by small roundoff from this.
    damaged = _lowpass_once(target.astype(np.float64), 1500.0, 225.0, 2.0).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    full_target = out / "target-full.wav"
    full_input = out / "input-full.wav"
    _write_float_wav(full_target, target)
    _write_float_wav(full_input, damaged)
    paths["target_full"] = full_target
    paths["input_full"] = full_input

    views = []
    for view_id, origin, length in normalized_views:
        target_name = f"{view_id}-target.wav"
        input_name = f"{view_id}-input.wav"
        target_path, input_path = out / target_name, out / input_name
        _write_float_wav(target_path, target[origin:origin + length])
        _write_float_wav(input_path, damaged[origin:origin + length])
        paths[f"{view_id}_target"] = target_path
        paths[f"{view_id}_input"] = input_path
        local_events = [{**e, "view_sample": e["sample"] - origin}
                        for e in event_rows if origin <= e["sample"] < origin + length]
        views.append({"id": view_id, "input": input_name, "target": target_name,
                      "origin_sample": origin, "samples": length,
                      "events": local_events})

    comparison_start, comparison_end = int(8 * RATE), int(21 * RATE)
    if n < comparison_end:
        comparison_start, comparison_end = 0, n
    manifest = {
        "schema": "long-coherence-procedural-fixture-v1",
        "sample_rate": RATE, "channels": 2, "samples": n,
        "duration_seconds": n / RATE, "seed": int(seed),
        "source_sha256": {name: _hash_file(path) for name, path in paths.items()},
        "generator_sha256": _hash_file(Path(__file__)),
        "environment": {"numpy": np.__version__, "python": platform.python_version()},
        "recipe": {
            "source": "continuous shared harmonic oscillator bank with integrated slow frequency drift and multisecond level motion; dense globally filtered aspiration; fixed attacks with global-index early and late decay fields",
            "event_times_seconds": list(map(float, events)),
            "common_peak_target": 0.62,
            "harmonic_section_envelope": {"seconds": [0, 8, 12, 18, 22, 48], "relative_levels": [0.4, 0.4, 1.0, 1.0, 0.2, 0.2]},
            "early_reflection_delays_seconds_lr": [[0.031, 0.041], [0.067, 0.059], [0.113, 0.129]],
            "late_decay_proxy_seconds": 2.8,
            "damage": {"type": "whole-source Fourier lowpass", "cutoff_hz": 1500.0,
                       "transition_hz": 225.0, "power": 2.0, "applications": 1,
                       "implementation": "NumPy float64 rfft/irfft; mirrors src/dsp.rs gain and global finite-clip filtering, with expected small differences from Rust f32 rustfft"},
            "continuity_controls": ["oscillator phase integrated over full source", "shared slow frequency and level trajectories", "aspiration generated and filtered over full source", "decay noise indexed on full source", "damage applied before crop"]
        },
        "events": event_rows,
        "views": views,
        "native_stft_hop_samples": 256,
        "view_origins_hop_aligned": all(origin % 256 == 0 for _, origin, _ in normalized_views),
        "comparison_interior_global_samples": [comparison_start, comparison_end],
        "audio_policy": "procedural only; no owner audio",
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Verify all files through the repository's float WAV reader and check the
    # expected shape and exact samples after its f32 decoding.
    if __package__:
        from .roach_ears.audio_io import read_wav
    else:
        from roach_ears.audio_io import read_wav
    for key, path in paths.items():
        decoded = read_wav(path).samples.T
        expected = (target if "target" in key else damaged)
        if "_full" not in key:
            view_id = key.rsplit("_", 1)[0]
            _, origin, length = next(v for v in normalized_views if v[0] == view_id)
            expected = expected[origin:origin + length]
        if decoded.shape != expected.shape or not np.array_equal(decoded, expected):
            raise RuntimeError(f"WAV roundtrip mismatch: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--duration-seconds", type=float, default=32.0)
    args = parser.parse_args()
    manifest = prepare(args.out, args.seed, args.duration_seconds)
    print(json.dumps({"out": str(args.out), "samples": manifest["samples"],
                      "hashes": manifest["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
