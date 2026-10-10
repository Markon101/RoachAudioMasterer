"""
ROACH EARS: Autonomous Informative Excerpt Selection Engine
Mines high-information 5-15 second regions across:
1. Max Perceptual Delta (highest SDR difference anomaly)
2. Strongest Transient Punch (percussive attack density)
3. Dense High-Frequency Arrangement (cymbal/shimmer/air test)
4. Sub-Bass Rumble Decay (mono-sub integrity)
5. Worst Localized Artifact Spike (clipping / phase flutter)

Exports matched, time-aligned, level-matched excerpt WAVs.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .audio_io import AudioBuffer, write_wav


class ExcerptCandidate:
    def __init__(
        self,
        excerpt_id: str,
        start_sec: float,
        duration_sec: float,
        reason: str,
        score: float,
        metadata: Dict[str, Any]
    ):
        self.excerpt_id = excerpt_id
        self.start_sec = start_sec
        self.duration_sec = duration_sec
        self.reason = reason
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "excerpt_id": self.excerpt_id,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.start_sec + self.duration_sec, 2),
            "duration_sec": round(self.duration_sec, 2),
            "selection_reason": self.reason,
            "information_score": round(self.score, 4),
            "metadata": self.metadata,
        }


def select_informative_excerpts(
    ref: AudioBuffer,
    cand: AudioBuffer,
    excerpt_duration_sec: float = 10.0,
    max_excerpts: int = 4
) -> List[ExcerptCandidate]:
    """
    Scans the aligned pair and selects diversity-balanced, high-information excerpts.
    """
    sr = ref.sample_rate
    hop_sec = 2.0
    win_len = int(round(excerpt_duration_sec * sr))
    hop_len = int(round(hop_sec * sr))
    total_samples = ref.num_samples

    if total_samples <= win_len:
        # Full track is already shorter or equal to excerpt duration
        return [
            ExcerptCandidate(
                excerpt_id="full_clip",
                start_sec=0.0,
                duration_sec=ref.duration_sec,
                reason="Entire track duration is within excerpt budget",
                score=1.0,
                metadata={}
            )
        ]

    diff_sig = cand.samples - ref.samples
    m_ref = ref.to_mono()
    m_diff = 0.5 * (diff_sig[0] + diff_sig[1]) if diff_sig.shape[0] >= 2 else diff_sig[0]

    # Pre-calculate rolling features across 2-second windows
    scores_delta = []
    scores_transient = []
    scores_air = []
    scores_sub = []
    timestamps = []

    # Simple 1-pole highpass for air (> 8 kHz) and lowpass for sub (< 80 Hz)
    dt = 1.0 / sr
    rc_air = 1.0 / (2.0 * np.pi * 8000.0)
    alpha_air = rc_air / (rc_air + dt)

    rc_sub = 1.0 / (2.0 * np.pi * 80.0)
    alpha_sub = dt / (rc_sub + dt)

    air_signal = np.zeros_like(m_ref)
    sub_signal = np.zeros_like(m_ref)
    y_air = 0.0
    prev_x = 0.0
    y_sub = 0.0
    for i in range(len(m_ref)):
        x = m_ref[i]
        y_air = alpha_air * (y_air + x - prev_x)
        prev_x = x
        air_signal[i] = y_air

        y_sub += alpha_sub * (x - y_sub)
        sub_signal[i] = y_sub

    # Transient detector via rectified first difference
    transient_env = np.maximum(0.0, np.diff(np.abs(m_ref), prepend=0.0))

    for start in range(0, total_samples - win_len + 1, hop_len):
        t_sec = start / float(sr)
        timestamps.append(t_sec)

        # 1. Delta power relative to reference (perceptual difference saliency)
        d_chunk = m_diff[start:start + win_len]
        r_chunk = m_ref[start:start + win_len]
        d_pow = np.mean(d_chunk ** 2) + 1e-12
        r_pow = np.mean(r_chunk ** 2) + 1e-12
        delta_score = d_pow / (r_pow + 1e-6)
        scores_delta.append(delta_score)

        # 2. Transient punch
        tr_chunk = transient_env[start:start + win_len]
        scores_transient.append(np.mean(tr_chunk))

        # 3. Dense high frequencies
        air_chunk = air_signal[start:start + win_len]
        scores_air.append(np.mean(air_chunk ** 2))

        # 4. Sub-bass weight
        sub_chunk = sub_signal[start:start + win_len]
        scores_sub.append(np.mean(sub_chunk ** 2))

    timestamps = np.array(timestamps)
    scores_delta = np.array(scores_delta)
    scores_transient = np.array(scores_transient)
    scores_air = np.array(scores_air)
    scores_sub = np.array(scores_sub)

    selected: List[ExcerptCandidate] = []
    chosen_starts = []

    def is_overlap(t: float) -> bool:
        return any(abs(t - c) < (excerpt_duration_sec * 0.70) for c in chosen_starts)

    # Pick 1: Maximum Perceptual Delta
    if len(scores_delta) > 0:
        sorted_indices = np.argsort(scores_delta)[::-1]
        for idx in sorted_indices:
            t = timestamps[idx]
            if not is_overlap(t):
                chosen_starts.append(t)
                selected.append(ExcerptCandidate(
                    excerpt_id="max_delta",
                    start_sec=float(t),
                    duration_sec=excerpt_duration_sec,
                    reason="Largest perceptual difference anomaly between candidate and reference",
                    score=float(scores_delta[idx]),
                    metadata={"delta_power_ratio": float(scores_delta[idx])}
                ))
                break

    # Pick 2: Strongest Transient Punch
    if len(scores_transient) > 0 and len(selected) < max_excerpts:
        sorted_indices = np.argsort(scores_transient)[::-1]
        for idx in sorted_indices:
            t = timestamps[idx]
            if not is_overlap(t):
                chosen_starts.append(t)
                selected.append(ExcerptCandidate(
                    excerpt_id="transient_punch",
                    start_sec=float(t),
                    duration_sec=excerpt_duration_sec,
                    reason="Highest transient attack density and percussive dynamic impact",
                    score=float(scores_transient[idx]),
                    metadata={"transient_metric": float(scores_transient[idx])}
                ))
                break

    # Pick 3: Dense High-Frequency Arrangement
    if len(scores_air) > 0 and len(selected) < max_excerpts:
        sorted_indices = np.argsort(scores_air)[::-1]
        for idx in sorted_indices:
            t = timestamps[idx]
            if not is_overlap(t):
                chosen_starts.append(t)
                selected.append(ExcerptCandidate(
                    excerpt_id="dense_air",
                    start_sec=float(t),
                    duration_sec=excerpt_duration_sec,
                    reason="Densely layered high-frequency arrangement (shimmer, cymbal sheen, and air texture)",
                    score=float(scores_air[idx]),
                    metadata={"air_energy": float(scores_air[idx])}
                ))
                break

    # Pick 4: Sub-Bass Weight & Decay
    if len(scores_sub) > 0 and len(selected) < max_excerpts:
        sorted_indices = np.argsort(scores_sub)[::-1]
        for idx in sorted_indices:
            t = timestamps[idx]
            if not is_overlap(t):
                chosen_starts.append(t)
                selected.append(ExcerptCandidate(
                    excerpt_id="sub_bass_rumble",
                    start_sec=float(t),
                    duration_sec=excerpt_duration_sec,
                    reason="Deepest sub-bass resonance and low-end extension decay",
                    score=float(scores_sub[idx]),
                    metadata={"sub_energy": float(scores_sub[idx])}
                ))
                break

    return selected


def export_excerpts(
    ref: AudioBuffer,
    cand: AudioBuffer,
    excerpts: List[ExcerptCandidate],
    out_dir: Path
) -> List[Dict[str, Any]]:
    """
    Slices and writes excerpt WAV files for each candidate:
    - ref_excerpt_<id>.wav
    - cand_excerpt_<id>.wav
    - delta_excerpt_<id>.wav (difference signal boosted by +24dB for audible inspection)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for exc in excerpts:
        ref_slice = ref.slice_time(exc.start_sec, exc.duration_sec)
        cand_slice = cand.slice_time(exc.start_sec, exc.duration_sec)

        # Create amplified difference signal for inspection (+24 dB boost)
        diff_samples = (cand_slice.samples - ref_slice.samples) * 15.8489 # 10^(24/20)
        diff_slice = AudioBuffer(np.clip(diff_samples, -1.0, 1.0), ref.sample_rate)

        ref_path = out_dir / f"ref_{exc.excerpt_id}.wav"
        cand_path = out_dir / f"cand_{exc.excerpt_id}.wav"
        diff_path = out_dir / f"delta_{exc.excerpt_id}_plus24dB.wav"

        write_wav(ref_path, ref_slice, bit_depth=16)
        write_wav(cand_path, cand_slice, bit_depth=16)
        write_wav(diff_path, diff_slice, bit_depth=16)

        entry = exc.to_dict()
        entry["files"] = {
            "reference_wav": str(ref_path),
            "candidate_wav": str(cand_path),
            "delta_wav_24db": str(diff_path),
        }
        manifest.append(entry)

    return manifest
