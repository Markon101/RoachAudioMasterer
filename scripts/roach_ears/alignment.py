"""
ROACH EARS: High-Precision Audio Alignment & Null Differential Engine
Computes sub-sample cross-correlation alignment, drift detection,
and uncovers true null differences between audio versions.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .audio_io import AudioBuffer


def align_audio_pair(
    ref: AudioBuffer,
    cand: AudioBuffer,
    max_lag_sec: float = 0.50
) -> Tuple[AudioBuffer, AudioBuffer, Dict[str, Any]]:
    """
    Aligns candidate audio to reference audio using cross-correlation.
    Returns: (aligned_ref, aligned_cand, alignment_metadata)
    """
    assert ref.sample_rate == cand.sample_rate, (
        f"Sample rate mismatch: {ref.sample_rate} vs {cand.sample_rate}"
    )
    sr = ref.sample_rate
    max_lag = int(round(max_lag_sec * sr))

    # Downmix to mono for robust timing correlation
    m_ref = ref.to_mono()
    m_cand = cand.to_mono()

    # Use first 30 seconds for alignment search to keep computation bounded
    search_len = min(len(m_ref), len(m_cand), int(30.0 * sr))
    sig_ref = m_ref[:search_len]
    sig_cand = m_cand[:search_len]

    # Normalize
    ref_norm = sig_ref - np.mean(sig_ref)
    cand_norm = sig_cand - np.mean(sig_cand)
    ref_std = np.std(ref_norm) + 1e-12
    cand_std = np.std(cand_norm) + 1e-12

    # FFT-based cross-correlation over search window
    n = len(sig_ref) + len(sig_cand) - 1
    # next power of 2
    n_fft = 1 << (n - 1).bit_length()
    f_ref = np.fft.rfft(ref_norm, n_fft)
    f_cand = np.fft.rfft(cand_norm, n_fft)
    xcorr = np.fft.irfft(f_cand * np.conj(f_ref), n_fft)

    # Convert circular indices to linear lags [-len(sig_ref)+1, len(sig_cand)-1]
    # Peak index in xcorr
    lags = np.arange(len(xcorr))
    lags[lags > n_fft // 2] -= n_fft

    # Restrict to [-max_lag, max_lag]
    valid_mask = np.abs(lags) <= max_lag
    valid_lags = lags[valid_mask]
    valid_xcorr = xcorr[valid_mask] / (search_len * ref_std * cand_std)

    best_idx = np.argmax(valid_xcorr)
    int_lag = int(valid_lags[best_idx])
    peak_corr = float(valid_xcorr[best_idx])

    # Sub-sample parabolic interpolation around peak
    sub_lag = float(int_lag)
    if 0 < best_idx < len(valid_xcorr) - 1:
        y0 = valid_xcorr[best_idx - 1]
        y1 = valid_xcorr[best_idx]
        y2 = valid_xcorr[best_idx + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        if abs(denom) > 1e-12:
            delta = (y0 - y2) / denom
            if abs(delta) < 1.0:
                sub_lag += float(delta)

    # Shift candidate to match reference
    if int_lag > 0:
        # cand is delayed relative to ref -> cand starts late, slice cand from int_lag
        cand_samples = cand.samples[:, int_lag:]
        ref_samples = ref.samples
    elif int_lag < 0:
        # cand is early relative to ref -> ref starts from -int_lag
        cand_samples = cand.samples
        ref_samples = ref.samples[:, -int_lag:]
    else:
        cand_samples = cand.samples
        ref_samples = ref.samples

    # Trim to identical common frame length
    common_len = min(ref_samples.shape[1], cand_samples.shape[1])
    aligned_ref_samples = ref_samples[:, :common_len]
    aligned_cand_samples = cand_samples[:, :common_len]

    # Compute SDR (Signal to Difference Ratio in dB)
    diff = aligned_cand_samples - aligned_ref_samples
    ref_pow = np.mean(aligned_ref_samples ** 2) + 1e-12
    diff_pow = np.mean(diff ** 2) + 1e-12
    sdr_db = 10.0 * np.log10(ref_pow / diff_pow)
    delta_rms_dbfs = 20.0 * np.log10(np.sqrt(diff_pow) + 1e-12)

    meta = {
        "integer_lag_samples": int_lag,
        "subsample_lag_samples": round(sub_lag, 3),
        "lag_time_ms": round((sub_lag / sr) * 1000.0, 3),
        "peak_cross_correlation": round(peak_corr, 5),
        "common_duration_sec": round(common_len / sr, 3),
        "sdr_db": round(float(sdr_db), 2),
        "delta_rms_dbfs": round(float(delta_rms_dbfs), 2),
        "is_bit_identical": bool(np.all(diff == 0.0)),
        "is_near_null": bool(sdr_db > 60.0),
    }

    aligned_ref = AudioBuffer(aligned_ref_samples, sr)
    aligned_cand = AudioBuffer(aligned_cand_samples, sr)
    return aligned_ref, aligned_cand, meta
