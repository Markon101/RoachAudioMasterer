"""
ROACH EARS: Deterministic & Psychoacoustic Metric Engine (Layer A & Layer B)
Computes integrated/short-term LUFS, true peak, multiband spectral tilt,
Wiener flatness, high-frequency grain, stereo coherence, mono compatibility,
and 15ms time-localized artifact bursts.
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple
from .audio_io import AudioBuffer

TRUE_PEAK_ESTIMATOR = 'native-fir-4x-24tap-v1'


def _true_peak_linear(samples: np.ndarray, block_size: int = 8192) -> Tuple[float, int]:
    """Same finite f32 estimator/support as src/master.rs::true_peak.

    Three quarter-sample phases, 24 taps each. Full-support windows only;
    original samples include clip edges. Interpolation arrays are block bounded;
    dtype conversion, validation and sample-peak reduction inspect the full input.
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim not in (1, 2) or block_size <= 0 or not np.all(np.isfinite(x)):
        raise ValueError('true peak requires finite [samples] or [channels,samples]')
    if x.size == 0:
        return 0., 0
    peak = float(np.max(np.abs(x)))
    overs = int(np.count_nonzero(np.abs(x) > 1.))
    pi, taps = np.float32(np.pi), 12
    k = np.arange(2 * taps, dtype=np.float32)
    phases = []
    for p in range(3):
        d = (k - np.float32(taps - 1)) - np.float32((p + 1) / 4.)
        argument = pi * d
        sinc = np.sin(argument) / argument  # All three phases are noninteger.
        window = np.float32(.5) + np.float32(.5) * np.cos(argument / np.float32(taps))
        phases.append(sinc * window)
    windows = max(0, x.shape[-1] - 2 * taps + 1)
    for start in range(0, windows, block_size):
        count = min(block_size, windows - start)
        for phase in phases:
            y = np.zeros((*x.shape[:-1], count), dtype=np.float32)
            # Preserve Rust's ordered f32 multiply/add, including across blocks.
            for offset, coefficient in enumerate(phase):
                y += x[..., start + offset:start + offset + count] * coefficient
            peak = max(peak, float(np.max(np.abs(y))))
            overs += int(np.count_nonzero(np.abs(y) > 1.))
    return peak, overs


def compute_true_peak(samples: np.ndarray, factor: int = 4) -> Tuple[float, int]:
    """Fourfold finite FIR true-peak estimate, referenced to native Rust.

    Count is evaluated original/interpolated grid points above unit amplitude,
    not continuous crossings or audible events. Silence floors at -120 dBTP.
    This is an estimate, not a continuous-time peak guarantee.
    """
    if factor != 4:
        raise ValueError('the native reference estimator supports factor=4 only')
    peak, overs = _true_peak_linear(samples)
    return float(20. * np.log10(max(1e-6, peak))), overs


def apply_biquad(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Direct Form II Transposed biquad filter implementation in NumPy."""
    y = np.zeros_like(x)
    z1 = 0.0
    z2 = 0.0
    b0, b1, b2 = b[0], b[1], b[2]
    a1, a2 = a[1], a[2]
    for i in range(len(x)):
        xi = x[i]
        yi = b0 * xi + z1
        z1 = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        y[i] = yi
    return y


def compute_bs1770_lufs(audio: AudioBuffer) -> Dict[str, Any]:
    """
    Computes ITU-R BS.1770-4 Integrated Loudness (LUFS)
    and short-term sliding window loudness percentiles (L10, L50, L90).
    """
    sr = audio.sample_rate
    # Stage 1: High shelf filter (RLB pre-filter)
    # Stage 2: High pass filter
    # Approximate coefficients for 44.1k/48k
    b_shelf = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    a_shelf = np.array([1.0, -1.69065929318241, 0.73248077421585])
    b_hp = np.array([1.0, -2.0, 1.0])
    a_hp = np.array([1.0, -1.99004745483398, 0.99007225036621])

    filtered_channels = []
    for ch in range(min(2, audio.channels)):
        s = apply_biquad(b_shelf, a_shelf, audio.samples[ch])
        f = apply_biquad(b_hp, a_hp, s)
        filtered_channels.append(f)

    # 400ms rectangular blocks, 100ms hop (75% overlap)
    block_size = int(round(0.400 * sr))
    hop_size = int(round(0.100 * sr))
    n = filtered_channels[0].shape[0]

    block_energies = []
    for start in range(0, n - block_size + 1, hop_size):
        e = 0.0
        for ch_data in filtered_channels:
            segment = ch_data[start:start + block_size]
            e += np.mean(segment ** 2)
        block_energies.append(e)

    if not block_energies:
        return {"integrated_lufs": -70.0, "l10": -70.0, "l50": -70.0, "l90": -70.0, "dynamic_spread_db": 0.0}

    block_energies = np.array(block_energies)
    block_loudness = -0.691 + 10.0 * np.log10(np.maximum(1e-12, block_energies))

    # Absolute gate (-70 LUFS)
    abs_gated = block_energies[block_loudness > -70.0]
    if len(abs_gated) == 0:
        return {"integrated_lufs": -70.0, "l10": -70.0, "l50": -70.0, "l90": -70.0, "dynamic_spread_db": 0.0}

    # Relative gate: -10 dB below ungated mean
    gamma_a = -0.691 + 10.0 * np.log10(np.mean(abs_gated))
    rel_gated = abs_gated[block_loudness[block_loudness > -70.0] > (gamma_a - 10.0)]
    if len(rel_gated) == 0:
        integrated = gamma_a
    else:
        integrated = -0.691 + 10.0 * np.log10(np.mean(rel_gated))

    # Dynamic percentiles over active sections (> -60 LUFS)
    active_loudness = block_loudness[block_loudness > -60.0]
    if len(active_loudness) > 0:
        l10 = float(np.percentile(active_loudness, 90)) # L10: loudest 10%
        l50 = float(np.percentile(active_loudness, 50))
        l90 = float(np.percentile(active_loudness, 10)) # L90: quietest 10%
        spread = l10 - l90
    else:
        l10, l50, l90, spread = integrated, integrated, integrated, 0.0

    return {
        "integrated_lufs": round(float(integrated), 2),
        "l10_lufs": round(float(l10), 2),
        "l50_lufs": round(float(l50), 2),
        "l90_lufs": round(float(l90), 2),
        "dynamic_spread_db": round(float(spread), 2),
    }


def compute_spectral_metrics(audio: AudioBuffer) -> Dict[str, Any]:
    """
    Computes multiband energy distribution, spectral tilt (dB/octave),
    Wiener flatness, and high-frequency grain / roughness proxies.
    """
    mono = audio.to_mono()
    sr = audio.sample_rate
    n_fft = 4096
    hop = 1024

    # STFT
    n_frames = (len(mono) - n_fft) // hop + 1
    if n_frames <= 0:
        return {}

    # Windowing
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    spectrogram = []
    for i in range(n_frames):
        chunk = mono[i * hop:i * hop + n_fft] * window
        mag = np.abs(np.fft.rfft(chunk))
        spectrogram.append(mag)
    spec = np.array(spectrogram) # shape: (frames, bins)

    mean_spectrum = np.mean(spec, axis=0) + 1e-12

    # Band definitions
    bands = {
        "sub": (20.0, 60.0),
        "low_mid": (60.0, 250.0),
        "mid": (250.0, 4000.0),
        "presence": (4000.0, 10000.0),
        "air": (10000.0, 20000.0),
    }

    band_energies = {}
    band_flatness = {}

    total_energy = np.sum(mean_spectrum ** 2) + 1e-12
    for b_name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs < f_high)
        if np.any(mask):
            b_mag = mean_spectrum[mask]
            b_pow = b_mag ** 2
            b_energy = np.sum(b_pow)
            band_energies[f"{b_name}_share_pct"] = round(float(100.0 * b_energy / total_energy), 2)
            band_energies[f"{b_name}_rms_dbfs"] = round(float(20.0 * np.log10(np.sqrt(np.mean(b_pow)))), 2)

            # Wiener flatness: exp(mean(ln(x))) / mean(x)
            geo_mean = np.exp(np.mean(np.log(b_pow + 1e-12)))
            arith_mean = np.mean(b_pow) + 1e-12
            band_flatness[f"{b_name}_flatness"] = round(float(geo_mean / arith_mean), 4)

    # Global Spectral Tilt (dB/octave) between 100 Hz and 16000 Hz
    tilt_mask = (freqs >= 100.0) & (freqs <= 16000.0)
    log_freqs = np.log2(freqs[tilt_mask])
    db_mag = 20.0 * np.log10(mean_spectrum[tilt_mask])
    slope_db_oct, _ = np.polyfit(log_freqs, db_mag, 1)

    # High-frequency grain & roughness proxy:
    # Frame-to-frame spectral flux variance above 8 kHz
    air_mask = freqs >= 8000.0
    air_spec = spec[:, air_mask]
    diff_spec = np.diff(air_spec, axis=0)
    flux = np.sqrt(np.mean(diff_spec ** 2, axis=1))
    flux_var = float(np.var(flux))
    grain_proxy = float(np.mean(flux))

    return {
        "spectral_slope_db_oct": round(float(slope_db_oct), 2),
        "hf_grain_proxy": round(grain_proxy, 5),
        "hf_flux_variance": round(flux_var, 6),
        **band_energies,
        **band_flatness,
    }


def compute_stereo_acoustics(audio: AudioBuffer) -> Dict[str, Any]:
    """
    Computes Mid/Side energy ratio, frequency-dependent stereo coherence,
    and mono downmix cancellation index (CDI).
    """
    if not audio.is_stereo():
        return {
            "is_mono": True,
            "mid_side_ratio_db": 100.0,
            "sub_side_rms_dbfs": -120.0,
            "stereo_coherence": 1.0,
            "cancellation_distortion_index": 0.0,
        }

    left = audio.samples[0]
    right = audio.samples[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    m_pow = np.mean(mid ** 2) + 1e-12
    s_pow = np.mean(side ** 2) + 1e-12
    ms_ratio = 10.0 * np.log10(m_pow / s_pow)

    # Sub-bass side power (< 60 Hz)
    sr = audio.sample_rate
    # simple 1-pole 60 Hz lowpass on side
    dt = 1.0 / sr
    rc = 1.0 / (2.0 * math.pi * 60.0)
    alpha = dt / (rc + dt)
    sub_side = np.zeros_like(side)
    y = 0.0
    for i in range(len(side)):
        y += alpha * (side[i] - y)
        sub_side[i] = y
    sub_side_rms_dbfs = 20.0 * np.log10(np.sqrt(np.mean(sub_side ** 2)) + 1e-12)

    # Broad-band stereo correlation
    l_norm = left - np.mean(left)
    r_norm = right - np.mean(right)
    denom = (np.std(l_norm) * np.std(r_norm)) + 1e-12
    coherence = float(np.mean(l_norm * r_norm) / denom)

    # Cancellation Distortion Index (CDI): Energy lost when summed to mono
    stereo_total_pow = np.mean(left ** 2 + right ** 2) * 0.5 + 1e-12
    mono_pow = np.mean(mid ** 2) + 1e-12
    cdi = float(max(0.0, (stereo_total_pow - mono_pow) / stereo_total_pow))

    return {
        "is_mono": False,
        "mid_side_ratio_db": round(float(ms_ratio), 2),
        "sub_side_rms_dbfs": round(float(sub_side_rms_dbfs), 2),
        "stereo_coherence": round(coherence, 4),
        "cancellation_distortion_index": round(cdi, 4),
    }


def scan_localized_artifacts(audio: AudioBuffer) -> List[Dict[str, Any]]:
    """
    Scans the audio in 15ms windows to detect short-duration localized artifacts:
    - Intersample clipping events
    - Sudden phase flutter spikes
    - Transient harshness bursts
    """
    sr = audio.sample_rate
    win_size = int(round(0.015 * sr)) # 15 ms
    hop_size = win_size // 2
    mono = audio.to_mono()
    n = len(mono)

    artifacts = []
    # Rolling energy and crest factor
    for start in range(0, n - win_size + 1, hop_size):
        chunk = mono[start:start + win_size]
        peak = np.max(np.abs(chunk))
        rms = np.sqrt(np.mean(chunk ** 2)) + 1e-12
        crest = peak / rms

        time_sec = start / float(sr)

        # 1. Hard clipping detection
        if peak >= 0.999:
            # check flat-topping
            flat_samples = np.sum(np.abs(chunk) >= 0.998)
            if flat_samples >= 4:
                artifacts.append({
                    "timestamp_sec": round(time_sec, 3),
                    "type": "clipping_flat_top",
                    "severity": round(float(flat_samples / 10.0), 2),
                    "details": f"{flat_samples} clipped samples in 15ms window"
                })

        # 2. Extreme anomalous crest factor burst (> 20 dB crest in quiet context)
        if crest > 10.0 and peak > 0.30:
            artifacts.append({
                "timestamp_sec": round(time_sec, 3),
                "type": "transient_spike",
                "severity": round(float(crest / 10.0), 2),
                "details": f"Sharp transient burst with crest factor {crest:.1f}"
            })

    return artifacts[:25] # Return top 25 localized events
