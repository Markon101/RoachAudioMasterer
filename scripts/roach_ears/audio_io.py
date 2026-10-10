"""
ROACH EARS: Audio I/O Engine
Pure NumPy + wave module for zero-dependency, bit-exact reading, writing,
and format verification of WAV files on Android/Termux.
"""

import math
import struct
import wave
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class AudioBuffer:
    def __init__(self, samples: np.ndarray, sample_rate: int):
        """
        samples: shape (channels, num_samples) as float32 in [-1.0, 1.0]
        """
        assert samples.ndim == 2, f"Expected (channels, samples), got {samples.shape}"
        self.samples = samples.astype(np.float32)
        self.sample_rate = sample_rate

    @property
    def channels(self) -> int:
        return self.samples.shape[0]

    @property
    def num_samples(self) -> int:
        return self.samples.shape[1]

    @property
    def duration_sec(self) -> float:
        return self.num_samples / float(self.sample_rate)

    def is_stereo(self) -> bool:
        return self.channels >= 2

    def to_mono(self) -> np.ndarray:
        if self.channels == 1:
            return self.samples[0].copy()
        return 0.5 * (self.samples[0] + self.samples[1])

    def mid_side(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_stereo():
            mono = self.samples[0]
            return mono, np.zeros_like(mono)
        mid = 0.5 * (self.samples[0] + self.samples[1])
        side = 0.5 * (self.samples[0] - self.samples[1])
        return mid, side

    def slice_time(self, start_sec: float, duration_sec: float) -> 'AudioBuffer':
        start_idx = int(round(start_sec * self.sample_rate))
        end_idx = int(round((start_sec + duration_sec) * self.sample_rate))
        start_idx = max(0, min(self.num_samples, start_idx))
        end_idx = max(start_idx, min(self.num_samples, end_idx))
        return AudioBuffer(self.samples[:, start_idx:end_idx].copy(), self.sample_rate)


def read_wav(path: Path) -> AudioBuffer:
    """
    Reads a standard or extensible WAV file (PCM 16/24/32-bit or IEEE 32-bit Float)
    directly by parsing RIFF chunks, avoiding standard wave module limitations.
    """
    path = Path(path)
    with open(path, "rb") as f:
        riff_tag = f.read(4)
        if riff_tag != b"RIFF":
            raise ValueError(f"Not a valid RIFF file: {path}")
        _file_size = struct.unpack("<I", f.read(4))[0]
        wave_tag = f.read(4)
        if wave_tag != b"WAVE":
            raise ValueError(f"Not a valid WAVE file: {path}")

        fmt_chunk = None
        data_chunk = None

        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = struct.unpack("<I", chunk_header[4:])[0]

            if chunk_id == b"fmt ":
                fmt_chunk = f.read(chunk_size)
                if chunk_size % 2 == 1:
                    f.read(1)  # Pad byte
            elif chunk_id == b"data":
                data_chunk = f.read(chunk_size)
                if chunk_size % 2 == 1:
                    f.read(1)  # Pad byte
                break
            else:
                # Skip unknown chunks (JUNK, LIST, etc.)
                f.seek(chunk_size + (chunk_size % 2), 1)

    if not fmt_chunk or not data_chunk:
        raise ValueError(f"Missing fmt or data chunk in {path}")

    # Parse fmt chunk
    audio_format, n_channels, rate, _byte_rate, _block_align, bits_per_sample = struct.unpack(
        "<HHIIHH", fmt_chunk[:16]
    )

    is_float = (audio_format == 3)

    if audio_format == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE
        if len(fmt_chunk) >= 40:
            subformat_guid = fmt_chunk[24:40]
            subformat_tag = struct.unpack("<H", subformat_guid[:2])[0]
            if subformat_tag == 3:
                is_float = True
            elif subformat_tag == 1:
                is_float = False

    if is_float and bits_per_sample == 32:
        raw_float = np.frombuffer(data_chunk, dtype=np.float32)
        samples = raw_float.reshape(-1, n_channels).T
    elif bits_per_sample == 16:
        raw_int = np.frombuffer(data_chunk, dtype=np.int16)
        samples = (raw_int.astype(np.float32) / 32768.0).reshape(-1, n_channels).T
    elif bits_per_sample == 24:
        a = np.frombuffer(data_chunk, dtype=np.uint8)
        reshaped = a.reshape(-1, 3)
        ints = (reshaped[:, 0].astype(np.int32) |
                (reshaped[:, 1].astype(np.int32) << 8) |
                (reshaped[:, 2].astype(np.int32) << 16))
        ints[ints >= 0x800000] -= 0x1000000
        samples = (ints.astype(np.float32) / 8388608.0).reshape(-1, n_channels).T
    elif bits_per_sample == 32 and not is_float:
        raw_int = np.frombuffer(data_chunk, dtype=np.int32)
        samples = (raw_int.astype(np.float32) / 2147483648.0).reshape(-1, n_channels).T
    else:
        raise ValueError(f"Unsupported format: tag={audio_format}, bits={bits_per_sample}, float={is_float}")

    return AudioBuffer(samples, rate)


def write_wav(path: Path, audio: AudioBuffer, bit_depth: int = 16) -> None:
    """Writes an AudioBuffer to a standard 16-bit or 24-bit PCM WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_channels = audio.channels
    rate = audio.sample_rate
    num_samples = audio.num_samples

    # Interleave channels
    interleaved = audio.samples.T.flatten()

    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setframerate(rate)

        if bit_depth == 16:
            wf.setsampwidth(2)
            clipped = np.clip(interleaved, -1.0, 1.0)
            int_samples = (clipped * 32767.0).round().astype(np.int16)
            wf.writeframes(int_samples.tobytes())
        elif bit_depth == 24:
            wf.setsampwidth(3)
            clipped = np.clip(interleaved, -1.0, 1.0)
            int_samples = (clipped * 8388607.0).round().astype(np.int32)
            # pack 24-bit little endian
            b0 = (int_samples & 0xFF).astype(np.uint8)
            b1 = ((int_samples >> 8) & 0xFF).astype(np.uint8)
            b2 = ((int_samples >> 16) & 0xFF).astype(np.uint8)
            packed = np.column_stack([b0, b1, b2]).flatten()
            wf.writeframes(packed.tobytes())
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
