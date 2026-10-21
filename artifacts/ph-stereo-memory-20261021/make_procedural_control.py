#!/usr/bin/env python3
"""Regenerate the procedural parity fixture without invoking a mastering run."""
import argparse
import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from long_coherence_fixture import _write_float_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Same formula as master-stage-audit-20261020/validate_engaged.py, without
    # invoking its unrelated render suite or needing any historical binaries.
    t = np.arange(96000, dtype=np.float64) / 48000
    level = np.where(t < 1 / 3, .02, 1.)
    x = level * (.13 * np.sin(2 * np.pi * 130 * t)
                 + .07 * np.sin(2 * np.pi * 440 * t)
                 + .04 * np.sin(2 * np.pi * 6300 * t))
    x[48000:48048] += 1.3
    audio = np.stack([x, .91 * x], axis=1).astype(np.float32)
    with tempfile.TemporaryDirectory() as temp:
        wave = Path(temp) / 'control.wav'
        _write_float_wav(wave, audio)
        data = wave.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as out:
        out.write(data)
    print(hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
