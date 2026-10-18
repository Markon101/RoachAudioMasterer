#!/usr/bin/env python3
"""Compare the offline meter to actual Rust reference functions on procedural data.

Extracts unchanged functions from src/master.rs, compiles a standalone Rust
wrapper, and compares f32 inputs. No mastering, GPU, speed test or owner audio.
On Termux: LD_PRELOAD=/data/data/com.termux/files/usr/lib/libc++_shared.so
"""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
from roach_ears.dsp_metrics import _true_peak_linear


def function(source, name):
    start = source.index('pub fn ' + name + '(')
    body = source.index('{', start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def check():
    source = Path('src/master.rs').read_text()
    extracted = '\n'.join(function(source, n) for n in ['sample_peak', 'true_peak'])
    wrapper = '''use std::f32::consts::PI;
pub struct Audio { channels: Vec<Vec<f32>> }
'''+extracted+'''
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let b = std::fs::read(path).unwrap();
    let channels = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let frames = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    assert_eq!(b.len(), 8+channels*frames*4);
    let samples: Vec<f32> = b[8..].chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
    let audio = Audio { channels: samples.chunks(frames).map(|x| x.to_vec()).collect() };
    println!("{:.10}", true_peak(&audio));
}
'''
    rng = np.random.default_rng(38271)
    fixtures = {'silence': np.zeros((2, 96), dtype=np.float32),
                'nyquist_even': np.tile(.8*(-1.)**np.arange(128), (2, 1)),
                'nyquist_odd': np.tile(.8*(-1.)**np.arange(127), (2, 1)),
                'quiet_sine': np.tile(.4*np.cos(np.arange(256)*np.pi/2+np.pi/4), (2, 1)),
                'random_stereo': rng.uniform(-.95, .95, (2, 16417)),
                'short': np.array([[0., .8, -1.2], [.1, .2, .3]])}
    impulses = np.zeros((2, 16417), dtype=np.float32)
    impulses[0, [0, 11, 8191, 8192, 8193, 16416]] = 1.2
    fixtures['edges_and_seams'] = impulses
    results = []
    with tempfile.TemporaryDirectory(prefix='roach-tp-reference-') as directory:
        root = Path(directory)
        (root/'reference.rs').write_text(wrapper)
        subprocess.run(['rustc', '--edition=2021', '-O', str(root/'reference.rs'),
                        '-o', str(root/'reference')], check=True)
        for name, samples in fixtures.items():
            x = np.asarray(samples, dtype='<f4')
            (root/'fixture.bin').write_bytes(struct.pack('<II', *x.shape)+x.tobytes())
            native = float(subprocess.check_output([str(root/'reference'), str(root/'fixture.bin')], text=True))
            measured, overs = _true_peak_linear(x)
            deviation = abs(measured-native)
            assert deviation <= 2e-6, (name, measured, native, deviation)
            results.append({'fixture': name, 'native_peak': native, 'python_peak': measured,
                            'absolute_deviation': deviation, 'evaluated_points_over_unit': overs})
    return {'schema': 'roach-ears-true-peak-native-parity-v1',
            'native_functions_sha256': hashlib.sha256(extracted.encode()).hexdigest(),
            'python_source_sha256': hashlib.sha256(Path('scripts/roach_ears/dsp_metrics.py').read_bytes()).hexdigest(),
            'rustc': subprocess.check_output(['rustc', '-V'], text=True).strip(),
            'numpy': np.__version__, 'amplitude_tolerance': 2e-6, 'fixtures': results,
            'scope': 'Finite fourfold native estimator parity on procedural fixtures; no continuous-time or perceptual guarantee.'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2, allow_nan=False))
