#!/usr/bin/env python3
"""Independent local FFmpeg metering; no audio processing or export."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_mid_profile import digest


def main():
    base = ROOT / 'runs/around-mastering-20261022/session2'
    complete = json.loads((base / 'COMPLETE.json').read_text())
    protocol = json.loads((base / 'protocol.json').read_text())
    if digest(base / 'protocol.json') != complete['protocol_sha256']:
        raise RuntimeError('render protocol changed')
    for path, expected in protocol['pins'].items():
        if digest(path) != expected:
            raise RuntimeError(f'qualification pin changed: {path}')
    out = ROOT / 'runs/around-analysis-20261022/independent-meters-v1'
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['LD_PRELOAD'] = '/data/data/com.termux/files/usr/lib/libc++_shared.so'
    rows = []
    for track, receipt in zip(protocol['plan']['tracks'], complete['rows']):
        for kind, path in [('source', Path(track['path'])),
                           ('exported_flac', base / track['id'] / 'mastered.flac')]:
            if (out / 'HALT').exists() or (out.parent / 'HALT').exists():
                raise RuntimeError('meter HALT')
            pinned = digest(path)
            if kind == 'exported_flac' and pinned != receipt['flac_sha256']:
                raise RuntimeError('exported master changed')
            command = ['ffmpeg', '-nostdin', '-hide_banner', '-nostats', '-i', str(path),
                '-map', '0:a:0', '-af', 'ebur128=peak=true:framelog=verbose', '-f', 'null', '-']
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
            log = out / f"{track['id']}-{kind}.log"
            with log.open('x') as f:
                f.write(result.stderr)
            if result.returncode:
                raise RuntimeError(f'meter failed: {track["id"]}/{kind}')
            values = {}
            for field, pattern in [('lufs_integrated', r'\bI:\s*([-+\d.]+)\s*LUFS'),
                                   ('lra_lu', r'\bLRA:\s*([-+\d.]+)\s*LU'),
                                   ('true_peak_dbtp', r'\bPeak:\s*([-+\d.]+)\s*dBFS')]:
                matches = re.findall(pattern, result.stderr)
                if not matches:
                    raise RuntimeError(f'missing {field}')
                values[field] = float(matches[-1])
            if digest(path) != pinned:
                raise RuntimeError('audio changed during scan')
            rows.append({'id': track['id'], 'kind': kind, 'path': str(path), 'sha256': pinned,
                         **values, 'argv': command, 'log_sha256': digest(log)})
    report = {'schema': 'around-independent-ebu-meter-v1', 'rows': rows,
        'tool_version': subprocess.check_output(['ffmpeg', '-version'], env=env, text=True).splitlines()[0],
        'precision_note': 'Installed FFmpeg scanner reports summary to0.1LU/dB; finite oversampled estimate, not continuous-time guarantee.',
        'native_results_separate': True, 'audio_changed': False}
    path = Path(__file__).with_name('independent_meters.json')
    with path.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'metered_files': len(rows), 'receipt': str(path)}))


if __name__ == '__main__':
    main()
