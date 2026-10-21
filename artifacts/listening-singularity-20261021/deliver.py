#!/usr/bin/env python3
"""Copy only the verified listener payload; never overwrite an earlier session."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    base = ROOT / 'runs/listening-singularity-20261021'
    ready = json.loads((base / 'LISTENING_READY.json').read_text())
    for parent in [base, base / 'fullsong-v1']:
        if any((parent / name).exists() for name in ['HALT', 'HALTED.json']):
            raise RuntimeError('latched session; no delivery')
    source = Path(ready['listener_dir'])
    expected = ready['files_sha256']
    paths = {str(p.relative_to(source)): p for p in source.rglob('*') if p.is_file()}
    allowed = {'INSTRUCTIONS.txt', 'responses.csv', 'notes.txt', 'READY'} | {
        f'T{i:04d}/{slot}.wav' for i in range(1, 5) for slot in ['A', 'B']}
    if set(paths) != allowed or set(expected) != allowed:
        raise RuntimeError('unexpected listener payload')
    if any(p.is_symlink() or sha(p) != expected[name] for name, p in paths.items()):
        raise RuntimeError('listener payload changed')
    destination = Path('/sdcard/Download/Roach Listening Singularity 20261021')
    destination.mkdir()  # fail on any existing path, including a partial copy
    # READY is last; interrupted delivery never carries a readiness marker.
    for name in sorted(allowed - {'READY'}):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[name], target)
        if sha(target) != expected[name]:
            raise RuntimeError(f'copy hash mismatch: {name}')
    for parent in [base, base / 'fullsong-v1']:
        if any((parent / name).exists() for name in ['HALT', 'HALTED.json']):
            raise RuntimeError('session halted during delivery')
    shutil.copy2(paths['READY'], destination / 'READY')
    if sha(destination / 'READY') != expected['READY']:
        (destination / 'READY').unlink()
        raise RuntimeError('copied readiness marker mismatch')
    with (Path(__file__).parent / 'delivery.json').open('x') as f:
        json.dump({'schema': 'cross-song-listener-delivery-v1', 'destination': str(destination),
                   'source': str(source), 'files_sha256': expected,
                   'verified_copy_files': len(expected), 'source_assignments_or_metrics_in_payload': False,
                   'readiness_copied_last': True}, f, indent=2)
        f.write('\n')
    print(json.dumps({'ready': True, 'folder': str(destination), 'files': len(expected)}))


if __name__ == '__main__':
    main()
