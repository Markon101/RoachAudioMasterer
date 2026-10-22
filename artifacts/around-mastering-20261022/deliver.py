#!/usr/bin/env python3
"""Copy the declared local listening package into a fresh Download directory."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    plan_path = HERE / 'delivery_plan.json'
    plan = json.loads(plan_path.read_text())
    run = ROOT / 'runs/around-mastering-20261022/session2'
    for parent in (HERE, run):
        if any((parent / name).exists() for name in ('HALT', 'HALTED.json')):
            raise RuntimeError('active session halted; preserve package state')
    complete = json.loads((run / 'COMPLETE.json').read_text())
    assert [r['qualified'] for r in complete['rows']] == [True, False]
    assert plan['v2_is_diagnostic'] is True and plan['production_promoted'] is False
    destination = Path(plan['destination'])
    assert destination.parent == Path('/sdcard/Download')
    assert len({r['filename'] for r in plan['files']}) == len(plan['files'])
    for row in plan['files']:
        assert Path(row['filename']).name == row['filename']
        assert digest(ROOT / row['source']) == row['sha256'], row['source']
    total = sum((ROOT / row['source']).stat().st_size for row in plan['files'])
    assert shutil.disk_usage(destination.parent).free > total + 4 * 1024 ** 3
    destination.mkdir(exist_ok=False)
    copied = []
    try:
        for row in plan['files']:
            if any((parent / name).exists() for parent in (HERE, run)
                   for name in ('HALT', 'HALTED.json')):
                raise RuntimeError('HALT during delivery')
            target = destination / row['filename']
            with (ROOT / row['source']).open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            assert digest(target) == row['sha256'], row['filename']
            copied.append({'filename': row['filename'], 'sha256': row['sha256'],
                           'bytes': target.stat().st_size})
        receipt = {'schema': 'around-listening-delivery-v1', 'destination': str(destination),
                   'plan_sha256': digest(plan_path), 'files': copied,
                   'all_hashes_match': True, 'v2_is_diagnostic': True,
                   'audio_modified_during_delivery': False, 'production_promoted': False}
        with (HERE / 'delivery_receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        (destination / 'READY.txt').write_text('Delivery verified. Read START_HERE.html. V2 is diagnostic.\n')
        print(json.dumps({'ready': str(destination), 'verified_files': len(copied)}))
    except BaseException as error:
        with (destination / 'DELIVERY_STOP.json').open('x') as stream:
            json.dump({'reason': str(error), 'verified_files': copied}, stream, indent=2)
        raise


if __name__ == '__main__':
    main()
