"""Adversarial offline-analyzer checks: semantically inconsistent audits must be rejected.

Both cases derive a valid synthetic audit from `make_audit`, prove it is accepted,
then mutate exactly one semantic field while leaving every snapshot waveform intact.
"""
import json
from pathlib import Path
import tempfile
import unittest

from test_analyze_master_stage_audit import make_audit, analysis


def _rewrite_trace(audit_dir: Path, mutate) -> None:
    """Apply `mutate(row)` to every observations.jsonl row, preserving all other fields."""
    trace = audit_dir/'observations.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    for row in rows:
        mutate(row)
    trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))


class MasterAuditAdversarialTests(unittest.TestCase):
    def test_final_trim_gain_metadata_must_match_unmodified_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit_dir, fixed = make_audit(root, dynamic=False)
            # Baseline: the untampered fixture is accepted.
            analysis.analyze(audit_dir, fixed, root/'baseline.json')

            def shrink_trim(row):
                if row['meta']['stage'] == 'final_output':
                    self.assertEqual(row['meta']['final_trim_gain_linear'], 1.0)
                    row['meta']['final_trim_gain_linear'] = 0.5

            _rewrite_trace(audit_dir, shrink_trim)
            # The final-output WAV, its FNV, and the summary are untouched: only the
            # declared trim gain now contradicts the delivered waveform.
            with self.assertRaises(ValueError):
                analysis.analyze(audit_dir, fixed, root/'tampered.json')
            self.assertFalse((root/'tampered.json').exists())

    def test_limiter_backend_spec_must_be_a_known_kind(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit_dir, fixed = make_audit(root, dynamic=True)
            analysis.analyze(audit_dir, fixed, root/'baseline.json')

            def invalidate_limiter(row):
                if row['meta']['stage'] in ('pre_limiter', 'post_limiter'):
                    self.assertEqual(row['meta']['limiter']['kind'], 'legacy')
                    row['meta']['limiter'] = dict(row['meta']['limiter'],
                                                  kind='unknown_backend')

            _rewrite_trace(audit_dir, invalidate_limiter)
            with self.assertRaises(ValueError):
                analysis.analyze(audit_dir, fixed, root/'tampered.json')
            self.assertFalse((root/'tampered.json').exists())


if __name__ == '__main__':
    unittest.main()
