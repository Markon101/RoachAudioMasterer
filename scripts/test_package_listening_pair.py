"""Verify the packaging qualification chain using tiny isolated files only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


RECIPE = (Path(__file__).resolve().parents[1] /
          'artifacts/listening-singularity-20261021/package_pair.py')


class PackageQualificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        spec = importlib.util.spec_from_file_location('_isolated_package_recipe', RECIPE)
        self.recipe = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', list(sys.path)):
            spec.loader.exec_module(self.recipe)
        self.recipe.ROOT = self.root
        self.recipe.__file__ = str(self.root / 'package_pair.py')
        packaging = patch.object(self.recipe.blind, 'prepare',
                                 side_effect=AssertionError('packaging must not run in these tests'))
        packaging.start()
        self.addCleanup(packaging.stop)

        self.plan_path = self.root / 'plan.json'
        self.revision_path = self.root / 'plan_revision_v2.json'
        self.binary_path = self.root / 'frozen-native'
        self.original_plan = {
            'source': {'rate': 48000, 'channels': 2, 'frames': 6293504},
            'gates': ['original prerequisite', 'unchanged quality gates']}
        self.write_json(self.plan_path, self.original_plan)
        self.write_json(self.revision_path, {
            'original_plan_sha256': self.digest(self.plan_path),
            'replacement_prerequisite': 'reviewed prerequisite'})
        self.binary_path.write_bytes(b'fixture native pin; never executable')
        effective_plan = json.loads(json.dumps(self.original_plan))
        effective_plan['gates'][0] = 'reviewed prerequisite'
        self.protocol = {
            'plan': effective_plan,
            'pins': {str(p): self.digest(p)
                     for p in (self.plan_path, self.revision_path, self.binary_path)}}
        self.protocol_path = self.run / 'protocol.json'
        self.write_json(self.protocol_path, self.protocol)
        self.receipt = {'eligible_for_listening': True,
                        'protocol_sha256': self.digest(self.protocol_path)}
        self.write_json(self.run / 'COMPLETE.json', self.receipt)

    @staticmethod
    def write_json(path, data):
        path.write_text(json.dumps(data) + '\n')

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_valid_chain_returns_verified_receipt_and_effective_plan_snapshot(self):
        receipt, protocol = self.recipe.verify_qualification(self.run)
        self.assertEqual(receipt, self.receipt)
        self.assertEqual(protocol, self.protocol)
        self.assertEqual(protocol['plan']['source'], self.original_plan['source'])
        self.assertEqual(protocol['plan']['gates'][0], 'reviewed prerequisite')
        self.assertEqual(json.loads(self.plan_path.read_text()), self.original_plan)
        self.assertNotEqual(protocol['plan'], self.original_plan)

    def test_mutated_protocol_rejected_even_when_json_still_valid(self):
        with self.protocol_path.open('a') as f:
            f.write(' \n')
        with self.assertRaisesRegex(RuntimeError, 'qualified protocol changed'):
            self.recipe.verify_qualification(self.run)

    def test_mutated_original_plan_rejected(self):
        changed = json.loads(json.dumps(self.original_plan))
        changed['source']['frames'] += 1
        self.write_json(self.plan_path, changed)
        with self.assertRaisesRegex(RuntimeError, 'render qualification pin changed') as caught:
            self.recipe.verify_qualification(self.run)
        self.assertIn(str(self.plan_path), str(caught.exception))

    def test_mutated_revision_or_binary_pin_rejected(self):
        for path in (self.revision_path, self.binary_path):
            with self.subTest(pin=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b' ')
                try:
                    with self.assertRaisesRegex(RuntimeError, 'render qualification pin changed') as caught:
                        self.recipe.verify_qualification(self.run)
                    self.assertIn(str(path), str(caught.exception))
                finally:
                    path.write_bytes(original)

    def test_noneligible_receipt_refused_before_hash_work(self):
        self.receipt['eligible_for_listening'] = False
        self.write_json(self.run / 'COMPLETE.json', self.receipt)
        with patch.object(self.recipe.blind, 'sha256') as digest:
            with self.assertRaisesRegex(RuntimeError, 'full-song gates incomplete'):
                self.recipe.verify_qualification(self.run)
            digest.assert_not_called()


if __name__ == '__main__':
    unittest.main()
