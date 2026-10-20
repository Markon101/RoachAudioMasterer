import importlib.util
from pathlib import Path
import unittest

MODULE = Path(__file__).with_name('analyze_passive_routing_audit.py')
spec = importlib.util.spec_from_file_location('passive_analysis', MODULE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def profile(sub=-30.0, low_mid=-20.0, metal=0.0, instability=0.0,
            cutoff=22_000.0, codec=0.0, mono=False):
    return {
        'sub_energy_dbfs': sub, 'low_mid_energy_dbfs': low_mid,
        'mid_energy_dbfs': -5.0, 'high_energy_dbfs': -20.0,
        'ultrasonic_energy_dbfs': -25.0, 'estimated_cutoff_hz': cutoff,
        'interchannel_correlation': 0.5, 'side_to_mid_ratio': 0.1,
        'sub_side_leak_ratio': 0.0, 'crest_factor_db': 6.0,
        'geometry_channels': 1 if mono else 2,
        'artifacts': {'metallic_grain': metal, 'sub_instability': instability,
                      'codec_swish': codec, 'ai_shimmer': 0.0, 'phase_haze': 0.0},
        'authorities': {},
    }


class AuthorityRuleTests(unittest.TestCase):
    def test_legacy_score_threshold_crossings_are_attributed_as_score_gates(self):
        a, basis_a = analysis.legacy_authorities(profile(sub=-31.0, low_mid=-20.0))
        b, basis_b = analysis.legacy_authorities(profile(sub=-40.0, low_mid=-39.0))
        self.assertEqual((a['sub_bass_authority'], a['mid_flow_authority']), (0.0, 0.0))
        self.assertEqual((b['sub_bass_authority'], b['mid_flow_authority']), (0.5, 0.6))
        left = {'profile': profile(sub=-31.0), 'basis': basis_a}
        right = {'profile': profile(sub=-40.0), 'basis': basis_b}
        self.assertEqual(analysis.transition_class('sub', left, right), 'legacy_score_threshold')
        left['profile']['low_mid_energy_dbfs'] = -20.0
        right['profile']['low_mid_energy_dbfs'] = -39.0
        self.assertEqual(analysis.transition_class('mid', left, right), 'legacy_score_threshold')

    def test_artifact_floor_and_cutoff_triggers_remain_distinct(self):
        floor, basis_floor = analysis.legacy_authorities(profile(sub=-120.0, low_mid=-120.0))
        self.assertEqual(floor['sub_bass_authority'], 0.85)
        self.assertEqual(floor['mid_flow_authority'], 0.6)
        self.assertTrue(basis_floor['sub_score_at_floor'])
        artifact, _ = analysis.legacy_authorities(profile(sub=-20.0, instability=0.5))
        self.assertEqual(artifact['sub_bass_authority'], 0.85)
        narrow, _ = analysis.legacy_authorities(profile(cutoff=8_000.0))
        self.assertEqual(narrow['high_field_authority'], 0.9)


if __name__ == '__main__':
    unittest.main()
