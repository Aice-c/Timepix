import unittest

from scripts.analyze_gmu_gate_behavior import summarize_gate_rows


class GmuGateAnalysisTests(unittest.TestCase):
    def test_summarizes_overall_class_correctness_and_cosr_groups(self):
        rows = [
            {"split": "val", "true_class": "Co", "pred_class": "Co", "correct": 1, "gate_tot": 0.8, "gate_toa": 0.2},
            {"split": "val", "true_class": "Co", "pred_class": "Sr", "correct": 0, "gate_tot": 0.6, "gate_toa": 0.4},
            {"split": "val", "true_class": "Sr", "pred_class": "Sr", "correct": 1, "gate_tot": 0.7, "gate_toa": 0.3},
            {"split": "val", "true_class": "Sr", "pred_class": "Co", "correct": 0, "gate_tot": 0.5, "gate_toa": 0.5},
            {"split": "test", "true_class": "Am", "pred_class": "Am", "correct": 1, "gate_tot": 0.9, "gate_toa": 0.1},
        ]

        summary = summarize_gate_rows(rows)
        keyed = {(row["split"], row["group_type"], row["group"]): row for row in summary}

        self.assertEqual(keyed[("val", "overall", "all")]["n"], 4)
        self.assertAlmostEqual(keyed[("val", "overall", "all")]["gate_tot_mean"], 0.65)
        self.assertAlmostEqual(keyed[("val", "true_class", "Co")]["gate_toa_mean"], 0.3)
        self.assertAlmostEqual(keyed[("val", "correctness", "wrong")]["gate_tot_mean"], 0.55)
        self.assertEqual(keyed[("val", "cosr_case", "Co_to_Sr")]["n"], 1)
        self.assertAlmostEqual(keyed[("val", "cosr_case", "Sr_to_Co")]["gate_toa_mean"], 0.5)
        self.assertEqual(keyed[("test", "true_class", "Am")]["n"], 1)


if __name__ == "__main__":
    unittest.main()
