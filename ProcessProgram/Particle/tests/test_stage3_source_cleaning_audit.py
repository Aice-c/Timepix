import unittest

import pandas as pd

from ProcessProgram.Particle.stage3a_source_cleaning_audit import (
    apply_cleaning_flags,
    build_group_thresholds,
    otsu_threshold_1d,
)


class Stage3SourceCleaningAuditTests(unittest.TestCase):
    def test_otsu_threshold_separates_low_and_high_npix_modes(self) -> None:
        values = [1, 1, 2, 2, 3, 35, 36, 38, 40, 42]

        threshold = otsu_threshold_1d(values)

        self.assertGreaterEqual(threshold, 3)
        self.assertLess(threshold, 35)

    def test_am_rule_rejects_low_active_pixel_candidates(self) -> None:
        df = pd.DataFrame(
            {
                "sample_key": ["low", "main"],
                "particle": ["Am", "Am"],
                "angle": [0, 0],
                "Npix": [2, 35],
                "S_total_ToT": [30.0, 2500.0],
                "Pmax": [0.9, 0.2],
                "Rg": [0.0, 3.0],
                "E_pca": [1.0, 1.5],
                "Fbox": [1.0, 0.8],
            }
        )
        thresholds = build_group_thresholds(df, group_cols=["particle", "angle"])

        out = apply_cleaning_flags(df, thresholds, am_npix_threshold=10)

        low = out[out["sample_key"] == "low"].iloc[0]
        main = out[out["sample_key"] == "main"].iloc[0]
        self.assertFalse(bool(low["recommended_keep"]))
        self.assertIn("am_low_npix", low["reject_reasons"])
        self.assertTrue(bool(main["recommended_keep"]))

    def test_cosr_rule_rejects_low_signal_but_keeps_extreme_tracks_for_review(self) -> None:
        df = pd.DataFrame(
            {
                "sample_key": ["normal_small", "low_noise", "huge"],
                "particle": ["Co60", "Co60", "Co60"],
                "angle": [0, 0, 0],
                "Npix": [3, 1, 200],
                "S_total_ToT": [80.0, 1.0, 20000.0],
                "Pmax": [0.6, 1.0, 0.05],
                "Rg": [1.0, 0.0, 35.0],
                "E_pca": [1.5, 1.0, 20.0],
                "Fbox": [0.8, 1.0, 0.02],
            }
        )
        thresholds = build_group_thresholds(df, group_cols=["particle", "angle"])

        out = apply_cleaning_flags(df, thresholds, am_npix_threshold=10)

        normal = out[out["sample_key"] == "normal_small"].iloc[0]
        low_noise = out[out["sample_key"] == "low_noise"].iloc[0]
        huge = out[out["sample_key"] == "huge"].iloc[0]
        self.assertTrue(bool(normal["recommended_keep"]))
        self.assertFalse(bool(low_noise["recommended_keep"]))
        self.assertIn("low_signal_noise_like", low_noise["reject_reasons"])
        self.assertTrue(bool(huge["recommended_keep"]))
        self.assertEqual("keep", huge["reject_reasons"])
        self.assertIn("extreme_large_component", huge["review_flags"])


if __name__ == "__main__":
    unittest.main()
