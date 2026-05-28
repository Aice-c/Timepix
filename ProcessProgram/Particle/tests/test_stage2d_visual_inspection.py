import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ProcessProgram.Particle.stage2d_visual_cluster_inspection import (
    SCALED_FEATURE_COLUMNS,
    build_interactive_hover_text,
    compute_pca_table,
    fit_gmm_reference,
    make_particle_pca_table,
    sample_for_plot,
)


class VisualInspectionTest(unittest.TestCase):
    def test_pca_uses_scaled_feature_columns(self) -> None:
        df = pd.DataFrame(
            {
                "sample_key": [f"s{i}" for i in range(6)],
                "particle": ["Co60"] * 6,
                "scaled_Npix": [-2, -1, 0, 1, 2, 3],
                "scaled_S_total_ToT": [-2, -1, 0, 1, 2, 3],
                "scaled_Pmax": [2, 1, 0, -1, -2, -3],
                "scaled_Rg": [-2, -1, 0, 1, 2, 3],
                "scaled_E_pca": [-2, -1, 0, 1, 2, 3],
                "scaled_Fbox": [2, 1, 0, -1, -2, -3],
            }
        )

        scores, explained, loadings = compute_pca_table(df)

        self.assertEqual(list(SCALED_FEATURE_COLUMNS), list(loadings["feature"]))
        self.assertEqual(["PC1", "PC2", "PC3"], list(scores.columns))
        self.assertEqual(len(scores), len(df))
        self.assertEqual(len(explained), 3)
        self.assertGreater(explained[0], 0)

    def test_sampling_is_reproducible_and_respects_limit(self) -> None:
        df = pd.DataFrame({"value": np.arange(100)})

        a = sample_for_plot(df, sample_size=10, seed=7)
        b = sample_for_plot(df, sample_size=10, seed=7)
        c = sample_for_plot(df, sample_size=200, seed=7)

        self.assertEqual(a["value"].tolist(), b["value"].tolist())
        self.assertEqual(len(a), 10)
        self.assertEqual(len(c), 100)

    def test_gmm_reference_returns_labels_and_confidence(self) -> None:
        scores = pd.DataFrame(
            {
                "PC1": [-3, -2, -1, 1, 2, 3],
                "PC2": [-0.2, 0.0, 0.2, -0.2, 0.0, 0.2],
            }
        )

        labels, confidence = fit_gmm_reference(scores, k=2, seed=13)

        self.assertEqual(len(labels), len(scores))
        self.assertEqual(len(confidence), len(scores))
        self.assertEqual(set(labels), {0, 1})
        self.assertTrue(np.all(confidence >= 0.0))
        self.assertTrue(np.all(confidence <= 1.0))

    def test_particle_pca_table_keeps_angle_metadata(self) -> None:
        df = pd.DataFrame(
            {
                "sample_key": [f"s{i}" for i in range(6)],
                "particle": ["Sr"] * 6,
                "condition_label": ["angle0", "angle0", "angle15", "angle15", "angle30", "angle30"],
                "source_subdir": ["0", "0", "15", "15", "30", "30"],
                "angle": [0, 0, 15, 15, 30, 30],
                "Npix": [1, 2, 3, 4, 5, 6],
                "S_total_ToT": [10, 20, 30, 40, 50, 60],
                "Pmax": [1.0, 0.8, 0.6, 0.5, 0.4, 0.3],
                "Rg": [0, 1, 1, 2, 2, 3],
                "E_pca": [1, 1.2, 1.4, 1.6, 1.8, 2.0],
                "Fbox": [1.0, 0.8, 0.7, 0.6, 0.5, 0.4],
                "scaled_Npix": [-2, -1, 0, 1, 2, 3],
                "scaled_S_total_ToT": [-3, -2, 0, 1, 2, 4],
                "scaled_Pmax": [2, 1, 0, -1, -2, -3],
                "scaled_Rg": [-2, -1, 0, 1, 2, 3],
                "scaled_E_pca": [3, 1, 0, -1, -2, -4],
                "scaled_Fbox": [2, 1, 0, -1, -2, -3],
            }
        )
        scores, _, _ = compute_pca_table(df)

        table = make_particle_pca_table(df, scores, seed=3)

        self.assertIn("angle", table.columns)
        self.assertIn("condition_label", table.columns)
        self.assertIn("gmm_pca_k2_reference", table.columns)
        self.assertIn("gmm_pca_k2_confidence", table.columns)
        self.assertIn("gmm_pca3_k3_reference", table.columns)
        self.assertIn("gmm_pca3_k3_confidence", table.columns)
        self.assertEqual(table["angle"].tolist(), [0, 0, 15, 15, 30, 30])

    def test_interactive_hover_text_contains_traceable_fields(self) -> None:
        df = pd.DataFrame(
            {
                "sample_key": ["s1"],
                "particle": ["Co60"],
                "angle": [45],
                "Npix": [7],
                "S_total_ToT": [123.45],
                "Pmax": [0.42],
                "Rg": [1.2],
                "E_pca": [2.5],
                "Fbox": [0.75],
                "gmm_pca3_k3_reference": [2],
                "gmm_pca3_k3_confidence": [0.91],
            }
        )

        text = build_interactive_hover_text(df)[0]

        self.assertIn("s1", text)
        self.assertIn("Co60", text)
        self.assertIn("angle: 45", text)
        self.assertIn("Npix: 7", text)
        self.assertIn("GMM3D: 2", text)


if __name__ == "__main__":
    unittest.main()
