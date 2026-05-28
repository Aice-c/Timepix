import math
import unittest

import numpy as np

from ProcessProgram.Particle.stage2_extract_cluster_features import compute_cluster_features


class Stage2ClusterFeatureTest(unittest.TestCase):
    def test_compact_square_features(self) -> None:
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        features = compute_cluster_features(matrix)

        self.assertEqual(features["Npix"], 4)
        self.assertAlmostEqual(features["S_total_ToT"], 10.0)
        self.assertAlmostEqual(features["Pmax"], 0.4)
        self.assertAlmostEqual(features["Rg"], math.sqrt(0.5))
        self.assertAlmostEqual(features["E_pca"], 1.0)
        self.assertAlmostEqual(features["Fbox"], 1.0)

    def test_line_like_features_are_more_elongated(self) -> None:
        matrix = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)

        features = compute_cluster_features(matrix)

        self.assertEqual(features["Npix"], 3)
        self.assertAlmostEqual(features["S_total_ToT"], 6.0)
        self.assertAlmostEqual(features["Pmax"], 0.5)
        self.assertAlmostEqual(features["Rg"], math.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(features["Fbox"], 1.0)
        self.assertGreater(features["E_pca"], 1.8)

    def test_empty_matrix_returns_zero_features(self) -> None:
        matrix = np.zeros((4, 4), dtype=np.float32)

        features = compute_cluster_features(matrix)

        self.assertEqual(features["Npix"], 0)
        self.assertEqual(features["S_total_ToT"], 0.0)
        self.assertEqual(features["Pmax"], 0.0)
        self.assertEqual(features["Rg"], 0.0)
        self.assertEqual(features["E_pca"], 1.0)
        self.assertEqual(features["Fbox"], 0.0)


if __name__ == "__main__":
    unittest.main()
