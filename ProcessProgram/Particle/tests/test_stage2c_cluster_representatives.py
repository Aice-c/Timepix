import unittest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ProcessProgram.Particle.stage2c_cluster_representatives import sample_rows_by_particle_cluster


class ClusterRepresentativeSamplingTest(unittest.TestCase):
    def test_samples_each_particle_cluster_independently(self) -> None:
        df = pd.DataFrame(
            {
                "sample_key": [f"s{i}" for i in range(10)],
                "particle": ["A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
                "gmm_k3_label": [0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
                "gmm_k3_confidence": [0.95, 0.40, 0.99, 0.98, 0.91, 0.92, 0.10, 0.93, 0.94, 0.95],
            }
        )

        sampled = sample_rows_by_particle_cluster(
            df,
            label_column="gmm_k3_label",
            confidence_column="gmm_k3_confidence",
            samples_per_cluster=2,
            min_confidence=0.9,
            seed=42,
        )

        counts = sampled.groupby(["particle", "gmm_k3_label"]).size().to_dict()
        self.assertEqual(counts[("A", 0)], 1)
        self.assertEqual(counts[("A", 1)], 2)
        self.assertEqual(counts[("B", 0)], 2)
        self.assertEqual(counts[("B", 1)], 2)
        self.assertIn("cluster_total_count", sampled.columns)
        self.assertIn("cluster_high_conf_count", sampled.columns)


if __name__ == "__main__":
    unittest.main()
