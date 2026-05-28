import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ProcessProgram.Particle.stage3b_export_cleaned_dataset import export_cleaned_dataset


class Stage3bExportCleanedDatasetTests(unittest.TestCase):
    def test_exports_only_recommended_keep_pairs_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage1 = root / "stage1"
            dataset = stage1 / "dataset"
            (dataset / "Am" / "ToT").mkdir(parents=True)
            (dataset / "Am" / "ToA").mkdir(parents=True)
            (dataset / "Co60" / "ToT").mkdir(parents=True)
            (dataset / "Co60" / "ToA").mkdir(parents=True)

            np.savetxt(dataset / "Am" / "ToT" / "keep.txt", np.ones((2, 2)), fmt="%.0f")
            np.savetxt(dataset / "Am" / "ToA" / "keep.txt", np.ones((2, 2)) * 2, fmt="%.0f")
            np.savetxt(dataset / "Am" / "ToT" / "reject.txt", np.ones((2, 2)) * 3, fmt="%.0f")
            np.savetxt(dataset / "Am" / "ToA" / "reject.txt", np.ones((2, 2)) * 4, fmt="%.0f")
            np.savetxt(dataset / "Co60" / "ToT" / "keep.txt", np.ones((2, 2)) * 5, fmt="%.0f")
            np.savetxt(dataset / "Co60" / "ToA" / "keep.txt", np.ones((2, 2)) * 6, fmt="%.0f")

            audit = pd.DataFrame(
                {
                    "sample_key": ["am_keep", "am_reject", "co_keep"],
                    "particle": ["Am", "Am", "Co60"],
                    "tot_path": ["Am/ToT/keep.txt", "Am/ToT/reject.txt", "Co60/ToT/keep.txt"],
                    "toa_path": ["Am/ToA/keep.txt", "Am/ToA/reject.txt", "Co60/ToA/keep.txt"],
                    "recommended_keep": [True, False, True],
                    "reject_reasons": ["keep", "am_low_npix", "keep"],
                    "review_flags": ["none", "none", "extreme_large_component"],
                }
            )
            audit_path = root / "source_cleaning_audit.csv"
            audit.to_csv(audit_path, index=False)

            output = root / "cleaned"
            summary = export_cleaned_dataset(stage1, audit_path, output)

            self.assertEqual(summary["exported_count"], 2)
            self.assertEqual(summary["rejected_count"], 1)
            self.assertTrue((output / "dataset" / "Am" / "ToT" / "keep.txt").is_file())
            self.assertTrue((output / "dataset" / "Am" / "ToA" / "keep.txt").is_file())
            self.assertFalse((output / "dataset" / "Am" / "ToT" / "reject.txt").exists())
            self.assertTrue((output / "dataset" / "Co60" / "ToT" / "keep.txt").is_file())

            manifest = pd.read_csv(output / "manifests" / "cleaned_manifest.csv")
            self.assertEqual(manifest["sample_key"].tolist(), ["am_keep", "co_keep"])
            self.assertIn("review_flags", manifest.columns)
            rejected = pd.read_csv(output / "manifests" / "cleaned_rejected_manifest.csv")
            self.assertEqual(rejected["sample_key"].tolist(), ["am_reject"])


if __name__ == "__main__":
    unittest.main()
