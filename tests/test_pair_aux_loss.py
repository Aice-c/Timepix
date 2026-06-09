import unittest

import torch
import torch.nn.functional as F

from timepix.losses import build_loss


class PairAuxLossTests(unittest.TestCase):
    def test_ce_pair_aux_applies_binary_loss_only_to_pair_targets(self):
        cfg = {
            "task": {"type": "classification"},
            "loss": {
                "name": "ce_pair_aux",
                "label_encoding": "onehot",
                "class_weight": "balanced",
                "pair_aux": {
                    "classes": ["Co", "Sr"],
                    "weight": 0.5,
                    "class_weight": "balanced",
                },
            },
        }
        label_map = {0: "Am", 1: "Co", 2: "P", 3: "Sr"}
        class_counts = [8, 80, 20, 10]
        logits = torch.tensor(
            [
                [0.1, 2.0, 0.2, 1.0],
                [0.0, 1.0, 0.1, 2.5],
                [3.0, 0.5, 0.2, 0.1],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([1, 3, 0], dtype=torch.long)

        loss = build_loss(
            cfg,
            num_classes=4,
            label_map=label_map,
            label_type="categorical_folder",
            class_counts=class_counts,
        )

        full_weights = torch.tensor(
            [
                118 / (4 * 8),
                118 / (4 * 80),
                118 / (4 * 20),
                118 / (4 * 10),
            ],
            dtype=torch.float32,
        )
        pair_weights = torch.tensor([90 / (2 * 80), 90 / (2 * 10)], dtype=torch.float32)
        expected = F.cross_entropy(logits, targets, weight=full_weights) + 0.5 * F.cross_entropy(
            logits[:2][:, [1, 3]],
            torch.tensor([0, 1], dtype=torch.long),
            weight=pair_weights,
        )

        self.assertTrue(torch.allclose(loss(logits, targets), expected))


if __name__ == "__main__":
    unittest.main()
