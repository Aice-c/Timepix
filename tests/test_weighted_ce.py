import torch

from timepix.losses import build_loss


def test_build_loss_balanced_class_weights_from_counts():
    cfg = {
        "task": {"type": "classification"},
        "loss": {
            "name": "cross_entropy",
            "label_encoding": "onehot",
            "class_weight": "balanced",
        },
    }
    label_map = {0: "Am", 1: "Co60", 2: "Sr"}

    loss = build_loss(
        cfg,
        num_classes=3,
        label_map=label_map,
        label_type="categorical_folder",
        class_counts=[10, 80, 10],
    )

    expected = torch.tensor([100 / 30, 100 / 240, 100 / 30], dtype=torch.float32)
    assert torch.allclose(loss.class_weights.cpu(), expected)
