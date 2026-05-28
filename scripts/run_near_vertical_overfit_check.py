#!/usr/bin/env python
"""Small-sample overfit sanity check for near-vertical Proton/C ToT data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.analysis.io import numeric_angle_dirs, read_matrix
from timepix.analysis.plotting import _plt, plot_confusion_matrix, save_figure


class MatrixDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int]) -> None:
        self.paths = paths
        self.labels = labels
        self.arrays = [self._load(path) for path in paths]

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        x = read_matrix(path).astype(np.float32, copy=False)
        x = np.log1p(np.clip(x, 0, None))
        nonzero = x[x != 0]
        if nonzero.size:
            mean = float(nonzero.mean())
            std = float(nonzero.std() or 1.0)
            x = (x - mean) / max(std, 1e-6)
        return x.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        x = self.arrays[idx]
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run near-vertical small-sample overfit sanity check")
    parser.add_argument("--dataset-root", required=True, help="Concrete Proton_C dataset directory")
    parser.add_argument("--output-root", default="outputs/resolution_limit")
    parser.add_argument("--angles", nargs="+", type=float, default=[80, 82, 84, 86, 88, 90])
    parser.add_argument("--modality", default="ToT")
    parser.add_argument("--samples-per-class", nargs="+", type=int, default=[5, 10, 50, 100])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-train-acc", type=float, default=0.995)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def list_paths(dataset_root: Path, angles: list[float], modality: str) -> dict[float, list[Path]]:
    available = {float(p.name): p for p in numeric_angle_dirs(dataset_root)}
    out: dict[float, list[Path]] = {}
    for angle in angles:
        angle_dir = available.get(float(angle))
        if angle_dir is None:
            raise FileNotFoundError(f"Missing angle directory: {angle:g}")
        modality_dir = angle_dir / modality
        if not modality_dir.is_dir():
            raise FileNotFoundError(f"Missing modality directory: {modality_dir}")
        paths = sorted(p for p in modality_dir.glob("*.txt") if p.is_file())
        if not paths:
            raise FileNotFoundError(f"No .txt files in {modality_dir}")
        out[float(angle)] = paths
    return out


def split_paths(paths_by_angle: dict[float, list[Path]], samples_per_class: int, seed: int):
    rng = random.Random(seed + samples_per_class)
    train_paths: list[Path] = []
    train_labels: list[int] = []
    val_paths: list[Path] = []
    val_labels: list[int] = []
    labels = {angle: idx for idx, angle in enumerate(sorted(paths_by_angle))}
    for angle, paths in sorted(paths_by_angle.items()):
        shuffled = list(paths)
        rng.shuffle(shuffled)
        train = shuffled[:samples_per_class]
        val_n = min(max(samples_per_class, 20), 200, max(0, len(shuffled) - samples_per_class))
        val = shuffled[samples_per_class : samples_per_class + val_n]
        train_paths.extend(train)
        train_labels.extend([labels[angle]] * len(train))
        val_paths.extend(val)
        val_labels.extend([labels[angle]] * len(val))
    return train_paths, train_labels, val_paths, val_labels, labels


def accuracy_and_cm(model, loader, device: torch.device, num_classes: int):
    model.eval()
    correct = 0
    total = 0
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            for yt, yp in zip(y.cpu().numpy(), pred.cpu().numpy()):
                cm[int(yt), int(yp)] += 1
    return correct / max(total, 1), cm


def train_one(train_ds, val_ds, labels: dict[float, int], args, device: torch.device, samples_per_class: int):
    torch.manual_seed(args.seed + samples_per_class)
    model = TinyCNN(num_classes=len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    best_val_acc = 0.0
    final_cm = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(y.numel())
            n += int(y.numel())
        train_acc, _ = accuracy_and_cm(model, eval_train_loader, device, len(labels))
        val_acc, final_cm = accuracy_and_cm(model, val_loader, device, len(labels))
        best_val_acc = max(best_val_acc, val_acc)
        rows.append(
            {
                "samples_per_class": samples_per_class,
                "epoch": epoch,
                "train_loss": loss_sum / max(n, 1),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "best_val_acc_so_far": best_val_acc,
            }
        )
        if train_acc >= args.target_train_acc:
            break
    return pd.DataFrame(rows), model, final_cm


def plot_curves(curves: pd.DataFrame, out_path: Path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for spc, group in curves.groupby("samples_per_class"):
        axes[0].plot(group["epoch"], group["train_loss"], label=f"{spc}/class")
        axes[1].plot(group["epoch"], group["train_acc"], label=f"Train {spc}/class")
        axes[1].plot(group["epoch"], group["val_acc"], linestyle="--", label=f"Val {spc}/class")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Train/Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.22)
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8, ncol=2)
    return save_figure(fig, out_path)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output_root = Path(args.output_root)
    figures_dir = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths_by_angle = list_paths(Path(args.dataset_root), args.angles, args.modality)
    all_curves = []
    summary_rows = []
    confusion_payload = {}
    angle_labels = [f"{a:g}" for a in sorted(paths_by_angle)]
    for spc in args.samples_per_class:
        train_paths, train_labels, val_paths, val_labels, label_map = split_paths(paths_by_angle, spc, args.seed)
        train_ds = MatrixDataset(train_paths, train_labels)
        val_ds = MatrixDataset(val_paths, val_labels)
        curves, _model, cm = train_one(train_ds, val_ds, label_map, args, device, spc)
        all_curves.append(curves)
        last = curves.iloc[-1]
        summary_rows.append(
            {
                "samples_per_class": spc,
                "epochs_run": int(last["epoch"]),
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "final_train_loss": float(last["train_loss"]),
                "final_train_acc": float(last["train_acc"]),
                "final_val_acc": float(last["val_acc"]),
                "best_val_acc": float(curves["val_acc"].max()),
                "reached_target_train_acc": bool(last["train_acc"] >= args.target_train_acc),
                "augmentation": "off",
                "dropout": 0.0,
                "weight_decay": 0.0,
                "early_stopping": "off",
                "device": str(device),
            }
        )
        if cm is not None:
            confusion_payload[str(spc)] = cm.tolist()
            plot_confusion_matrix(cm, angle_labels, figures_dir / f"11_overfit_check_confusion_matrix_spc{spc}", f"Overfit Check Confusion Matrix ({spc}/class)")
    curves_df = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    curves_df.to_csv(output_root / "11_overfit_check_learning_curve.csv", index=False, encoding="utf-8-sig")
    curves_df.to_csv(output_root / "near_vertical_overfit_learning_curve.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "11_overfit_check_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "near_vertical_overfit_experiment.csv", index=False, encoding="utf-8-sig")
    (output_root / "11_overfit_check_confusion_matrices.json").write_text(json.dumps(confusion_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not curves_df.empty:
        plot_curves(curves_df, figures_dir / "11_overfit_check_train_curves")
        plot_curves(curves_df, figures_dir / "near_vertical_overfit_learning_curve")
    print(f"Wrote near-vertical overfit sanity check to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
