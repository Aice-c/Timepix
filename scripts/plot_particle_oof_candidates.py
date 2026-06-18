#!/usr/bin/env python3
"""Plot paired ToT/ToA samples from P9a OOF diagnostic candidate tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIAG_DIR = Path("outputs/diagnostics/p9a_ptype_stage1_gmm02_p_v3_oof5_seed42")
DEFAULT_DATA_ROOT = Path(r"E:/TimepixData/particle/datasets/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3/dataset")
DEFAULT_OUT_DIR = Path("outputs/p9b_oof_candidate_samples")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _matrix_path(data_root: Path, sample_key: str, modality: str) -> Path:
    label, key = sample_key.split("/", 1)
    stem = Path(key).stem
    suffix = Path(key).suffix or ".txt"
    return data_root / label / modality / f"{stem}{modality}{suffix}"


def _load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.loadtxt(path, dtype=np.float32)


def _tight_crop(tot: np.ndarray, toa: np.ndarray, pad: int = 1) -> tuple[np.ndarray, np.ndarray]:
    mask = (tot > 0) | (toa != 0)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    r0 = max(0, int(r0) - pad)
    c0 = max(0, int(c0) - pad)
    r1 = min(tot.shape[0], int(r1) + pad)
    c1 = min(tot.shape[1], int(c1) + pad)
    return tot[r0:r1, c0:c1], toa[r0:r1, c0:c1]


def _relative_toa(toa: np.ndarray) -> np.ndarray:
    out = np.zeros_like(toa, dtype=np.float32)
    mask = toa != 0
    if not mask.any():
        return out
    vals = toa[mask].astype(np.float32)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo:
        out[mask] = 1.0
    else:
        out[mask] = (vals - lo) / (hi - lo)
    return out


def _transition(row: dict[str, str]) -> str:
    true_class = row.get("true_class", "")
    pred = row.get("gmu_totstrong_pred") or row.get("tot_pred") or row.get("pred_class") or ""
    return f"{true_class}_to_{pred}"


def _select(rows: list[dict[str, str]], transition: str, limit: int) -> list[dict[str, str]]:
    matched = [row for row in rows if _transition(row) == transition]
    return matched[:limit]


def _title(row: dict[str, str]) -> str:
    key = row["sample_key"].split("/", 1)[1].replace(".txt", "")
    true_class = row.get("true_class", "")
    tot = row.get("tot_pred", "")
    dual = row.get("dual_concat_pred", "")
    gmu = row.get("gmu_totstrong_pred", "")
    return f"{key}\nT:{true_class} | ToT:{tot} D:{dual} G:{gmu}"


def _plot_group(rows: list[dict[str, str]], data_root: Path, out_base: Path, cols: int) -> None:
    if not rows:
        return
    n = len(rows)
    cols = max(1, min(cols, n))
    sample_row_blocks = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        sample_row_blocks * 2,
        cols,
        figsize=(2.1 * cols, 2.6 * sample_row_blocks),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(sample_row_blocks * 2, cols)
    for ax in axes.ravel():
        ax.axis("off")

    cache: list[tuple[dict[str, str], np.ndarray, np.ndarray]] = []
    tot_scale: list[float] = []
    for row in rows:
        tot = _load_matrix(_matrix_path(data_root, row["sample_key"], "ToT"))
        toa = _load_matrix(_matrix_path(data_root, row["sample_key"], "ToA"))
        tot_crop, toa_crop = _tight_crop(tot, toa)
        cache.append((row, tot_crop, toa_crop))
        vals = tot_crop[tot_crop > 0]
        if vals.size:
            tot_scale.extend(np.log1p(vals).tolist())
    tot_vmax = float(np.quantile(tot_scale, 0.98)) if tot_scale else 1.0

    for idx, (row, tot_crop, toa_crop) in enumerate(cache):
        block = idx // cols
        col = idx % cols
        ax_tot = axes[block * 2, col]
        ax_toa = axes[block * 2 + 1, col]
        ax_tot.imshow(np.log1p(tot_crop), cmap="magma", interpolation="nearest", vmin=0, vmax=tot_vmax)
        ax_toa.imshow(_relative_toa(toa_crop), cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
        ax_tot.set_title(_title(row), fontsize=6)
        if col == 0:
            ax_tot.set_ylabel("ToT", fontsize=7)
            ax_toa.set_ylabel("RToA", fontsize=7)
        ax_tot.set_xticks([])
        ax_tot.set_yticks([])
        ax_toa.set_xticks([])
        ax_toa.set_yticks([])

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_manifest(path: Path, groups: dict[str, list[dict[str, str]]]) -> None:
    fieldnames = [
        "group",
        "sample_key",
        "true_class",
        "tot_pred",
        "dual_concat_pred",
        "gmu_totstrong_pred",
        "tot_confidence",
        "dual_concat_confidence",
        "gmu_totstrong_confidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group, rows in groups.items():
            for row in rows:
                writer.writerow({name: (group if name == "group" else row.get(name, "")) for name in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag-dir", type=Path, default=DEFAULT_DIAG_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--per-transition", type=int, default=20)
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    all_wrong = _read_rows(args.diag_dir / "all_wrong_same_pred_samples.csv")
    model_specific = _read_rows(args.diag_dir / "model_specific_failure_samples.csv")

    groups: dict[str, list[dict[str, str]]] = {}
    for transition in ["Sr_to_Co", "Co_to_Sr", "Am_to_P", "P_to_Am", "Sr_to_Am"]:
        groups[f"all_wrong_same_{transition}"] = _select(all_wrong, transition, args.per_transition)

    by_outcome: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in model_specific:
        by_outcome[row.get("outcome", "")].append(row)
    for outcome in [
        "tot_dual_concat_correct_gmu_totstrong_wrong",
        "tot_gmu_totstrong_correct_dual_concat_wrong",
        "dual_concat_gmu_totstrong_correct_tot_wrong",
        "only_gmu_totstrong_correct",
        "only_dual_concat_correct",
        "only_tot_correct",
    ]:
        groups[outcome] = by_outcome.get(outcome, [])[: args.per_transition]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.out_dir / "p9b_sample_manifest.csv", groups)
    for group, rows in groups.items():
        _plot_group(rows, args.data_root, args.out_dir / f"{group}", args.cols)

    print(f"Wrote manifest: {args.out_dir / 'p9b_sample_manifest.csv'}")
    print(f"Wrote figures to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
