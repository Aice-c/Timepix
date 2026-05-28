#!/usr/bin/env python3
"""Extract paired single-particle ToT/ToA candidate matrices from raw frames.

Stage 1 only performs technical extraction:
- pair raw ToT/ToA 256x256 frames,
- split each frame into connected components,
- place each accepted component bbox at the center of a fixed-size canvas,
- save paired ToT/ToA matrices and extraction manifests.

Physics/data-quality filtering based on active pixels, total ToT, or mean ToT is
intentionally left for the next stage after inspecting the extracted candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_RAW_ROOT = Path(r"E:\TimepixData\particle\raw")
DEFAULT_OUTPUT_ROOT = Path(r"E:\TimepixData\particle\stage1_single_particle_candidates_100x100")

NEIGHBOR_OFFSETS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class RawPair:
    particle: str
    source_subdir: str
    pair_stem: str
    tot_path: Path
    toa_path: Path


@dataclass(frozen=True)
class Component:
    coords: np.ndarray
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    touches_edge: bool

    @property
    def pixel_count(self) -> int:
        return int(self.coords.shape[0])

    @property
    def height(self) -> int:
        return self.row_max - self.row_min + 1

    @property
    def width(self) -> int:
        return self.col_max - self.col_min + 1


def strip_modality_suffix(stem: str, modality: str) -> str:
    suffix = f"_{modality}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem.replace(suffix, "")


def slugify(value: str) -> str:
    value = value.replace("\\", "_").replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "root"


def condition_label(particle: str, source_subdir: str) -> str:
    text = source_subdir.replace("\\", "/")
    text_no_particle = re.sub(re.escape(particle), "", text, flags=re.IGNORECASE)
    numbers = re.findall(r"\d+(?:\.\d+)?", text_no_particle)
    if numbers:
        value = numbers[-1].replace(".", "p")
        return f"angle{value}"
    return slugify(source_subdir)


def list_raw_pairs(raw_root: Path) -> tuple[list[RawPair], list[dict[str, str]]]:
    pairs: list[RawPair] = []
    audit_rows: list[dict[str, str]] = []
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw particle root does not exist: {raw_root}")

    for particle_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        particle = particle_dir.name
        grouped: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
        for path in sorted(particle_dir.rglob("*.txt")):
            stem = path.stem
            if "_ToT" in stem:
                pair_stem = strip_modality_suffix(stem, "ToT")
                rel_parent = path.parent.relative_to(particle_dir).as_posix()
                grouped[(rel_parent, pair_stem)]["ToT"] = path
            elif "_ToA" in stem:
                pair_stem = strip_modality_suffix(stem, "ToA")
                rel_parent = path.parent.relative_to(particle_dir).as_posix()
                grouped[(rel_parent, pair_stem)]["ToA"] = path

        for (source_subdir, pair_stem), modalities in sorted(grouped.items()):
            has_tot = "ToT" in modalities
            has_toa = "ToA" in modalities
            status = "paired" if has_tot and has_toa else "missing_modality"
            audit_rows.append(
                {
                    "particle": particle,
                    "source_subdir": source_subdir,
                    "pair_stem": pair_stem,
                    "has_ToT": str(has_tot),
                    "has_ToA": str(has_toa),
                    "status": status,
                    "tot_path": str(modalities.get("ToT", "")),
                    "toa_path": str(modalities.get("ToA", "")),
                }
            )
            if has_tot and has_toa:
                pairs.append(
                    RawPair(
                        particle=particle,
                        source_subdir=source_subdir,
                        pair_stem=pair_stem,
                        tot_path=modalities["ToT"],
                        toa_path=modalities["ToA"],
                    )
                )
    return pairs, audit_rows


def load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def extract_components(mask: np.ndarray) -> list[Component]:
    rows, cols = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[Component] = []

    active_coords = np.argwhere(mask)
    for start_r, start_c in active_coords:
        start_r = int(start_r)
        start_c = int(start_c)
        if visited[start_r, start_c]:
            continue

        stack = [(start_r, start_c)]
        visited[start_r, start_c] = True
        coords: list[tuple[int, int]] = []
        row_min, row_max = rows, -1
        col_min, col_max = cols, -1
        touches_edge = False

        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            row_min = min(row_min, r)
            row_max = max(row_max, r)
            col_min = min(col_min, c)
            col_max = max(col_max, c)
            if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
                touches_edge = True

            for dr, dc in NEIGHBOR_OFFSETS_8:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

        components.append(
            Component(
                coords=np.asarray(coords, dtype=np.int32),
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
                touches_edge=touches_edge,
            )
        )

    components.sort(key=lambda item: item.pixel_count, reverse=True)
    return components


def build_component_label_map(shape: tuple[int, int], components: list[Component]) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    for label, component in enumerate(components, start=1):
        labels[component.coords[:, 0], component.coords[:, 1]] = label
    return labels


def build_bbox_centered_component_canvas(source: np.ndarray, component: Component, target_size: int) -> np.ndarray:
    out_top = (target_size - component.height) // 2
    out_left = (target_size - component.width) // 2
    local_rows = component.coords[:, 0] - component.row_min
    local_cols = component.coords[:, 1] - component.col_min
    out_rows = out_top + local_rows
    out_cols = out_left + local_cols
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    canvas[out_rows, out_cols] = source[component.coords[:, 0], component.coords[:, 1]]
    return canvas


def save_matrix(path: Path, matrix: np.ndarray, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt=fmt)


def summarize_values(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "total_ToT": 0.0,
            "mean_ToT_nonzero": 0.0,
            "max_ToT": 0.0,
        }
    return {
        "total_ToT": float(values.sum()),
        "mean_ToT_nonzero": float(values.mean()),
        "max_ToT": float(values.max()),
    }


def compare_component_toa_tot_region(
    component: Component,
    tot: np.ndarray,
    toa_label_map: np.ndarray,
    toa_components: list[Component],
) -> dict[str, int | bool]:
    tot_on_component = tot[component.coords[:, 0], component.coords[:, 1]] > 0
    tot_coords = component.coords[tot_on_component]
    toa_labels_on_tot = toa_label_map[tot_coords[:, 0], tot_coords[:, 1]]
    overlap_labels = sorted(int(label) for label in np.unique(toa_labels_on_tot) if int(label) > 0)

    tot_coord_set = {tuple(coord) for coord in tot_coords.tolist()}
    toa_coord_set: set[tuple[int, int]] = set()
    for label in overlap_labels:
        toa_component = toa_components[label - 1]
        toa_coord_set.update(tuple(coord) for coord in toa_component.coords.tolist())

    missing_toa_pixels = len(tot_coord_set - toa_coord_set)
    extra_toa_pixels = len(toa_coord_set - tot_coord_set)
    return {
        "match": bool(len(overlap_labels) == 1 and missing_toa_pixels == 0 and extra_toa_pixels == 0),
        "matched_toa_component_count": int(len(overlap_labels)),
        "toa_active_on_component_pixels": int((toa_labels_on_tot > 0).sum()),
        "toa_missing_pixels": int(missing_toa_pixels),
        "toa_extra_pixels": int(extra_toa_pixels),
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sample_key_for(raw_pair: RawPair, component_index: int) -> str:
    source_slug = slugify(raw_pair.source_subdir)
    pair_slug = slugify(raw_pair.pair_stem)
    return f"{raw_pair.particle}_{source_slug}_{pair_slug}_c{component_index:03d}"


def process_pairs(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    dataset_root = output_root / "dataset"
    manifest_root = output_root / "manifests"

    pairs, pairing_rows = list_raw_pairs(raw_root)
    if args.limit_pairs_per_class is not None:
        limited: list[RawPair] = []
        counts: dict[str, int] = defaultdict(int)
        for pair in pairs:
            if counts[pair.particle] < args.limit_pairs_per_class:
                limited.append(pair)
                counts[pair.particle] += 1
        pairs = limited

    accepted_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    summary = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "target_size": args.target_size,
        "component_mask": args.component_mask,
        "reject_edge": bool(args.reject_edge),
        "min_component_pixels": int(args.min_component_pixels),
        "paired_frames_processed": 0,
        "saved_components": 0,
        "rejected_components": 0,
        "failed_pairs": 0,
        "dry_run": bool(args.dry_run),
        "by_particle": defaultdict(lambda: {"pairs": 0, "saved": 0, "rejected": 0}),
        "discovered_pairing_status": Counter(row["status"] for row in pairing_rows),
        "reject_reasons": Counter(),
    }

    for raw_pair in pairs:
        summary["paired_frames_processed"] += 1
        summary["by_particle"][raw_pair.particle]["pairs"] += 1
        try:
            tot = load_matrix(raw_pair.tot_path)
            toa = load_matrix(raw_pair.toa_path)
        except Exception as exc:  # noqa: BLE001 - keep processing remaining files.
            summary["failed_pairs"] += 1
            summary["reject_reasons"]["load_failed"] += 1
            rejected_rows.append(
                {
                    "particle": raw_pair.particle,
                    "source_subdir": raw_pair.source_subdir,
                    "pair_stem": raw_pair.pair_stem,
                    "component_index": "",
                    "reject_reason": f"load_failed:{exc}",
                    "active_pixel_count": "",
                    "matched_toa_component_count": "",
                    "toa_active_on_component_pixels": "",
                    "toa_missing_pixels": "",
                    "toa_extra_pixels": "",
                    "total_ToT": "",
                    "mean_ToT_nonzero": "",
                    "bbox_height": "",
                    "bbox_width": "",
                    "raw_tot_path": str(raw_pair.tot_path),
                    "raw_toa_path": str(raw_pair.toa_path),
                }
            )
            continue

        if tot.shape != toa.shape or tot.ndim != 2:
            summary["failed_pairs"] += 1
            summary["reject_reasons"]["shape_mismatch"] += 1
            rejected_rows.append(
                {
                    "particle": raw_pair.particle,
                    "source_subdir": raw_pair.source_subdir,
                    "pair_stem": raw_pair.pair_stem,
                    "component_index": "",
                    "reject_reason": f"shape_mismatch:{tot.shape}/{toa.shape}",
                    "active_pixel_count": "",
                    "matched_toa_component_count": "",
                    "toa_active_on_component_pixels": "",
                    "toa_missing_pixels": "",
                    "toa_extra_pixels": "",
                    "total_ToT": "",
                    "mean_ToT_nonzero": "",
                    "bbox_height": "",
                    "bbox_width": "",
                    "raw_tot_path": str(raw_pair.tot_path),
                    "raw_toa_path": str(raw_pair.toa_path),
                }
            )
            continue

        mask = tot > 0
        if args.component_mask == "union":
            mask = mask | (toa > 0)
        components = extract_components(mask)
        toa_components = extract_components(toa > 0)
        toa_label_map = build_component_label_map(toa.shape, toa_components)

        for component_index, component in enumerate(components, start=1):
            tot_values = tot[component.coords[:, 0], component.coords[:, 1]]
            value_summary = summarize_values(tot_values[tot_values > 0])
            region_summary = compare_component_toa_tot_region(component, tot, toa_label_map, toa_components)
            reject_reason = ""
            if component.pixel_count < args.min_component_pixels:
                reject_reason = "too_few_pixels"
            elif args.reject_edge and component.touches_edge:
                reject_reason = "touches_detector_edge"
            elif component.height > args.target_size or component.width > args.target_size:
                reject_reason = "component_larger_than_target"
            elif not region_summary["match"]:
                reject_reason = "toa_tot_region_mismatch"

            if reject_reason:
                summary["rejected_components"] += 1
                summary["by_particle"][raw_pair.particle]["rejected"] += 1
                summary["reject_reasons"][reject_reason] += 1
                rejected_rows.append(
                    {
                        "particle": raw_pair.particle,
                        "source_subdir": raw_pair.source_subdir,
                        "pair_stem": raw_pair.pair_stem,
                        "component_index": component_index,
                        "reject_reason": reject_reason,
                        "active_pixel_count": component.pixel_count,
                        "matched_toa_component_count": region_summary["matched_toa_component_count"],
                        "toa_active_on_component_pixels": region_summary["toa_active_on_component_pixels"],
                        "toa_missing_pixels": region_summary["toa_missing_pixels"],
                        "toa_extra_pixels": region_summary["toa_extra_pixels"],
                        "total_ToT": value_summary["total_ToT"],
                        "mean_ToT_nonzero": value_summary["mean_ToT_nonzero"],
                        "bbox_height": component.height,
                        "bbox_width": component.width,
                        "raw_tot_path": str(raw_pair.tot_path),
                        "raw_toa_path": str(raw_pair.toa_path),
                    }
                )
                continue

            sample_key = sample_key_for(raw_pair, component_index)
            tot_out_rel = Path(raw_pair.particle) / "ToT" / f"{sample_key}_ToT.txt"
            toa_out_rel = Path(raw_pair.particle) / "ToA" / f"{sample_key}_ToA.txt"
            tot_out_path = dataset_root / tot_out_rel
            toa_out_path = dataset_root / toa_out_rel

            tot_canvas = build_bbox_centered_component_canvas(tot, component, args.target_size)
            toa_canvas = build_bbox_centered_component_canvas(toa, component, args.target_size)

            if not args.dry_run:
                save_matrix(tot_out_path, tot_canvas, args.output_format)
                save_matrix(toa_out_path, toa_canvas, args.output_format)

            summary["saved_components"] += 1
            summary["by_particle"][raw_pair.particle]["saved"] += 1
            accepted_rows.append(
                {
                    "sample_key": sample_key,
                    "particle": raw_pair.particle,
                    "condition_label": condition_label(raw_pair.particle, raw_pair.source_subdir),
                    "source_subdir": raw_pair.source_subdir,
                    "raw_pair_key": f"{raw_pair.source_subdir}/{raw_pair.pair_stem}",
                    "component_index": component_index,
                    "active_pixel_count": component.pixel_count,
                    "matched_toa_component_count": region_summary["matched_toa_component_count"],
                    "toa_active_on_component_pixels": region_summary["toa_active_on_component_pixels"],
                    "total_ToT": value_summary["total_ToT"],
                    "mean_ToT_nonzero": value_summary["mean_ToT_nonzero"],
                    "max_ToT": value_summary["max_ToT"],
                    "bbox_height": component.height,
                    "bbox_width": component.width,
                    "bbox_long": max(component.height, component.width),
                    "bbox_short": min(component.height, component.width),
                    "touches_edge": str(component.touches_edge),
                    "tot_path": str(tot_out_rel).replace("\\", "/"),
                    "toa_path": str(toa_out_rel).replace("\\", "/"),
                    "raw_tot_path": str(raw_pair.tot_path),
                    "raw_toa_path": str(raw_pair.toa_path),
                }
            )

    accepted_fields = [
        "sample_key",
        "particle",
        "condition_label",
        "source_subdir",
        "raw_pair_key",
        "component_index",
        "active_pixel_count",
        "matched_toa_component_count",
        "toa_active_on_component_pixels",
        "total_ToT",
        "mean_ToT_nonzero",
        "max_ToT",
        "bbox_height",
        "bbox_width",
        "bbox_long",
        "bbox_short",
        "touches_edge",
        "tot_path",
        "toa_path",
        "raw_tot_path",
        "raw_toa_path",
    ]
    rejected_fields = [
        "particle",
        "source_subdir",
        "pair_stem",
        "component_index",
        "reject_reason",
        "active_pixel_count",
        "matched_toa_component_count",
        "toa_active_on_component_pixels",
        "toa_missing_pixels",
        "toa_extra_pixels",
        "total_ToT",
        "mean_ToT_nonzero",
        "bbox_height",
        "bbox_width",
        "raw_tot_path",
        "raw_toa_path",
    ]
    pairing_fields = [
        "particle",
        "source_subdir",
        "pair_stem",
        "has_ToT",
        "has_ToA",
        "status",
        "tot_path",
        "toa_path",
    ]

    if not args.dry_run:
        write_csv(manifest_root / "extraction_manifest.csv", accepted_rows, accepted_fields)
        write_csv(manifest_root / "rejected_components.csv", rejected_rows, rejected_fields)
        write_csv(manifest_root / "pairing_audit.csv", pairing_rows, pairing_fields)
        summary_for_json = dict(summary)
        summary_for_json["by_particle"] = dict(summary["by_particle"])
        summary_for_json["discovered_pairing_status"] = dict(summary["discovered_pairing_status"])
        summary_for_json["reject_reasons"] = dict(summary["reject_reasons"])
        (manifest_root / "summary.json").write_text(
            json.dumps(summary_for_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Raw root       :", raw_root)
    print("Output root    :", output_root)
    print("Dataset root   :", dataset_root)
    print("Dry run        :", args.dry_run)
    print("Paired frames  :", summary["paired_frames_processed"])
    print("Saved samples  :", summary["saved_components"])
    print("Rejected comps :", summary["rejected_components"])
    print("Failed pairs   :", summary["failed_pairs"])
    if summary["discovered_pairing_status"]:
        print("Discovered pairing status:")
        for status, count in sorted(summary["discovered_pairing_status"].items()):
            print(f"  {status}: {count}")
    for particle, stats in sorted(summary["by_particle"].items()):
        print(f"  {particle}: pairs={stats['pairs']} saved={stats['saved']} rejected={stats['rejected']}")
    if summary["reject_reasons"]:
        print("Reject reasons :")
        for reason, count in sorted(summary["reject_reasons"].items()):
            print(f"  {reason}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract stage-1 paired single-particle ToT/ToA candidates from "
            "raw 256x256 particle frames."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help=f"Raw input root. Default: {DEFAULT_RAW_ROOT}")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Stage-1 output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument("--target-size", type=int, default=100, help="Centered output matrix size. Default: 100")
    parser.add_argument(
        "--component-mask",
        choices=["tot", "union"],
        default="tot",
        help="Mask used to find connected components. Default: tot",
    )
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=1,
        help="Technical minimum component size. Keep at 1 for stage-1 candidate extraction.",
    )
    parser.add_argument(
        "--allow-edge",
        dest="reject_edge",
        action="store_false",
        help="Keep components touching the detector edge. Default rejects edge-touching components.",
    )
    parser.set_defaults(reject_edge=True)
    parser.add_argument(
        "--output-format",
        default="%.6f",
        help="np.savetxt format for output matrices. Default: %%.6f",
    )
    parser.add_argument(
        "--limit-pairs-per-class",
        type=int,
        default=None,
        help="Debug option: process at most N paired frames per particle class.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run extraction and print counts without writing files.")
    return parser.parse_args()


def main() -> None:
    process_pairs(parse_args())


if __name__ == "__main__":
    main()
