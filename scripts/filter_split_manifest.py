#!/usr/bin/env python
"""Filter a split manifest by sample-key prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter train/val/test split keys by prefix")
    parser.add_argument("--input", required=True, help="Input split manifest JSON")
    parser.add_argument("--output", required=True, help="Output filtered split manifest JSON")
    parser.add_argument(
        "--prefix",
        action="append",
        required=True,
        help="Sample-key prefix to keep, e.g. Co/. Can be provided multiple times.",
    )
    return parser.parse_args()


def _filter_keys(keys: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [key for key in keys if key.startswith(prefixes)]


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    prefixes = tuple(str(item) for item in args.prefix)

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    filtered: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        if split not in payload:
            raise KeyError(f"Missing split '{split}' in {in_path}")
        filtered[split] = _filter_keys(list(payload[split]), prefixes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote filtered split: {out_path}")
    print(f"Prefixes: {', '.join(prefixes)}")
    for split in ("train", "val", "test"):
        print(f"{split}: {len(payload[split])} -> {len(filtered[split])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
