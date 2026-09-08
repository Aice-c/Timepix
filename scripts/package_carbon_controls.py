#!/usr/bin/env python
"""Package only generated carbon protocol artifacts; code must still deploy via git."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_carbon_controls import immutable_text, json_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'outputs/carbon_angle_controls_20260909')
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Package source must be inside project outputs')
    names = ['T7_frame_split.json', 'V6_frame_split.json', 'T7_frame_audit.csv', 'V6_frame_audit.csv',
             'group_split_summary.csv', 'preparation_verification.json', 'local_preflight.json',
             'reuse_inventory.csv', 'reuse_and_experiment_report.md',
             'T7_chance_references.json', 'V6_chance_references.json', 'supplemental_source_mapping.csv']
    files = [directory / name for name in names]
    if any(not p.is_file() for p in files):
        raise FileNotFoundError('Preparation/preflight/report incomplete')
    destination = directory / 'carbon_controls_protocol.zip'
    if destination.exists():
        raise FileExistsError('Package already exists; retain it, do not overwrite')
    entries = []
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            member = path.relative_to(ROOT).as_posix()
            z.write(path, member)
            entries.append(dict(member=member, bytes=path.stat().st_size,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with zipfile.ZipFile(destination) as z:
        for entry in entries:
            if hashlib.sha256(z.read(entry['member'])).hexdigest() != entry['sha256']:
                raise ValueError(f'Package verification failed: {entry["member"]}')
    result = dict(package=str(destination), bytes=destination.stat().st_size,
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), members=entries,
                  includes_raw_data=False, includes_checkpoints=False, includes_code=False)
    immutable_text(directory / 'protocol_package_verification.json', json_text(result))
    print(json_text({k: v for k, v in result.items() if k != 'members'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
