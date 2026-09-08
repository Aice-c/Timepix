#!/usr/bin/env python
"""Prepare immutable frame-group manifests from filenames and prior provenance only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timepix.data.dataset import collect_samples
from timepix.data.frame_groups import event_frame_stem, frame_key, grouped_manifest, manifest_digest
from timepix.training.validation_export import chance_references

TASKS = {'T7': ['10', '20', '30', '45', '50', '60', '70'],
         'V6': ['80', '82', '84', '86', '88', '90']}


def archive_index(path):
    index = defaultdict(list)
    with zipfile.ZipFile(path) as archive:
        for original in archive.namelist():
            member = PurePosixPath(original.replace('\\', '/'))
            if member.suffix.lower() == '.txt' and member.parent.name.isdigit():
                index[(member.parent.name, member.stem)].append(member.as_posix())
    return index


def source_rows(records, index, prior, aliases):
    rows = []
    for r in records:
        name = r.modalities['ToT'].name
        stem = event_frame_stem(name)
        matches = index.get((r.angle, stem), [])
        if len(matches) != 1:
            raise ValueError(f'Ambiguous or missing raw frame for {r.key}: {matches}')
        member = matches[0]
        old = prior.get(r.key)
        if old and (str(old['angle']) != r.angle or old['raw_frame_member'].replace('\\', '/') != member):
            raise ValueError(f'Existing verified provenance disagrees: {r.key}')
        canonical = aliases.get(member, member)
        if canonical not in index.get((r.angle, PurePosixPath(canonical).stem), []):
            raise ValueError(f'Alias target not in same-angle archive: {canonical}')
        rows.append(dict(sample_key=r.key, angle=r.angle, event_relative_path=f'{r.angle}/ToT/{name}',
                         raw_archive='C.zip', raw_frame_member=member, canonical_frame_member=canonical,
                         frame_group_key=frame_key(r.angle, canonical),
                         source_directory_group=PurePosixPath(member).parts[0],
                         provenance='reused_verified_mapping' if old else 'unique_archive_index_match',
                         historical_split_full=old.get('split_full_manifest', 'not_audited') if old else 'not_audited',
                         historical_split_t7=old.get('split_7class_manifest', 'not_audited') if old else 'not_audited'))
    return rows


def immutable_text(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise FileExistsError(f'Refusing to overwrite different artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        f.write(text)


def json_text(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2) + '\n'


def csv_text(rows, fields=None):
    rows = list(rows)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields or list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return buffer.getvalue()


def supplemental_csv(rows):
    return csv_text(rows, ['sample_key', 'angle', 'event_relative_path', 'raw_archive', 'raw_frame_member',
                          'canonical_frame_member', 'frame_group_key', 'source_directory_group',
                          'provenance', 'historical_split_full', 'historical_split_t7'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--raw-archive', default=str(ROOT / 'Data/C.zip'))
    parser.add_argument('--prior-events', default=str(ROOT / 'outputs/proton_near_vertical_diagnostic_20260908/event_statistics.csv'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/carbon_angle_controls_20260909'))
    parser.add_argument('--aliases', help='Optional JSON {alias_member: canonical_member}; requires independently verified evidence')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    aliases = json.loads(Path(args.aliases).read_text(encoding='utf-8')) if args.aliases else {}
    if any(v in aliases and aliases[v] != v for v in aliases.values()):
        raise ValueError('Alias map must be flattened to canonical members')
    prior = {}
    with Path(args.prior_events).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['sample_key'] in prior:
                raise ValueError(f'Duplicate prior event: {row["sample_key"]}')
            prior[row['sample_key']] = row
    index = archive_index(args.raw_archive)
    summary = []; tasks = {}; added = []; reuse = []
    for task, angles in TASKS.items():
        records, labels = collect_samples(args.data_root, ['ToT'], class_names=angles)
        rows = source_rows(records, index, prior, aliases)
        if len(rows) != len({r['sample_key'] for r in rows}):
            raise ValueError('Duplicate angle-qualified event keys')
        manifest = grouped_manifest(rows, seed=args.seed, aliases=aliases)
        manifest.update(task_id=task, class_names=angles, data_version='existing_Proton_C_selected_events',
                        split_method='per-angle shuffled source frames; floor frame counts, event ratios approximate',
                        alias_map=aliases, test_is_new_untouched=False,
                        test_history='Same acquisition previously analyzed and event-split evaluated; reserved only for this protocol')
        manifest['manifest_sha256'] = manifest_digest(manifest)
        immutable_text(output / f'{task}_frame_split.json', json_text(manifest))
        assignments = {key: split for split in ('train', 'val', 'test') for key in manifest[split]}
        membership = [dict(r, split=assignments[r['sample_key']]) for r in rows]
        immutable_text(output / f'{task}_event_membership.csv', csv_text(membership))
        frame_rows = []
        for angle in angles:
            all_angle = [r for r in membership if r['angle'] == angle]
            for split in ('train', 'val', 'test'):
                selected = [r for r in all_angle if r['split'] == split]
                summary.append(dict(task_id=task, angle=angle, split=split, events=len(selected),
                                    source_frames=len({r['frame_group_key'] for r in selected}),
                                    event_fraction=len(selected) / len(all_angle)))
        by_frame = defaultdict(list)
        for r in membership:
            by_frame[r['frame_group_key']].append(r)
        for key, group in sorted(by_frame.items()):
            frame_rows.append(dict(angle=group[0]['angle'], frame_group_key=key,
                                   raw_frame_members='|'.join(sorted({r['raw_frame_member'] for r in group})),
                                   split=group[0]['split'], events=len(group), split_count=len({r['split'] for r in group})))
        immutable_text(output / f'{task}_frame_audit.csv', csv_text(frame_rows))
        label_ids = {v: k for k, v in labels.items()}
        y = {s: [label_ids[manifest['samples'][k]['angle']] for k in manifest[s]] for s in ('train', 'val')}
        immutable_text(output / f'{task}_chance_references.json', json_text(chance_references(y['train'], y['val'], list(map(float, angles)))))
        tasks[task] = dict(events=len(rows), source_frames=len(by_frame), class_names=angles,
                           manifest_sha256=manifest['manifest_sha256'], cross_split_frames=0,
                           split_counts={s: len(manifest[s]) for s in ('train', 'val', 'test')})
        added.extend(r for r in rows if r['provenance'] != 'reused_verified_mapping')
        reuse.append(dict(task_id=task, experiment_id=f'{task}-' + ('ToT' if task == 'T7' else 'Base'),
                          reuse='historical_reference_only' if task == 'T7' else 'not_found',
                          evidence='outputs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed' if task == 'T7' else 'searched configs, experiments, diagnostic reports',
                          reason='Old event split not source-frame isolated' if task == 'T7' else 'No compatible full-matrix six-class formal CNN located',
                          training_status='not_started'))
        reuse.append(dict(task_id=task, experiment_id=f'{task}-' + ('Mask' if task == 'T7' else 'HiRes'),
                          reuse='not_found', evidence='searched configs/experiments',
                          reason='New controlled input/stride setting', training_status='not_started'))
    immutable_text(output / 'group_split_summary.csv', csv_text(summary))
    immutable_text(output / 'supplemental_source_mapping.csv', supplemental_csv(added))
    immutable_text(output / 'reuse_inventory.csv', csv_text(reuse))
    inventory = dict(data_root=str(Path(args.data_root).resolve()), raw_archive=str(Path(args.raw_archive).resolve()),
                     prior_events=str(Path(args.prior_events).resolve()), tasks=tasks,
                     supplemental_mapping_events=len(added), source_check='filenames and archive members, not full pixel provenance',
                     training_started=False, old_outputs_modified=False, aliases=aliases)
    immutable_text(output / 'preparation_verification.json', json_text(inventory))
    print(json_text(inventory))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
