"""Explicit source-frame grouping, independent of particle-component suffixes."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import PurePosixPath
import random
import re


def event_frame_stem(filename: str) -> str:
    match = re.fullmatch(r'(C_r\d+_\d+)_\d+\.txt', filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f'Unrecognized carbon event filename: {filename}')
    return match.group(1)


def frame_key(angle: str, member: str) -> str:
    path = PurePosixPath(member.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or len(path.parts) < 3:
        raise ValueError(f'Expected complete archive member path: {member}')
    if path.parent.name != str(angle):
        raise ValueError(f'Frame angle does not match class {angle}: {member}')
    return json.dumps([str(angle), path.as_posix()], separators=(',', ':'))


def manifest_digest(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != 'manifest_sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def grouped_manifest(rows: list[dict], seed: int = 42, ratios=(.8, .1, .1), aliases: dict | None = None) -> dict:
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError('Three positive split ratios must sum to one')
    samples = {}
    groups = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: r['sample_key']):
        key = row['sample_key']
        if key in samples:
            raise ValueError(f'Duplicate sample key: {key}')
        angle = str(row['angle'])
        if not key.startswith(angle + '/'):
            raise ValueError(f'Sample key must include angle: {key}')
        samples[key] = dict(row, angle=angle)
        groups[angle][row['frame_group_key']].append(key)
    payload = dict(schema='timepix.frame_group.v1', split_seed=seed, ratios=list(ratios),
                   train=[], val=[], test=[], samples=samples)
    rng = random.Random(seed)
    for angle in sorted(groups, key=float):
        keys = sorted(groups[angle])
        if len(keys) < 3:
            raise ValueError(f'Need at least three source frames per angle: {angle}')
        rng.shuffle(keys)
        n = len(keys)
        nt = min(max(1, int(n * ratios[0])), n - 2)
        nv = min(max(1, int(n * ratios[1])), n - nt - 1)
        for split, subset in zip(('train', 'val', 'test'), (keys[:nt], keys[nt:nt+nv], keys[nt+nv:])):
            payload[split].extend(k for g in subset for k in groups[angle][g])
    for split in ('train', 'val', 'test'):
        payload[split].sort()
    if aliases:
        payload['alias_map'] = dict(aliases)
    payload['manifest_sha256'] = manifest_digest(payload)
    validate_group_manifest(payload, list(samples))
    return payload


def validate_group_manifest(payload: dict, record_keys: list[str]) -> None:
    if payload.get('schema') != 'timepix.frame_group.v1':
        raise ValueError('Required frame-group manifest schema missing')
    if payload.get('manifest_sha256') != manifest_digest(payload):
        raise ValueError('Frame-group manifest checksum mismatch')
    all_keys = [k for split in ('train', 'val', 'test') for k in payload[split]]
    if len(all_keys) != len(set(all_keys)) or len(record_keys) != len(set(record_keys)):
        raise ValueError('Duplicate sample keys or overlapping splits')
    if set(all_keys) != set(record_keys) or set(payload['samples']) != set(record_keys):
        raise ValueError('Frame-group manifest must cover the exact dataset')
    ownership = {}; raw_ownership = {}
    aliases = payload.get('alias_map', {})
    for split in ('train', 'val', 'test'):
        for key in payload[split]:
            row = payload['samples'][key]
            angle = str(row['angle'])
            canonical = row.get('canonical_frame_member', row['raw_frame_member'])
            raw = row['raw_frame_member']
            if canonical != aliases.get(raw, raw):
                raise ValueError(f'Unverified or inconsistent raw-to-canonical alias: {key}')
            raw_key = frame_key(angle, raw)
            if raw_ownership.setdefault(raw_key, split) != split:
                raise ValueError(f'Raw source frame overlaps splits: {raw_key}')
            expected = frame_key(angle, canonical)
            if row['frame_group_key'] != expected or not key.startswith(angle + '/'):
                raise ValueError(f'Invalid angle-qualified source-frame key: {key}')
            if ownership.setdefault(expected, split) != split:
                raise ValueError(f'Source frame overlaps splits: {expected}')
