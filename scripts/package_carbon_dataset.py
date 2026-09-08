#!/usr/bin/env python
"""Transfer exact frozen carbon event files without changing their bytes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '\\' in value or ':' in value or not p.parts:
        raise ValueError(f'Unsafe member: {value}')
    return p.as_posix()


def build_package(data_root, manifests, destination):
    root = Path(data_root).resolve()
    destination = Path(destination)
    partial = destination.with_suffix(destination.suffix + '.partial')
    if destination.exists() or partial.exists():
        raise FileExistsError('Existing package or partial retained; use a new destination')
    members = set()
    sources = []
    for path in manifests:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        sources.append({'name': Path(path).name, 'sha256': digest(path)})
        for sample in payload['samples'].values():
            relative = safe_relative(sample['event_relative_path'])
            if relative in members:
                raise ValueError(f'Duplicate event across manifests: {relative}')
            source = (root / relative).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError(f'Missing or outside-root event: {relative}')
            members.add(relative)
    if not members:
        raise ValueError('No events')
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    counts = Counter()
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for i, relative in enumerate(sorted(members), 1):
            source = root / relative
            before = source.stat()
            content = source.read_bytes()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'Source changed while reading: {relative}')
            member = 'Proton_C/' + relative
            archive.writestr(member, content)
            entries.append({'member': member, 'bytes': len(content),
                            'sha256': hashlib.sha256(content).hexdigest()})
            counts[relative.split('/')[0]] += 1
            if i % 10000 == 0:
                print(f'Packed {i}/{len(members)} events', flush=True)
        metadata = {'schema': 'carbon-byte-transfer-v1', 'events': entries,
                    'source_manifests': sources, 'angle_counts': dict(counts)}
        archive.writestr('carbon_transfer_manifest.json', json.dumps(metadata))
    with zipfile.ZipFile(partial) as archive:
        for entry in entries:
            if hashlib.sha256(archive.read(entry['member'])).hexdigest() != entry['sha256']:
                raise ValueError(f'Archive content mismatch: {entry["member"]}')
    partial.rename(destination)
    result = {'package': str(destination), 'event_count': len(entries),
              'uncompressed_event_bytes': sum(e['bytes'] for e in entries),
              'angle_counts': dict(counts), 'bytes': destination.stat().st_size,
              'sha256': digest(destination), 'source_manifests': sources}
    with destination.with_suffix('.verification.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return result


def verify_extracted(directory):
    root = Path(directory).resolve()
    metadata = json.loads((root / 'carbon_transfer_manifest.json').read_text(encoding='utf-8'))
    seen = set()
    for entry in metadata['events']:
        relative = safe_relative(entry['member'])
        path = (root / relative).resolve()
        if relative in seen or not path.is_relative_to(root):
            raise ValueError(f'Duplicate or outside-root member: {relative}')
        seen.add(relative)
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'Extracted content mismatch: {relative}')
    actual = {p.relative_to(root).as_posix() for p in (root / 'Proton_C').rglob('*') if p.is_file()}
    if actual != seen:
        raise ValueError('Extracted event inventory mismatch')
    return {'verified_events': len(seen), 'angle_counts': metadata['angle_counts'],
            'all_event_sha256_match': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--manifest', type=Path, action='append')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-extracted', type=Path)
    args = parser.parse_args()
    if args.verify_extracted:
        result = verify_extracted(args.verify_extracted)
    else:
        if not args.data_root or not args.manifest or not args.output:
            parser.error('Provide data-root, manifest(s), and output')
        result = build_package(args.data_root, args.manifest, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
