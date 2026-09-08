import json
import zipfile

import pytest

from scripts.package_carbon_dataset import build_package, verify_extracted


def test_package_exact_members_and_verify(tmp_path):
    root = tmp_path / 'source'
    member = root / '80/ToT/same_001.txt'
    member.parent.mkdir(parents=True)
    member.write_text('0 1\n2 0\n')
    (member.parent / 'unselected.txt').write_text('not selected')
    manifest = tmp_path / 'split.json'
    manifest.write_text(json.dumps({'samples': {'80/same_001.txt': {
        'event_relative_path': '80/ToT/same_001.txt'}}}))
    output = tmp_path / 'events.zip'
    result = build_package(root, [manifest], output)
    assert result['event_count'] == 1
    with zipfile.ZipFile(output) as z:
        assert set(z.namelist()) == {'Proton_C/80/ToT/same_001.txt', 'carbon_transfer_manifest.json'}
        z.extractall(tmp_path / 'target')
    assert verify_extracted(tmp_path / 'target')['verified_events'] == 1
    extracted = tmp_path / 'target/Proton_C/80/ToT/same_001.txt'
    extracted.write_text('changed')
    with pytest.raises(ValueError, match='mismatch'):
        verify_extracted(tmp_path / 'target')
    with pytest.raises(FileExistsError):
        build_package(root, [manifest], output)


@pytest.mark.parametrize('member', ['../outside.txt', '/outside.txt', '80/../outside.txt'])
def test_reject_unsafe_member(tmp_path, member):
    manifest = tmp_path / 'split.json'
    manifest.write_text(json.dumps({'samples': {'key': {'event_relative_path': member}}}))
    with pytest.raises(ValueError):
        build_package(tmp_path, [manifest], tmp_path / 'out.zip')
