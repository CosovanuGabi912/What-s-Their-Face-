"""scripts/download_classifier.py (issue #6): a bad download never replaces the checkpoint."""

from __future__ import annotations

from scripts import download_classifier as dc


def test_wrong_file_is_deleted_and_the_old_checkpoint_kept(tmp_path, monkeypatch):
    dest = tmp_path / dc.FILE_NAME
    dest.write_bytes(b"old checkpoint")
    monkeypatch.setattr(dc, "fetch", lambda url, path: path.write_bytes(b"not the model"))
    assert dc.download(dest) is False
    assert dest.read_bytes() == b"old checkpoint"
    assert not dest.with_suffix(".part").exists()


def test_matching_file_is_moved_into_place(tmp_path, monkeypatch):
    dest = tmp_path / dc.FILE_NAME
    monkeypatch.setattr(dc, "fetch", lambda url, path: path.write_bytes(b"model"))
    monkeypatch.setattr(dc, "SHA256", dc.hashlib.sha256(b"model").hexdigest())
    assert dc.download(dest) is True
    assert dest.read_bytes() == b"model"
    assert dc.download(dest) is True  # second run: already downloaded, no fetch needed


def test_failed_connection_leaves_nothing_behind(tmp_path, monkeypatch):
    def offline(url, path):
        path.write_bytes(b"half")
        raise OSError("no network")
    dest = tmp_path / dc.FILE_NAME
    monkeypatch.setattr(dc, "fetch", offline)
    assert dc.download(dest) is False
    assert not dest.exists() and not dest.with_suffix(".part").exists()
