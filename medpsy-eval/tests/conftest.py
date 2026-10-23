import pytest
from golden_harness import FakeRemote


@pytest.fixture
def fake_remote(tmp_path, monkeypatch) -> FakeRemote:
    """Remote reads answered with synthetic rows (see ``golden_harness``); no network."""
    remote = FakeRemote(tmp_path / "remote")
    remote.workdir.mkdir()
    remote.install(monkeypatch)
    return remote
