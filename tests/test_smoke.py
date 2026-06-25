import orbscreen


def test_version_present():
    assert isinstance(orbscreen.__version__, str)
