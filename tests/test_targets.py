from orbscreen.data.targets import stability_label


def _rec(**over):
    base = dict(
        mofchecker_valid=True,
        smact_valid=True,
        no_atom_too_close=True,
        geo_converged=True,
        reconstruction_failed=False,
    )
    base.update(over)
    return base


def test_stable_when_all_pass():
    assert stability_label(_rec()) == 1


def test_unstable_on_each_failing_flag():
    assert stability_label(_rec(mofchecker_valid=False)) == 0
    assert stability_label(_rec(smact_valid=False)) == 0
    assert stability_label(_rec(no_atom_too_close=False)) == 0
    assert stability_label(_rec(geo_converged=False)) == 0
    assert stability_label(_rec(reconstruction_failed=True)) == 0
