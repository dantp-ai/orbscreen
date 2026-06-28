import pytest

from orbscreen.screen.cost import (
    cascade_cost_per_million,
    dollars_per_million,
    end_to_end_throughput,
    speedup,
)


def test_dollars_per_million_arithmetic():
    # 1000 structs/s -> 1e6 take 1000 s = 0.27778 h; at $3.60/h -> $1.00
    assert dollars_per_million(1000.0, usd_per_hour=3.60) == pytest.approx(1.0, rel=1e-6)


def test_dollars_per_million_rejects_nonpositive():
    with pytest.raises(ValueError):
        dollars_per_million(0.0)


def test_cascade_cost_is_surrogate_at_zero_budget():
    assert cascade_cost_per_million(0.0, surrogate_cpm=0.5, orb_cpm=100.0) == 0.5
    assert cascade_cost_per_million(0.1, surrogate_cpm=0.5, orb_cpm=100.0) == pytest.approx(10.5)


def test_speedup():
    assert speedup(orb_cpm=100.0, cascade_cpm=10.0) == 10.0


def test_end_to_end_throughput_is_slower_than_inference():
    result = end_to_end_throughput(1e6, graph_build_sec_per_struct=0.027)
    assert result == pytest.approx(1.0 / 0.027, rel=1e-3)
