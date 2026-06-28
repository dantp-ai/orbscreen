"""Throughput -> cost model for the Phase 3 screen and cascade.

All costs are USD per million structures, derived from a measured throughput and an
A10G hourly rate. The cascade cost adds the Orb-v3 cost only for the routed fraction.
"""

A10G_USD_PER_HOUR = 1.10  # Modal A10G on-demand; verify against current Modal pricing at run time.
GRAPH_BUILD_SEC_PER_STRUCT = 0.027  # Phase 2 measured: ~1.5 h to build 200k PBC graphs on CPU.


def dollars_per_million(throughput_per_sec, usd_per_hour=A10G_USD_PER_HOUR):
    """USD to process 1e6 structures at a given throughput (structures/second)."""
    if throughput_per_sec <= 0:
        raise ValueError("throughput_per_sec must be positive")
    seconds = 1e6 / throughput_per_sec
    return (seconds / 3600.0) * usd_per_hour


def cascade_cost_per_million(budget, surrogate_cpm, orb_cpm):
    """Cost/million when a fraction `budget` of candidates is also sent to Orb-v3."""
    return surrogate_cpm + budget * orb_cpm


def speedup(orb_cpm, cascade_cpm):
    """How many times cheaper the cascade is vs running Orb-v3 on everything."""
    if cascade_cpm <= 0:
        raise ValueError("cascade_cpm must be positive")
    return orb_cpm / cascade_cpm


def end_to_end_throughput(inference_tp, graph_build_sec_per_struct=GRAPH_BUILD_SEC_PER_STRUCT):
    """Throughput including CPU graph construction (the cold path), not just GPU inference."""
    per_struct = 1.0 / inference_tp + graph_build_sec_per_struct
    return 1.0 / per_struct
