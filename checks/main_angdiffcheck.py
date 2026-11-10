# =============================================================================
# File:        main_angdiffcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-09
#
# Purpose:     Check the neighbor angle difference input (experiment 2)
#              before any training run uses it. Two parts.
#              Part 1 needs only numpy and runs anywhere: the calculation
#              against hand-worked values. A two-meter pair (10 and 40
#              degrees gives -30 and +30), the wrap at +179 and -179
#              degrees (a 2-degree difference, not 358), two neighbors
#              pointing opposite ways (no direction, feature invalid, all
#              zeros), no usable neighbor, a late own capture, a late
#              neighbor, the phase map (a meter whose L1 is phase c gets
#              its phase-a feature on L3; a line-to-line channel is never
#              compared), current channels always zero, and a period
#              boundary (the slot before a wiring change uses the earlier
#              edge set, the slot at the change the later one, so no future
#              topology enters a history), with the period bounds given as
#              integer seconds, as the real time-indexed file stores them,
#              and a bound in the wrong unit refused. Then, when the repo's
#              graph_meters_periods.json is present (or --graph names a
#              file), the real file: every period bound reads as a
#              plausible epoch second, the periods are contiguous, the
#              first slot of each period selects that period's edge set,
#              every edge set has edges, and a pair the file names as an
#              edge (its first edge) are neighbors in the adjacency. Then
#              the data side: a synthetic
#              month and late file through socal_modeldata with the inputs
#              on, eight features per channel, the first five identical to
#              the inputs off, refusal when both added-input kinds are on,
#              when the late file is missing, when it has the wrong number
#              of meter columns, when its meters are in another order, and
#              when the late manifest's threshold is missing or not finite.
#              Part 2 needs torch and torch_geometric: a masked forecast
#              must not change when the three added features change, and a
#              checkpoint is refused when "angdiff" or "angdiff_settings"
#              differ. Skipped with a message where torch is absent.
#
# Usage (from the repo root):
#   python checks/main_angdiffcheck.py
#
# Functions:
#   parse_args()            command-line options with defaults
#   close(a, b, tol)        absolute closeness
#   check_calculation()     part 1, the hand-worked cases
#   period_grid_slots(p)    first and last ten-second grid timestamps inside a period, or None
#   check_timeline(tl, label)
#                           contiguity, interval selection and grid selection of a timeline
#   check_period_fixtures() part 1, awkward synthetic periods and structural eligibility
#   check_real_graph(path)  part 1, the repo's time-indexed graph file
#   check_data_side(tmp)    part 1, through socal_modeldata
#   check_align_verdicts(tmp)
#                           part 1, the late-file alignment checker's verdicts on synthetic tables
#   check_torch(args)       part 2
#   main()                  run the parts and report
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON
import json
# Standard library: paths
import os
# Standard library: import path for the repo root
import sys
# Standard library: temporary folder
import tempfile
# Third party: numerics
import numpy as np
# Third party: timestamps for the block fixtures
import pandas as pd

# Repo root on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The helpers under test
import socal_angdiff as sa
# The data side
import socal_modeldata as sd


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Check the neighbor angle difference input: calculation, data side, and with torch the masked and checkpoint safeguards")
    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the torch part")
    # Real graph file
    parser.add_argument("--graph", default="graph_meters_periods.json", help="Time-indexed graph file to check against; skipped with a message when absent")
    # Parse
    return parser.parse_args()


def close(a, b, tol=1e-5):
    """True when two numbers or arrays agree to tol."""
    # Compare
    return bool(np.all(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) <= tol))


def check_calculation():
    """Part 1: the calculation against hand-worked values."""
    # Channels as in the feature files
    channels = ["L1", "L2", "L3", "S1", "S2", "S3"]
    # Four meters: m0 and m1 standard, m2 with L1 = phase c and L3 = phase a, m3 line-to-line (never compared)
    meters = ["m0", "m1", "m2", "m3"]
    # Phase maps
    records = {"m0": {"node": 0, "phase": {"L1": "ag", "L2": "bg", "L3": "cg"}}, "m1": {"node": 1, "phase": {"L1": "ag", "L2": "bg", "L3": "cg"}},
               "m2": {"node": 2, "phase": {"L3": "ag", "L2": "bg", "L1": "cg"}}, "m3": {"node": 3, "phase": {"L1": "ab"}}}
    # Phase index
    idx = sa.phase_channel_index(records, meters, channels)
    # m2's phase a is channel L3 (index 2), m3 has nothing
    assert idx[2].tolist() == [2, 1, 0] and idx[3].tolist() == [-1, -1, -1] and idx[0].tolist() == [0, 1, 2], idx
    # Say
    print("phase map: m2 phase a on L3, m3 (line-to-line) unmapped: ok")
    # Two edge sets: graph 0 joins m0-m1 and m1-m2 and m1-m3; graph 1 joins m0-m1 and m0-m2 (m2 moves)
    body = {"nodes": [{"name": m} for m in meters], "meters": records,
            "graphs": [{"index": 0, "edges": [{"source": 0, "target": 1}, {"source": 1, "target": 2}, {"source": 1, "target": 3}]}, {"index": 1, "edges": [{"source": 0, "target": 1}, {"source": 0, "target": 2}]}],
            "periods": [{"index": 0, "start_utc": 1735689600, "end_utc_exclusive": 1735689660, "graph": 0}, {"index": 1, "start_utc": 1735689660, "end_utc_exclusive": 1735776000, "graph": 1}]}
    # The bounds above are integer seconds, as the real file stores them; a text timestamp reads the same
    assert sa.period_seconds(1735689600) == sa.period_seconds("2025-01-01T00:00:00+00:00") == 1735689600
    # Bounds in the wrong unit are refused: the collapse to 1, a millisecond value, a nanosecond value
    for bad in (1, 1735689600000, 1735689600000000000):
        # Attempt
        try:
            # Read
            sa.period_seconds(bad)
            # Failure
            raise AssertionError(f"a period bound of {bad} was accepted")
        # Expected
        except ValueError as e:
            # Say
            print(f"refused: {e}")
    # Timeline and adjacency
    timeline = sa.timeline_from_graph(body)
    # Matrices
    A = sa.adjacency_matrices(timeline, records, meters)
    # Symmetric with the right neighbors
    assert A[0][0, 1] == 1 and A[0][1, 0] == 1 and A[0][1, 2] == 1 and A[0][0, 2] == 0 and A[1][0, 2] == 1 and A[1][1, 2] == 0
    # Eight slots: six in period 0 (0 to 50 s), two in period 1 (60 and 70 s)
    t0 = int(1735689600)
    # Times
    t = t0 + 10 * np.arange(8)
    # Edge set per slot
    gs = sa.graph_per_slot(t, timeline)
    # Boundary: slot 5 (50 s) is graph 0, slot 6 (60 s) is graph 1
    assert gs.tolist() == [0, 0, 0, 0, 0, 0, 1, 1], gs
    # Say
    print("period boundary: slots before 00:01:00 use edge set 0, from 00:01:00 edge set 1: ok")
    # Angles (T, M, C), start with every meter at 0 on every channel
    T, M, C = len(t), len(meters), len(channels)
    # Angles
    ang = np.zeros((T, M, C), dtype=np.float32)
    # Validity, all valid
    ok = np.ones((T, M, C), dtype=bool)
    # Late, none
    late = np.zeros((T, M), dtype=bool)
    # Slot 0: m0 at 10, m1 at 40 on phase a (L1 for both); m2 phase a (L3) at 40 too so m1's mean is from m0 and m2
    ang[0, 0, 0], ang[0, 1, 0], ang[0, 2, 2] = 10.0, 40.0, 40.0
    # Slot 1: wrap: m0 at 179, m1 at -179; m2 phase a at -179
    ang[1, 0, 0], ang[1, 1, 0], ang[1, 2, 2] = 179.0, -179.0, -179.0
    # Slot 2: opposite neighbors for m1: m0 at 0, m2 phase a at 180
    ang[2, 0, 0], ang[2, 2, 2] = 0.0, 180.0
    # Slot 3: m0 late, so m1's mean comes from m2 only (at 90) and m0's own feature is invalid
    late[3, 0] = True
    # m2 phase a at 90
    ang[3, 2, 2] = 90.0
    # Slot 4: m0's angle invalid on L1 (angle_ok False), m1 at 30: m1 still has m2 at 0 as a neighbor
    ok[4, 0, 0] = False
    # m1 at 30
    ang[4, 1, 0] = 30.0
    # Slot 6 (period 1): m2 is now m0's neighbor, not m1's; m0 at 0, m2 phase a at 60, m1 at 0
    ang[6, 2, 2] = 60.0
    # Compute
    nb_cos, nb_sin, nb_valid, reasons = sa.neighbor_differences(ang, ok, late, idx, A, gs, 0.5)
    # Slot 0: m0's only neighbor in graph 0 is m1 at 40: d = -30
    assert nb_valid[0, 0, 0] and close(nb_cos[0, 0, 0], np.cos(np.deg2rad(-30))) and close(nb_sin[0, 0, 0], np.sin(np.deg2rad(-30)))
    # m1's neighbors m0 (10), m2 (40) and m3 (unmapped, excluded): mean 25, d = 15
    assert nb_valid[0, 1, 0] and close(nb_cos[0, 1, 0], np.cos(np.deg2rad(15))) and close(nb_sin[0, 1, 0], np.sin(np.deg2rad(15)))
    # m2's phase-a feature sits on L3, neighbor m1's phase a (L1) at 40: d = 0; m2's L1 carries phase c against m1's L3 (both 0): d = 0 too
    assert nb_valid[0, 2, 2] and close(nb_cos[0, 2, 2], 1.0) and close(nb_sin[0, 2, 2], 0.0) and nb_valid[0, 2, 0] and close(nb_cos[0, 2, 0], 1.0)
    # m3 never
    assert not nb_valid[:, 3, :].any() and not nb_cos[:, 3, :].any()
    # Say
    print("two-meter pair 10 and 40 degrees: -30 and (with a third neighbor) +15; phase-c meter's feature on L3: ok")
    # Slot 1: m0 at 179 against m1 at -179: d = -2 degrees, not 358
    assert nb_valid[1, 0, 0] and close(nb_cos[1, 0, 0], np.cos(np.deg2rad(-2))) and close(nb_sin[1, 0, 0], np.sin(np.deg2rad(-2)))
    # Say
    print("wrap at +179 / -179 degrees: difference -2 degrees: ok")
    # Slot 2: m1's neighbors at 0 and 180: resultant 0, invalid and all zero
    assert not nb_valid[2, 1, 0] and nb_cos[2, 1, 0] == 0.0 and nb_sin[2, 1, 0] == 0.0
    # Say
    print("opposite neighbors: no direction, feature zero with flag 0 (not cos 0 = 1): ok")
    # Slot 3: m0 late: own invalid; m1's mean from m2 at 90 only: d = -90
    assert not nb_valid[3, 0, 0] and nb_cos[3, 0, 0] == 0.0
    # m1
    assert nb_valid[3, 1, 0] and close(nb_cos[3, 1, 0], 0.0) and close(nb_sin[3, 1, 0], -1.0)
    # Say
    print("late own capture: invalid; late neighbor dropped from the mean: ok")
    # Slot 4: m0's angle invalid: own invalid; m1 (30) against m2 (0): d = 30
    assert not nb_valid[4, 0, 0] and nb_valid[4, 1, 0] and close(nb_cos[4, 1, 0], np.cos(np.deg2rad(30)))
    # Slot 5: everyone at 0, all valid with d = 0 for mapped meters with neighbors
    assert nb_valid[5, 0, 0] and nb_valid[5, 1, 0] and nb_valid[5, 2, 2]
    # Slot 6, period 1: m0's neighbors are m1 (0) and m2 (60): mean 30, d = -30; m1's only neighbor is m0 (0): d = 0; m2's neighbor is m0: d = 60
    assert close(nb_cos[6, 0, 0], np.cos(np.deg2rad(-30))) and close(nb_sin[6, 0, 0], np.sin(np.deg2rad(-30)))
    # m1
    assert close(nb_cos[6, 1, 0], 1.0) and close(nb_sin[6, 1, 0], 0.0)
    # m2
    assert close(nb_cos[6, 2, 2], np.cos(np.deg2rad(60))) and close(nb_sin[6, 2, 2], np.sin(np.deg2rad(60)))
    # Slot 5 must not have used period 1's wiring: m2's neighbor there is m1 (0), and m2 is at 0, d = 0; under period 1 m2 would see m0 which is also 0, so test the other way: m1 at slot 5 sees m0 and m2 and m3-excluded under graph 0
    # Put a distinguishing value: recompute with m2 at 60 in slot 5 as well; under graph 0 m0 does not see m2, so m0's d stays 0
    ang2 = ang.copy()
    # m2 at 60 in slot 5
    ang2[5, 2, 2] = 60.0
    # Recompute
    c2, s2, v2, _ = sa.neighbor_differences(ang2, ok, late, idx, A, gs, 0.5)
    # m0 at slot 5 only sees m1 (0): d = 0; at slot 6 sees m1 and m2: d = -30
    assert close(c2[5, 0, 0], 1.0) and close(s2[5, 0, 0], 0.0) and close(c2[6, 0, 0], np.cos(np.deg2rad(-30)))
    # Say
    print("period boundary: the slot before the change does not see the later neighbor; the slot at the change does: ok")
    # Current channels always zero
    assert not nb_valid[:, :, 3:].any() and not nb_cos[:, :, 3:].any() and not nb_sin[:, :, 3:].any()
    # Say
    print("current channels: always zero with flag 0: ok")
    # Reason counts add up to T x M x 3
    assert sum(reasons.values()) == T * M * 3, reasons
    # Unmapped: m3 three phases every slot
    assert reasons["unmapped"] == T * 3
    # Say
    print(f"reason counts sum to slots x meters x phases: {reasons}: ok")


def period_grid_slots(p, grid_seconds=10):
    """First and last ten-second grid timestamps inside [start_s, end_s) of a period, or None when the period holds no grid timestamp."""
    # First grid time at or after the start
    first = -(-p["start_s"] // grid_seconds) * grid_seconds
    # Last grid time before the exclusive end
    last = (p["end_s"] - 1) // grid_seconds * grid_seconds
    # None inside
    if first > last:
        # Say
        return None
    # Return
    return first, last


def check_timeline(tl, label):
    """
    The period and selection checks shared by the real file and the
    synthetic fixtures: periods contiguous; interval selection at each
    period's start and at the second before its exclusive end; grid
    selection at the first and last ten-second grid timestamps inside the
    period, reported when a period holds none.
    """
    # Contiguous
    for a, b in zip(tl["periods"][:-1], tl["periods"][1:]):
        # Check
        assert a["end_s"] == b["start_s"], (a, b)
    # Interval selection: start and the second before the exclusive end
    t_int = np.array([x for p in tl["periods"] for x in (p["start_s"], p["end_s"] - 1)], dtype=np.int64)
    # Expected
    want = np.array([p["graph"] for p in tl["periods"] for _ in (0, 1)], dtype=np.int64)
    # Check
    got = sa.graph_per_slot(t_int, tl)
    # Compare
    assert np.array_equal(got, want), ("interval selection", got, want)
    # Grid selection
    empty = []
    # Each period
    for k, p in enumerate(tl["periods"]):
        # Grid slots
        slots = period_grid_slots(p)
        # None
        if slots is None:
            # Record
            empty.append(k)
            # Next
            continue
        # First and last
        got = sa.graph_per_slot(np.array(slots, dtype=np.int64), tl)
        # Both select this period's edge set
        assert got.tolist() == [p["graph"], p["graph"]], ("grid selection", k, slots, got)
    # Say
    print(f"{label}: {len(tl['periods'])} periods from {tl['periods'][0]['start_s']} to {tl['periods'][-1]['end_s']} s, contiguous; interval selection at start and end-1 ok; grid selection at the first and last ten-second slots ok" + (f"; periods with no grid timestamp (reported, not an error): {empty}" if empty else ""))
    # Return the empty ones for the caller
    return empty


def check_real_graph(path):
    """Part 1 continued: the repo's time-indexed graph file, when present."""
    # Absent
    if not os.path.exists(path):
        # Say
        print(f"real graph check skipped: {path} not found")
        # Done
        return
    # Read
    with open(path) as f:
        # Parse
        body = json.load(f)
    # Not time-indexed
    if "periods" not in body:
        # Say
        print(f"real graph check skipped: {path} has no periods")
        # Done
        return
    # Timeline
    tl = sa.timeline_from_graph(body)
    # Periods and selection
    check_timeline(tl, f"real graph {path}")
    # Meters in the file, in a fixed order
    meters = sorted(body["meters"], key=lambda m: int(m.split("_")[1]))
    # Channels as the feature files name them
    channels = ["L1", "L2", "L3", "S1", "S2", "S3"]
    # Adjacency
    A = sa.adjacency_matrices(tl, body["meters"], meters)
    # Every edge set has edges
    assert all(a.sum() > 0 for a in A), [int(a.sum()) for a in A]
    # The first edge of each edge set joins two meters that are neighbors in the matrix
    by_node = {int(r["node"]): m for m, r in body["meters"].items()}
    # Each edge set
    for k, g in enumerate(body["graphs"]):
        # First edge
        e = g["edges"][0]
        # Meters at its ends
        a, b = meters.index(by_node[int(e["source"])]), meters.index(by_node[int(e["target"])])
        # Neighbors both ways
        assert A[k][a, b] == 1 and A[k][b, a] == 1, (k, e)
    # Say
    print(f"real graph: {len(A)} edge sets with {[int(a.sum() // 2) for a in A]} edges; first edge of each set joins neighbors: ok")
    # Structural eligibility, printed for comparison with the reviewers' table
    elig = sa.structural_eligibility(tl, body["meters"], meters, channels)
    # Print
    sa.print_eligibility(elig, meters)


def check_period_fixtures():
    """Part 1 continued: synthetic timelines with awkward periods: shorter than ten seconds, off-grid boundaries, none holding a grid timestamp."""
    # Two meters joined
    meters = {"m0": {"node": 0, "phase": {"L1": "ag"}}, "m1": {"node": 1, "phase": {"L1": "ag"}}}
    # Edge sets
    graphs = [{"index": 0, "edges": [{"source": 0, "target": 1}]}, {"index": 1, "edges": [{"source": 0, "target": 1}]}, {"index": 2, "edges": [{"source": 0, "target": 1}]}]
    # Periods: a normal one, an eight-second one starting off-grid, one of three seconds holding no grid time, then a long one
    t0 = 1735689600
    # Periods
    periods = [{"index": 0, "start_utc": t0, "end_utc_exclusive": t0 + 95, "graph": 0},
               {"index": 1, "start_utc": t0 + 95, "end_utc_exclusive": t0 + 103, "graph": 1},
               {"index": 2, "start_utc": t0 + 103, "end_utc_exclusive": t0 + 106, "graph": 2},
               {"index": 3, "start_utc": t0 + 106, "end_utc_exclusive": t0 + 1000, "graph": 0}]
    # Body
    body = {"nodes": [{"name": "n0"}, {"name": "n1"}], "meters": meters, "graphs": graphs, "periods": periods}
    # Timeline
    tl = sa.timeline_from_graph(body)
    # Checks
    empty = check_timeline(tl, "period fixtures")
    # The eight-second period holds grid time t0+100 only; the three-second one holds none
    assert period_grid_slots(tl["periods"][1]) == (t0 + 100, t0 + 100) and empty == [2], (period_grid_slots(tl["periods"][1]), empty)
    # Grid time t0+100 selects edge set 1, t0+110 selects edge set 0 again
    assert sa.graph_per_slot(np.array([t0 + 90, t0 + 100, t0 + 110]), tl).tolist() == [0, 1, 0]
    # Say
    print("period fixtures: eight-second off-grid period selects its edge set at its one grid slot; three-second period reported as holding none: ok")
    # Structural eligibility on a small case: m2 line-to-line, m3 whose only neighbor is m2, m4 isolated
    meters2 = {"m0": {"node": 0, "phase": {"L1": "ag"}}, "m1": {"node": 1, "phase": {"L1": "ag"}}, "m2": {"node": 2, "phase": {"L1": "ab"}}, "m3": {"node": 3, "phase": {"L1": "ag"}}, "m4": {"node": 4, "phase": {"L1": "ag"}}}
    # One edge set
    body2 = {"nodes": [{"name": f"n{i}"} for i in range(5)], "meters": meters2, "edges": [{"source": 0, "target": 1}, {"source": 2, "target": 3}, {"source": 1, "target": 2}]}
    # Eligibility
    elig = sa.structural_eligibility(sa.timeline_from_graph(body2), meters2, ["m0", "m1", "m2", "m3", "m4"], ["L1"])
    # Expected
    assert elig["eligible"][0].tolist() == [True, True, False, False, False], elig
    # Reasons
    assert elig["reason"][0][2] == "not a phase-to-neutral voltage meter" and elig["reason"][0][3] == "every neighbor excluded" and elig["reason"][0][4] == "no edges", elig["reason"]
    # Say
    print("structural eligibility: line-to-line meter, meter whose only neighbor is excluded, isolated meter: each named: ok")


def check_data_side(tmp):
    """Part 1 continued: a synthetic month and late file through socal_modeldata with the inputs on."""
    # Folder
    fdir = os.path.join(tmp, "features")
    # Make
    os.makedirs(fdir)
    # Two meters, six channels, one hour
    meters, channels = ["egauge_1", "egauge_2"], ["L1", "L2", "L3", "S1", "S2", "S3"]
    # Slot times inside January 2025 in the files' convention
    t = int(1735718400) + 10 * np.arange(360)
    # Sizes
    T, M, C = len(t), 2, 6
    # Random generator
    rng = np.random.default_rng(0)
    # Magnitudes
    rms = (277 + rng.standard_normal((T, M, C))).astype(np.float16)
    # Angles: meter 2 leads meter 1 by 20 degrees on every voltage channel
    ang = rng.uniform(-180, 180, (T, 1, C)).astype(np.float32).repeat(M, axis=1)
    # Lead
    ang[:, 1, :3] += 20.0
    # Wrap
    ang = ((ang + 180.0) % 360.0 - 180.0).astype(np.float32)
    # Masks
    mask = np.ones((T, M, C), dtype=bool)
    # Angle ok
    angle_ok = np.ones((T, M, C), dtype=bool)
    # Save the month
    np.savez_compressed(os.path.join(fdir, "2025-01.npz"), t=t, rms=rms, ang=ang, freq=np.full((T, M), 60.0, np.float32), mask=mask, angle_ok=angle_ok)
    # Manifest
    json.dump({"version": "1", "meters": meters, "channels": channels, "months": ["2025-01"], "arrays": {}}, open(os.path.join(fdir, "manifest.json"), "w"))
    # Late: meter 1 late in the first 36 slots
    late = np.zeros((T, M), dtype=bool)
    # Late
    late[:36, 0] = True
    # Write
    sa.write_late(fdir, "2025-01", t, late, meters)
    # Manifest
    sa.write_late_manifest(fdir, ["2025-01"], 0.5, "processed", meters)
    # Graph: fixed, the two meters joined
    body = {"nodes": [{"name": m} for m in meters], "edges": [{"source": 0, "target": 1, "admittance_abs": 1.0}], "meters": {"egauge_1": {"node": 0, "phase": {"L1": "ag", "L2": "bg", "L3": "cg"}}, "egauge_2": {"node": 1, "phase": {"L1": "ag", "L2": "bg", "L3": "cg"}}}}
    # Manifest
    manifest = sd.load_manifest(fdir)
    # Off: five features
    arrays_off = sd.load_month(fdir, "2025-01", manifest=manifest)
    # Scale of ones
    scale = np.ones((M, C), dtype=np.float32)
    # Features off
    X_off, _, _ = sd.build_features(arrays_off, scale)
    # Five
    assert X_off.shape[-1] == 5 and sd.n_features() == 5
    # On
    settings = sd.configure_angdiff(fdir, body, manifest, 0.5, 0.5)
    # Eight
    assert sd.n_features() == 8 and sd.extra_features() == list(sa.ANGDIFF_FEATURES)
    # Load
    arrays_on = sd.load_month(fdir, "2025-01", manifest=manifest)
    # Build
    X_on, _, _ = sd.build_features(arrays_on, scale)
    # First five identical
    assert X_on.shape[-1] == 8 and np.array_equal(X_on[..., :5], X_off)
    # Meter 1 against meter 2: d = -20 on voltage channels, valid where meter 1 is not late
    assert close(X_on[36:, 0, :3, 5], np.cos(np.deg2rad(-20)), 1e-4) and close(X_on[36:, 0, :3, 6], np.sin(np.deg2rad(-20)), 1e-4) and np.all(X_on[36:, 0, :3, 7] == 1)
    # Meter 2 against meter 1: +20
    assert close(X_on[36:, 1, :3, 5], np.cos(np.deg2rad(20)), 1e-4) and close(X_on[36:, 1, :3, 6], np.sin(np.deg2rad(20)), 1e-4)
    # Late slots: both invalid (meter 1 late, meter 2 has no usable neighbor)
    assert np.all(X_on[:36, :, :3, 5:] == 0)
    # Current channels zero
    assert np.all(X_on[:, :, 3:, 5:] == 0)
    # Say
    print(f"data side: 8 features per channel, first five identical to the inputs off, +20/-20 degrees recovered, late slots zero, settings {settings}: ok")
    # Training availability guard: counted inside the history inputs of the windows training would retain
    # (history 60, horizon 1, gap rule 12, at least one meter), on the first 61 slots of the month
    def block(a, b):
        # A block piece in the splits layout
        return {"start": "x", "end": "y", "start_utc": pd.Timestamp(a, unit="s", tz="UTC").isoformat(), "end_utc_exclusive": pd.Timestamp(b, unit="s", tz="UTC").isoformat()}
    # The month as written has valid features from slot 36 on; a 61-slot piece has one window (history 0..59, target 60)
    # Positive history case: slots 36..59 of its history hold valid features
    v, n, w = sd.angdiff_training_valid(fdir, [block(int(t[0]), int(t[61]))], manifest, 60, 1, 12, 1)
    # One window, 60 history slots, 24 valid slots x 2 meters x 3 channels
    assert (v, n, w) == (24 * 2 * 3, 60, 1), (v, n, w)
    # Target-only case: make both meters late on slots 0..59 and not late on slot 60, so valid features exist only on the target row
    late_t = late.copy()
    # Late history
    late_t[:60, :] = True
    # Not late target
    late_t[60, :] = False
    # Write
    sa.write_late(fdir, "2025-01", t, late_t, meters)
    # Count
    v, n, w = sd.angdiff_training_valid(fdir, [block(int(t[0]), int(t[61]))], manifest, 60, 1, 12, 1)
    # The window is retained (targets are valid), its history holds nothing valid, the target row's 6 valid values do not count
    assert (v, n, w) == (0, 60, 1), (v, n, w)
    # Say
    print("training guard, target-only case: 61 slots, history 60, horizon 1, six valid features on the target row only: count 0 (guard fails as required)")
    # Discarded-window case: restore the late flags, then remove both meters' readings on slots 40..51 (a run of 12 absent slots inside the only window's history), so the window is discarded although slots 36..39 hold valid features
    sa.write_late(fdir, "2025-01", t, late, meters)
    # Month arrays as written
    with np.load(os.path.join(fdir, "2025-01.npz")) as d:
        # Copy
        month = {k: d[k] for k in d.files}
    # Absent run
    month["mask"][40:52, :, :] = False
    # Save under the month name after keeping the original
    os.replace(os.path.join(fdir, "2025-01.npz"), os.path.join(fdir, "2025-01.keep"))
    # Write
    np.savez_compressed(os.path.join(fdir, "2025-01.npz"), **month)
    # Count
    v, n, w = sd.angdiff_training_valid(fdir, [block(int(t[0]), int(t[61]))], manifest, 60, 1, 12, 1)
    # No window retained, nothing counted
    assert (v, n, w) == (0, 0, 0), (v, n, w)
    # A longer piece has retained windows after the gap, and they count
    v2, n2, w2 = sd.angdiff_training_valid(fdir, [block(int(t[0]), int(t[200]))], manifest, 60, 1, 12, 1)
    # Positive
    assert v2 > 0 and w2 > 0, (v2, n2, w2)
    # Restore the month
    os.replace(os.path.join(fdir, "2025-01.keep"), os.path.join(fdir, "2025-01.npz"))
    # Say
    print(f"training guard, discarded-window case: a 12-slot absent run inside the only window's history discards it, count 0 although slots 36..39 hold valid features; a longer piece with retained windows counts {v2} values in {w2} windows: ok")
    # The dataset and the guard share the rule: retained_starts gives the same starts the graph dataset keeps
    starts, usable = sd.retained_starts(arrays_on["mag_valid"][:61], 60, 1, 12, 1)
    # One window at start 0
    assert starts.tolist() == [0] and usable.shape == (1, 2)
    # Say
    print("training guard: selection rule shared with the graph training set (socal_modeldata.retained_starts): ok")
    # Both kinds on is refused
    try:
        # Attempt
        sd.configure_physics(fdir)
        # Failure
        raise AssertionError("physics inputs were turned on alongside the neighbor angle inputs")
    # Expected
    except ValueError as e:
        # Say
        print(f"refused: {e}")
    # A late file with one column for two meters is refused (write_late refuses to write it; a hand-made one is refused on load)
    try:
        # Attempt to write
        sa.write_late(fdir, "2025-01", t, late[:, :1], meters)
        # Failure
        raise AssertionError("a (T, 1) late array was written for two meters")
    # Expected
    except ValueError as e:
        # Say
        print(f"refused: {e}")
    # Hand-made file with one column but both meter names
    np.savez_compressed(os.path.join(fdir, "2025-01_late.npz"), t=t, late=late[:, :1], meters=np.array(meters))
    # Attempt to load
    try:
        # Load
        sd.load_month(fdir, "2025-01", manifest=manifest)
        # Failure
        raise AssertionError("a (T, 1) late array loaded for two meters")
    # Expected
    except ValueError as e:
        # Say
        print(f"refused: {e}")
    # Reordered meters are refused
    sa.write_late(fdir, "2025-01", t, late[:, ::-1], meters[::-1])
    # Attempt
    try:
        # Load
        sd.load_month(fdir, "2025-01", manifest=manifest)
        # Failure
        raise AssertionError("a late file with the meters in another order loaded")
    # Expected
    except ValueError as e:
        # Say
        print(f"refused: {e}")
    # The manifest threshold must be present and finite: missing, NaN and a string are refused; the right one passes
    for bad in ({}, {"threshold_seconds": float("nan"), "meters": meters}, {"threshold_seconds": "0.5", "meters": meters}, {"threshold_seconds": 0.5}):
        # Write it
        json.dump(bad, open(os.path.join(fdir, "late_manifest.json"), "w"))
        # Attempt
        try:
            # Read
            sa.read_late_manifest(fdir, 0.5)
            # Failure
            raise AssertionError(f"late manifest {bad} was accepted")
        # Expected
        except ValueError as e:
            # Say
            print(f"refused: {e}")
    # A good manifest with another threshold is refused too
    sa.write_late_manifest(fdir, ["2025-01"], 0.7, "processed", meters)
    # Attempt
    try:
        # Read
        sa.read_late_manifest(fdir, 0.5)
        # Failure
        raise AssertionError("a manifest with threshold 0.7 passed for a requested 0.5")
    # Expected
    except ValueError as e:
        # Say
        print(f"refused: {e}")
    # The right one passes
    sa.write_late_manifest(fdir, ["2025-01"], 0.5, "processed", meters)
    # Read
    body, _ = sa.read_late_manifest(fdir, 0.5)
    # Check
    assert body["threshold_seconds"] == 0.5 and body["meters"] == meters
    # Say
    print("late manifest: missing, NaN, string and mismatched thresholds refused; the recorded one accepted")
    # Missing late file is refused
    os.remove(os.path.join(fdir, "2025-01_late.npz"))
    # Attempt
    try:
        # Load
        sd.load_month(fdir, "2025-01", manifest=manifest)
        # Failure
        raise AssertionError("a month loaded without its late file")
    # Expected
    except FileNotFoundError as e:
        # Say
        print(f"refused: {e}")
    # Off again
    sd.configure_angdiff(None)
    # Five
    assert sd.n_features() == 5


def check_align_verdicts(tmp):
    """
    Part 1 continued: the late-file alignment checker's verdicts on
    synthetic processed tables: aligned time-zone-aware rows with
    matching flags (passed); an altered saved flag (failed); rows shifted
    so none matches a slot (failed, never passed); a time-zone-naive index
    (not verified); rows beyond the documented month bounds (failed); a
    feature month missing its first eight hours (not verified without
    independent bounds, passed with matching given bounds, not verified
    with non-matching ones); full coverage (passed). Then main()'s exit
    code on a failing case.
    """
    # The checker, imported from this folder
    import importlib.util
    # Path
    cpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_latealigncheck.py")
    # Load
    spec = importlib.util.spec_from_file_location("latealign", cpath)
    # Module
    la = importlib.util.module_from_spec(spec)
    # Execute
    spec.loader.exec_module(la)
    # Folders
    fdir, pdir = os.path.join(tmp, "features"), os.path.join(tmp, "processed", "2025-01")
    # Make
    os.makedirs(fdir)
    # Processed
    os.makedirs(pdir)
    # Documented month bounds for 2025-01
    m0, m1 = sd.month_bounds("2025-01")
    # A feature month covering the whole documented month at ten seconds (long, but only the first hour gets rows)
    t = np.arange(m0, m1, 10, dtype=np.int64)
    # Meters
    meters = ["egauge_1", "egauge_2"]
    # Late flags: meter 1 late on slots 100..199 (offset 0.7), meter 2 never
    late = np.zeros((len(t), 2), dtype=bool)
    # Late
    late[100:200, 0] = True
    # Minimal feature month (only t is read by the checker beyond the shape)
    T, M, C = len(t), 2, 6
    # Save
    np.savez_compressed(os.path.join(fdir, "2025-01.npz"), t=t, rms=np.ones((T, M, C), np.float16), ang=np.zeros((T, M, C), np.float32), freq=np.full((T, M), 60.0, np.float32), mask=np.ones((T, M, C), bool), angle_ok=np.ones((T, M, C), bool))
    # Manifest
    json.dump({"version": "1", "meters": meters, "channels": ["L1", "L2", "L3", "S1", "S2", "S3"], "months": ["2025-01"], "arrays": {}}, open(os.path.join(fdir, "manifest.json"), "w"))
    # Late file and manifest
    sa.write_late(fdir, "2025-01", t, late, meters)
    # Manifest
    sa.write_late_manifest(fdir, ["2025-01"], 0.5, pdir, meters)
    # Processed table for meter 1: the whole month, tz-aware UTC, offsets 0.05 except 0.7 on rows 100..199
    idx = pd.to_datetime(t, unit="s", utc=True)
    # Offsets
    toff = np.full(len(t), 0.05)
    # Late ones
    toff[100:200] = 0.7
    # Write
    pd.DataFrame({"egauge_1|toff": toff}, index=idx).to_parquet(os.path.join(pdir, "egauge_1.parquet"))
    # Helper to run one meter and return its verdict
    def run(meter, mi, path, late_arr, t_arr):
        # Call
        return la.inspect_meter(meter, mi, path, f"{meter}|toff", t_arr, late_arr, 0.5, m0, m1, 1)
    # Case 1: aligned, aware, matching flags
    assert run("egauge_1", 0, os.path.join(pdir, "egauge_1.parquet"), late, t) == "passed"
    # Case 2: altered saved flag
    bad = late.copy()
    # Flip one
    bad[150, 0] = False
    # Failed
    assert run("egauge_1", 0, os.path.join(pdir, "egauge_1.parquet"), bad, t) == "failed"
    # Case 3: rows shifted by five seconds: none matches a slot, all inside the range
    pd.DataFrame({"egauge_1|toff": toff}, index=idx + pd.Timedelta(seconds=5)).to_parquet(os.path.join(pdir, "shifted.parquet"))
    # Column name must match the meter
    v3 = la.inspect_meter("egauge_1", 0, os.path.join(pdir, "shifted.parquet"), "egauge_1|toff", t, late, 0.5, m0, m1, 1)
    # Never passed
    assert v3 == "failed", v3
    # Case 4: naive index, otherwise aligned
    pd.DataFrame({"egauge_1|toff": toff}, index=idx.tz_localize(None)).to_parquet(os.path.join(pdir, "naive.parquet"))
    # Not verified
    v4 = la.inspect_meter("egauge_1", 0, os.path.join(pdir, "naive.parquet"), "egauge_1|toff", t, late, 0.5, m0, m1, 1)
    # Check
    assert v4 == "not verified", v4
    # Case 5: rows beyond the documented month (one day after its end), otherwise aligned
    extra = pd.to_datetime(np.arange(m1 + 86400, m1 + 86400 + 100, 10), unit="s", utc=True)
    # Table
    pd.DataFrame({"egauge_1|toff": np.concatenate([toff, np.full(10, 0.05)])}, index=idx.append(extra)).to_parquet(os.path.join(pdir, "beyond.parquet"))
    # Failed
    v5 = la.inspect_meter("egauge_1", 0, os.path.join(pdir, "beyond.parquet"), "egauge_1|toff", t, late, 0.5, m0, m1, 1)
    # Check
    assert v5 == "failed", v5
    # Case 6: the processed table covers the month but the feature month omits its first eight hours, every overlapping row and flag agrees, no independent explanation: not verified, never passed
    cut = 8 * 360
    # Verdict
    v6 = la.inspect_meter("egauge_1", 0, os.path.join(pdir, "egauge_1.parquet"), "egauge_1|toff", t[cut:], late[cut:], 0.5, m0, m1, 1)
    # Not verified
    assert v6 == "not verified", v6
    # Case 6b: the same partial coverage with independently given bounds that match the files exactly: passed, the uncovered rows explained
    v6b = la.inspect_meter("egauge_1", 0, os.path.join(pdir, "egauge_1.parquet"), "egauge_1|toff", t[cut:], late[cut:], 0.5, m0, m1, 1, expected=(int(t[cut]), m1))
    # Passed
    assert v6b == "passed", v6b
    # Case 6c: given bounds that do not match the files: not verified
    v6c = la.inspect_meter("egauge_1", 0, os.path.join(pdir, "egauge_1.parquet"), "egauge_1|toff", t[cut:], late[cut:], 0.5, m0, m1, 1, expected=(int(t[cut]) + 10, m1))
    # Not verified
    assert v6c == "not verified", v6c
    # Case 6d: full processed month and full feature coverage, said again after the partial cases: passed
    assert run("egauge_1", 0, os.path.join(pdir, "egauge_1.parquet"), late, t) == "passed"
    # Overall verdicts and codes
    assert la.overall({"a": "passed", "b": "passed"}) == ("passed", 0) and la.overall({"a": "passed", "b": "not verified"}) == ("not verified", 2) and la.overall({"a": "failed", "b": "not verified"}) == ("failed", 1) and la.overall({}) == ("not verified", 2)
    # main() exits nonzero when one meter is not verified: meter 2 gets a naive-index table whose offsets agree with its flags (never late)
    pd.DataFrame({"egauge_2|toff": np.full(len(t), 0.05)}, index=idx.tz_localize(None)).to_parquet(os.path.join(pdir, "egauge_2.parquet"))
    # Arguments
    saved = sys.argv
    # Set
    sys.argv = ["main_latealigncheck.py", "--processed-dir", os.path.join(tmp, "processed"), "--features-dir", fdir, "--meters", "egauge_1", "egauge_2", "--rows", "1"]
    # Run
    try:
        # Main
        la.main()
        # No exit is a failure
        raise AssertionError("main() did not exit")
    # Exit
    except SystemExit as e:
        # Not verified overall: meter 1 passed, meter 2 naive
        assert e.code == 2, e.code
    # Restore
    finally:
        # Arguments
        sys.argv = saved
    # Say
    print("alignment checker verdicts: aligned passed; altered flag failed; shifted rows failed; naive index not verified; rows beyond the month failed; feature month missing its first eight hours not verified, passed only with matching independently given bounds and not verified with non-matching ones; full coverage passed; overall codes 0/1/2; main() exits 2 when one meter is not verified: ok")


def check_torch(args):
    """Part 2: with torch, the masked safeguard and the checkpoint refusal for the neighbor angle options."""
    # Try torch
    try:
        # Import
        import torch
        # Models
        import socal_models as sm
    # Absent
    except ImportError as e:
        # Say
        print(f"torch part skipped here ({e}); run this check on Colab before training")
        # Done
        return
    # Sizes
    M, C, H, B = 4, 6, 12, 5
    # Manifest
    manifest = {"meters": [f"egauge_{i}" for i in range(M)], "channels": [f"ch{i}" for i in range(C)]}
    # Eight features
    nf = sd.F + len(sa.ANGDIFF_FEATURES)
    # Graph
    graph = {"nodes": [{"name": f"node_{i}"} for i in range(M)], "edges": [{"source": i, "target": i + 1, "admittance_abs": 1.0} for i in range(M - 1)], "meters": {f"egauge_{i}": {"node": i} for i in range(M)}}
    # Masked model
    model = sm.make_model("gcn", manifest, graph, edges="true", hidden=16, embed=4, seed=args.seed, masked=True, maskable_meters=manifest["meters"], n_features=nf)
    # Eval
    model.eval()
    # Inputs
    g = torch.Generator().manual_seed(args.seed)
    # Values
    xf = torch.rand(B, H, M, C, nf, generator=g)
    # Flags
    xf[..., 3] = 1.0
    # Angle flag
    xf[..., 4] = 1.0
    # Flat
    x1 = xf.reshape(B, H, M, C * nf)
    # Other added features
    xf2 = xf.clone()
    # Replace
    xf2[..., sd.F:] = torch.rand(B, H, M, C, len(sa.ANGDIFF_FEATURES), generator=g)
    # Flat
    x2 = xf2.reshape(B, H, M, C * nf)
    # Mask meter 0
    mask = torch.zeros(B, M, dtype=torch.bool)
    # Hide
    mask[:, 0] = True
    # Forecasts
    with torch.no_grad():
        # Masked
        y1, y2 = model(x1, mask), model(x2, mask)
        # Unmasked
        u1, u2 = model(x1), model(x2)
        # Scoring path
        f1, _ = sm.masked_forecast(model, x1, 2)
        # Again
        f2, _ = sm.masked_forecast(model, x2, 2)
    # Differences
    d_masked, d_open, d_score = float((y1 - y2).abs().max()), float((u1 - u2).abs().max()), float((f1 - f2).abs().max())
    # Say
    print(f"masked forward: largest difference from changing the added features {d_masked:.3e} (must be 0); unmasked {d_open:.3e} (must be > 0); scoring path {d_score:.3e} (must be 0)")
    # Verdicts
    if d_masked != 0.0 or d_score != 0.0 or d_open == 0.0:
        # Fail
        raise AssertionError("the masked safeguard does not hold for the neighbor angle inputs")
    # Checkpoint refusal
    plain = sm.make_model("gru", manifest, hidden=8, embed=2, seed=args.seed, n_features=nf)
    # Options
    options = {k: None for k in sm.EXPERIMENT_KEYS}
    # Fill
    options.update({"kind": "gru", "physics": "none", "n_features": nf, "features": list(sd.FEATURES) + list(sa.ANGDIFF_FEATURES), "physics_scale": None, "angdiff": "late_manifest.json:abc", "angdiff_settings": {"min_resultant": 0.5, "late_threshold": 0.5, "phase_map": "p1", "weights": "equal", "period_rule": "each slot's own period", "n_edge_sets": 10, "graph_id": "g:1"}})
    # Temporary folder
    with tempfile.TemporaryDirectory() as tmp:
        # Path
        path = os.path.join(tmp, "checkpoint.pt")
        # Save
        sm.save_checkpoint(path, plain, torch.optim.Adam(plain.parameters()), options, 1, 1.0, 0, np.random.default_rng(0))
        # Same loads
        sm.load_checkpoint(path, plain, dict(options))
        # Say
        print("checkpoint with identical options: loads")
        # Changes
        changes = {"angdiff": "none", "angdiff_settings": dict(options["angdiff_settings"], min_resultant=0.6), "n_features": sd.F, "features": list(sd.FEATURES) + ["nb_sin", "nb_cos", "nb_valid"]}
        # Each
        for key, value in changes.items():
            # Other
            other = dict(options)
            # Change
            other[key] = value
            # Must raise naming the key
            try:
                # Attempt
                sm.load_checkpoint(path, plain, other)
                # Failure
                raise AssertionError(f"checkpoint loaded although {key} differed")
            # Expected
            except ValueError as e:
                # Named
                if key not in str(e):
                    # Fail
                    raise AssertionError(f"refusal for {key} does not name it: {e}")
                # Say
                print(f"changed {key}: refused as required")


def main():
    """Run the parts."""
    # Options
    args = parse_args()
    # Part 1
    check_calculation()
    # Awkward periods and structural eligibility on synthetic timelines
    check_period_fixtures()
    # The real file
    check_real_graph(args.graph)
    # Data side in a temporary folder
    with tempfile.TemporaryDirectory() as tmp:
        # Run
        check_data_side(tmp)
    # The alignment checker's verdicts, in another temporary folder
    with tempfile.TemporaryDirectory() as tmp:
        # Run
        check_align_verdicts(tmp)
    # Part 2
    check_torch(args)
    # Done
    print("neighbor angle input checks passed")


# Entry point
if __name__ == "__main__":
    # Run
    main()
