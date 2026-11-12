# =============================================================================
# File:        main_graphdelivery.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Functional check that the time-indexed meter graph reaches
#              the model as intended. The graph audit read the code; this
#              check runs it, on the real graph file and on small controlled
#              fixtures, so that delivery is shown rather than argued:
#                1. graph selection at every switching boundary, on the
#                   pipeline's aligned ten-second slot grid: the period the
#                   model's dataset class assigns to each window (from its
#                   last history slot) equals an independent lookup written
#                   here; periods that hold no aligned slot are reported,
#                   never given an invented slot;
#                2. edge and weight delivery: every edge set reaches the
#                   model with both directions of every recorded edge, in
#                   range, with the weight the file holds (1.0 on every
#                   meter-path edge); two independently constructed models
#                   under edge_weights admittance and binary have identical
#                   graph buffers before anything is copied, a copy of the
#                   learned parameters only leaves the buffers unchanged,
#                   and the two then give identical forecasts; a model with
#                   intentionally different edges is detected by the same
#                   buffer comparison;
#                3. batch isolation: stacked examples keep their edges inside
#                   their own node blocks, and changing one example's inputs
#                   leaves the other examples' forecasts unchanged;
#                4. message reach on a controlled fixture (three meters in a
#                   path a - b - c, one edge set, several random
#                   initializations): a one-hop neighbor can change a
#                   meter's forecast, a two-hop meter never does, and an
#                   isolated meter is unchanged by every other meter;
#                   "can change" is reported as the number of
#                   initializations in which the change exceeded the
#                   tolerance, since a random model need not respond to
#                   every perturbation; "never" is required in all;
#                5. no-edge invariance: a model built with --edges none gives
#                   every meter a forecast no other meter can change;
#                6. sender inputs: a meter whose future target is missing
#                   still has its history in the model input, so it remains
#                   available as a sender (only its loss term is masked).
#              It tests the plumbing with random initial weights, not the
#              trained models (trained no-edge invariance was shown by
#              checks/main_neighborcheck.py control 2 in Step 72). It prints
#              how many edge sets, boundaries and reach cases it exercised.
#              Needs torch and torch_geometric. It writes nothing.
#
#              Important: the project starts the forecast heads at zero
#              (zero_head) so that the first forecast under predict_delta is
#              persistence; a freshly built model therefore ignores every
#              node state. The first execution of this check (Step 77,
#              commit 2a3c381) showed exactly that: no response to a
#              neighbor, and forecast invariances that held trivially. The
#              check now (1) prints the zero-head lines it finds in the
#              checked-out socal_models.py, (2) gives every diagnostic
#              model's two heads controlled nonzero parameters before any
#              forecast-based case, (3) reads the learned correction, the
#              forecast minus the persistence base (sm.persistence_forecast
#              with the model's own keep rule), instead of the full
#              forecast, so persistence alone cannot produce a change, and
#              (4) runs positive controls that must respond: the correction
#              changes when the meter's own earlier history changes with the
#              base held fixed (main model, every fixture model, no-edge
#              model), a fixture meter's correction changes when a connected
#              neighbor's history changes, and on the real graph at least
#              one connected meter's correction responds to a neighbor in
#              the same model and input used for the isolated-meter cases,
#              per edge set. The positive controls attempted and the
#              positive controls responding are printed as separate counts,
#              with the number of real-graph neighbor pairs tried.
#
#              Test inputs: all-valid synthetic inputs. Every random input
#              has its validity flags (features 3 and 4 of each channel) set
#              to 1.0. Real inputs contain validity flags of 0 or 1.
#              The model reads a reading as valid when its flag is above
#              0.5, and persistence copies the last valid slot. With random
#              flags the last slot is rarely valid on every channel, so the
#              persistence base would come from earlier slots and could not
#              be held fixed. With flags of 1.0 the base is the last slot.
#
#              Status: the first version was executed once on Colab (Step
#              77, commit 2a3c381, exit 1, the neighbor response case
#              failed for the reason above). This revision is
#              syntax-checked only. The parts that need torch have not been
#              executed, so its next Colab run is their first execution.
#              Signatures were read from socal_models.py:
#              GraphWindowDataset(X, mag_valid, ang_valid, history, horizon,
#              max_gap, min_meters=1, t=None, timeline=None); make_model(kind,
#              manifest, graph=None, edges=..., edge_weights=..., ..., seed=0,
#              ..., timeline=None); GraphForecaster.forward(x, mask=None,
#              graph_ids=None); edge_set(k); batch_edges(B, graph_ids, device);
#              persistence_forecast(x, n_channels).
#
# Usage (from the repo root):
#   python checks/main_graphdelivery.py --features-dir /content/features
#   python checks/main_graphdelivery.py --features-dir /content/features --tolerance 1e-6 --reach-seeds 5
#
# Functions:
#   parse_args()                     command-line options with defaults
#   report(name, ok, note)           print one PASS or FAIL line, return ok
#   lookup_period(periods, t)        independent period lookup: linear scan over [start, end)
#   aligned_grid(c, history)         aligned ten-second slots around a change time
#   fixture_manifest_and_timeline()  three-meter path fixture a - b - c
#   zero_head_lines(root)            line numbers of the zero-head calls in the checked-out model code
#   randomize_heads(model, seed)     controlled nonzero forecast heads, since the project starts them at zero
#   forecasts(model, x, gid)         forecast for a batch under one graph id per example
#   correction(model, x, gid)        the learned correction: forecast minus the persistence base the model adds
#   max_change(a, b)                 largest absolute difference
#   valid_flags(x, C)                set the validity flags of a test input to 1.0
#   draw_inputs(shape, C)            a random test input with valid flags
#   redraw(x, C)                     new random values of the same shape with valid flags
#   own_history_control(model, x, gid, b, m)
#                                    positive control: change a meter's earlier history, base held fixed
#   main()                           run every case, exit 0 when all pass
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Standard library: exit code
import sys
# Third party: numerics
import numpy as np
# Third party: tensors
import torch

# The repo root, one level above checks/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Import from the root
sys.path.insert(0, ROOT)

# Project: model, graph loading, dataset
import socal_models as sm
# Project: feature count and slot length
import socal_modeldata as sd

# Slot length in seconds, the pipeline's grid
SLOT = 10


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Functional check that the time-indexed graph reaches the model as intended")
    # Graph
    parser.add_argument("--graph", default="graph_meters_periods.json", help="Time-indexed meter graph")
    # Manifest source
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with manifest.json (meter and channel order)")
    # Tolerance for "unchanged", absolute
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Largest absolute change that counts as unchanged (rtol is 0)")
    # History
    parser.add_argument("--history", type=int, default=60, help="History slots, as trained")
    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random weights and inputs")
    # Reach draws
    parser.add_argument("--reach-seeds", type=int, default=5, help="Random initializations tried for the message-reach fixture")
    # Parse
    return parser.parse_args()


def report(name, ok, note=""):
    """Print one PASS or FAIL line and return ok."""
    # Line
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + note if note else ''}")
    # Pass through
    return bool(ok)


def lookup_period(periods, t):
    """Independent period lookup: the index of the period with start_utc <= t < end_utc_exclusive, by linear scan; -1 when none."""
    # Scan
    for p in periods:
        # Inside
        if int(p["start_utc"]) <= int(t) < int(p["end_utc_exclusive"]):
            # Found
            return int(p["index"])
    # None
    return -1


def aligned_grid(c, history):
    """Aligned ten-second slots from history + 20 slots before the change time c to 20 slots after it; slots are multiples of SLOT seconds, as the pipeline floors times to the grid."""
    # First aligned slot at or before c
    c0 = (int(c) // SLOT) * SLOT
    # Grid
    return c0 + SLOT * np.arange(-(history + 20), 21)


def fixture_manifest_and_timeline(F):
    """A three-meter path fixture a - b - c with one edge set and one period; returns (manifest, timeline body, timeline)."""
    # Manifest
    manifest = {"meters": ["a", "b", "c"], "channels": ["L1"]}
    # Graph body in the time-indexed layout
    body = {"nodes": [{"meter": "a"}, {"meter": "b"}, {"meter": "c"}], "meters": {"a": {"node": 0}, "b": {"node": 1}, "c": {"node": 2}},
            "graphs": [{"index": 0, "edges": [{"source": 0, "target": 1, "admittance_abs": 1.0}, {"source": 1, "target": 2, "admittance_abs": 1.0}]}],
            "periods": [{"index": 0, "start_utc": 0, "end_utc_exclusive": 10 ** 10, "graph": 0}]}
    # Timeline
    return manifest, body, sm.load_timeline(body)


def zero_head_lines(root):
    """Line numbers in the checked-out socal_models.py where the heads are zeroed, and the definition of zero_head."""
    # Read
    with open(os.path.join(root, "socal_models.py")) as f:
        # Lines
        lines = f.readlines()
    # Matches
    return [(i + 1, line.strip()) for i, line in enumerate(lines) if "zero_head(" in line and ("self.head" in line or line.strip().startswith("def zero_head"))]


def randomize_heads(model, seed):
    """
    Give the model's forecast heads random weights. The project starts the
    heads at zero on purpose (zero_head, so the first forecast under
    predict_delta is persistence itself), which makes a freshly built
    model's forecast independent of every encoder state, its own and its
    neighbors'. A delivery check on such a model would pass every
    invariance case trivially and fail every response case, so the heads
    are randomized here before any forecast is compared.
    """
    # Deterministic
    g = torch.Generator().manual_seed(seed)
    # Both heads
    for head in (model.head_mag, model.head_ang):
        # No gradients
        with torch.no_grad():
            # Random weights
            head.weight.copy_(torch.randn(head.weight.shape, generator=g) * 0.1)
            # Random bias
            if head.bias is not None:
                # Set
                head.bias.copy_(torch.randn(head.bias.shape, generator=g) * 0.1)
    # Done
    return model


def forecasts(model, x, gid):
    """The model's forecast (B, M, C, 3) for inputs x under graph ids gid, no gradients."""
    # No gradients
    with torch.no_grad():
        # Forward
        return model(x, graph_ids=gid)


def correction(model, x, gid):
    """
    The learned correction of the model's forecast: the forecast minus the
    persistence base the model adds under predict_delta (same keep rule as
    the model: magnitude kept where a valid magnitude existed, both angle
    components where a valid angle existed). Persistence alone cannot change
    this quantity, so a change in it comes from the heads' inputs.
    """
    # Forecast
    y = forecasts(model, x, gid)
    # Not a delta model: the forecast is the readout itself
    if not getattr(model, "predict_delta", False):
        # Done
        return y
    # Persistence base and validity
    base, ok = sm.persistence_forecast(x, model.C)
    # The model's keep rule
    keep = torch.cat([ok[..., 0:1], ok[..., 1:2], ok[..., 1:2]], dim=-1)
    # Subtract
    return y - torch.where(keep, base, torch.zeros_like(base))


def max_change(a, b):
    """Largest absolute difference between two tensors."""
    # Value
    return float((a - b).abs().max())


def valid_flags(x, C):
    """
    Set the two validity flags (features 3 and 4 of every channel) of a test
    input to 1.0 and return the result: an all-valid synthetic input. x has
    C * F numbers in its last dimension. Real inputs contain validity flags
    of 0 or 1, and the model treats a reading as valid when its flag is
    above 0.5. Random flags would make
    most readings invalid, so the persistence base would be copied from an
    earlier slot instead of the last one.
    """
    # Split the last dimension into channels and features, on a copy
    xf = x.reshape(x.shape[:-1] + (C, x.shape[-1] // C)).clone()
    # Mark every magnitude and every angle as valid
    xf[..., 3:5] = 1.0
    # Back to the input layout
    return xf.reshape(x.shape)


def draw_inputs(shape, C):
    """A random test input of the given shape (last dimension C * F) with every validity flag at 1.0."""
    # Random values, then valid flags
    return valid_flags(torch.randn(*shape), C)


def redraw(x, C):
    """New random values with the shape of x (last dimension C * F) and every validity flag at 1.0, used to perturb part of an input."""
    # Random values, then valid flags
    return valid_flags(torch.randn_like(x), C)


def own_history_control(model, x, gid, b, m):
    """
    Positive control on one model: replace meter m's history in example b in
    every slot but the last, and return (base_same, change). base_same says
    the persistence base of that meter is identical before and after, which
    holds since the last slot is untouched and valid. change is the largest
    change of the meter's learned correction. A live path from the meter's
    history to the heads gives base_same True and a change above the
    tolerance.
    """
    # Copy of the input
    xo = x.clone()
    # New earlier history for the meter; the last slot stays as it was
    xo[b, :-1, m, :] = redraw(xo[b, :-1, m, :], model.C)
    # Persistence base of the meter before and after
    base_same = bool(torch.equal(sm.persistence_forecast(x, model.C)[0][b, m], sm.persistence_forecast(xo, model.C)[0][b, m]))
    # Change of the meter's learned correction
    change = max_change(correction(model, x, gid)[b, m], correction(model, xo, gid)[b, m])
    # Both
    return base_same, change


def main():
    """Run every case and exit 0 when all pass."""
    # Options
    args = parse_args()
    # Seeds
    torch.manual_seed(args.seed)
    # Manifest
    manifest = sd.load_manifest(args.features_dir)
    # Graph file
    body = sm.load_graph(args.graph)
    # Timeline
    timeline = sm.load_timeline(body)
    # Sizes
    M, C = len(manifest["meters"]), len(manifest["channels"])
    # Features per channel
    F = sd.n_features()
    # Node names
    names = [n["meter"] for n in body["nodes"]]
    # All pass so far
    ok = True
    # Alignment first
    ok &= report("meter order of the manifest equals the graph node order", list(manifest["meters"]) == names)
    # Where the checked-out model code zeroes the heads
    zl = zero_head_lines(ROOT)
    # Print
    print("zero-head initialization in the checked-out socal_models.py: " + "; ".join(f"line {n}: {t}" for n, t in zl))
    # Must be found, since the readout below depends on it being understood
    ok &= report("the checked-out model code zeroes both heads under predict_delta (zero_head calls found)", any("head_mag" in t for _, t in zl) and any("head_ang" in t for _, t in zl))
    # Positive controls that responded
    n_pos = 0
    # Positive controls attempted, counted apart from those that responded
    n_att = 0
    # Neighbor pairs tried by the real-graph positive controls, over all edge sets
    n_pairs = 0
    # ---- 1. graph selection at every switching boundary, on the aligned grid ----
    periods = timeline["periods"]
    # Periods that received at least one window as selected graph
    seen = set()
    # Boundaries exercised
    n_bound = 0
    # Each change time
    for p in periods[1:]:
        # Change time
        c = int(p["start_utc"])
        # Aligned grid around it
        t = aligned_grid(c, args.history)
        # Sizes
        T = len(t)
        # Inputs, all valid
        X = np.zeros((T, M, C, F), dtype=np.float32)
        # Validity
        valid = np.ones((T, M, C), dtype=bool)
        # The model's dataset with the timeline
        ds = sm.GraphWindowDataset(X, valid, valid, args.history, 1, 12, 1, t=t, timeline=timeline)
        # Last history slot of every window
        last = t[ds.starts + args.history - 1]
        # Independent lookup
        exp = np.array([lookup_period(periods, v) for v in last])
        # Agreement
        agree = np.array_equal(exp, np.asarray(ds.period))
        # The windows whose last history slot is the greatest aligned slot strictly before the change, and the first at or after it, if the grid has them
        i_before = np.flatnonzero(last == ((c - 1) // SLOT) * SLOT)
        # At or after
        i_at = np.flatnonzero(last >= c)
        # Note
        note = f"{len(ds.starts)} windows; period of last slot before the change {int(ds.period[i_before[0]]) if len(i_before) else 'n/a'}, first window at or after the change {int(ds.period[i_at[0]]) if len(i_at) else 'n/a'} (period {p['index']} starts at {c}, aligned {c % SLOT == 0})"
        # Record
        seen |= set(np.asarray(ds.period).tolist())
        # Count
        n_bound += 1
        # Check
        ok &= report(f"boundary {p['start']}: dataset period equals the independent lookup for every window on the aligned grid", agree, note)
    # Periods with no aligned slot: a period shorter than a slot whose span holds no multiple of SLOT
    no_slot = [q["index"] for q in periods if ((int(q["start_utc"]) + SLOT - 1) // SLOT) * SLOT >= int(q["end_utc_exclusive"])]
    # Report
    print(f"boundaries exercised: {n_bound} of {len(periods) - 1}; periods selected by at least one window: {sorted(seen)}; periods holding no aligned slot: {no_slot if no_slot else 'none'}")
    # Every period that holds a slot was selected at least once
    ok &= report("every period that holds an aligned slot was selected by some window", set(q["index"] for q in periods if q["index"] not in no_slot) <= seen)
    # ---- 2. edge and weight delivery ----
    model = sm.make_model("gcn", manifest, edges="true", edge_weights="admittance", seed=args.seed, timeline=timeline)
    # The heads start at zero by design; randomize them so forecasts depend on the encoder and node states
    randomize_heads(model, args.seed)
    # Say so once
    print("note: forecast heads given nonzero test weights (the project zero-initializes them for predict_delta); test inputs are all-valid synthetic inputs (real inputs contain validity flags of 0 or 1); every response and invariance case below reads the learned correction, and the positive controls show whether it responds")
    # Evaluation mode
    model.eval()
    # Each edge set
    for g, graph in enumerate(body["graphs"]):
        # Delivered tensors
        ei, ew = model.edge_set(g)
        # Recorded pairs
        pairs = {(e["source"], e["target"]) for e in graph["edges"]}
        # Delivered directed pairs
        got = {(int(a), int(b)) for a, b in ei.t().tolist()}
        # Both directions
        both = pairs | {(b, a) for a, b in pairs}
        # Check
        ok &= report(f"edge set {g}: {len(pairs)} recorded edges delivered as {ei.shape[1]} directed edges, both directions, nothing else, in range", got == both and ei.shape[1] == 2 * len(pairs) and int(ei.max()) < M and int(ei.min()) >= 0)
        # Weights
        ok &= report(f"edge set {g}: every delivered weight is 1.0", bool(torch.allclose(ew, torch.ones_like(ew), rtol=0, atol=0)) and len(ew) == ei.shape[1])
    # Binary against admittance: an independently constructed model, its graph buffers compared BEFORE anything is copied
    model_b = sm.make_model("gcn", manifest, edges="true", edge_weights="binary", seed=args.seed, timeline=timeline)
    # Independently built edge tensors must already agree (indices equal, weights equal ones)
    same_graph = all(torch.equal(model.edge_set(g)[0], model_b.edge_set(g)[0]) and torch.equal(model.edge_set(g)[1], model_b.edge_set(g)[1]) for g in range(len(body["graphs"])))
    # Check
    ok &= report("independently constructed edge indices and weights under admittance and binary are identical before any copy", same_graph)
    # Snapshot of model_b's graph buffers before the parameter copy
    before = {name: buf.clone() for name, buf in model_b.named_buffers()}
    # Copy learned parameters only, by name, never buffers
    with torch.no_grad():
        # Each parameter
        for (name, p_src), (name_b, p_dst) in zip(model.named_parameters(), model_b.named_parameters()):
            # Names must line up
            if name != name_b:
                # Stop
                raise SystemExit(f"parameter order differs between the two models: {name} against {name_b}")
            # Copy
            p_dst.copy_(p_src)
    # Buffers unchanged by the copy
    unchanged = all(torch.equal(before[name], buf) for name, buf in model_b.named_buffers())
    # Check
    ok &= report("the parameter copy left every graph buffer unchanged", unchanged, f"{len(before)} buffers")
    # Evaluation
    model_b.eval()
    # Inputs: random values with valid flags, so the persistence base is the last slot
    x = draw_inputs((4, args.history, M, C * F), C)
    # Graph ids: two examples on graph 0, one on graph 1, one on the last graph
    gid = torch.tensor([0, 0, 1, len(body["graphs"]) - 1])
    # Learned corrections (forecast minus persistence base), the readout used for every forecast-based case
    ya, yb = correction(model, x, gid), correction(model_b, x, gid)
    # Positive control, main model: the correction responds to the meter's own earlier history while the persistence base is held fixed
    base_same, own_resp = own_history_control(model, x, gid, 0, 0)
    # Passed
    passed = base_same and own_resp > args.tolerance
    # One control attempted
    n_att += 1
    # Count
    n_pos += int(passed)
    # Report
    ok &= report("positive control, main model: the learned correction responds to the meter's own earlier history with the persistence base held fixed", passed, f"base fixed {base_same}, correction change {own_resp:.2e}")
    # Equivalence on the corrections, identical heads included through the parameter copy
    ok &= report("with identical learned parameters, heads included, admittance and binary models give identical corrections (rtol 0)", bool(torch.allclose(ya, yb, rtol=0, atol=args.tolerance)), f"largest difference {max_change(ya, yb):.2e}")
    # An intentionally different graph buffer must be detected, not overwritten: a random-edge model has different edge indices
    model_r = sm.make_model("gcn", manifest, edges="random", edge_weights="admittance", seed=args.seed, timeline=timeline)
    # Compare buffers as above
    differs = any(not torch.equal(model.edge_set(g)[0], model_r.edge_set(g)[0]) for g in range(len(body["graphs"])))
    # Check
    ok &= report("an intentionally different graph buffer (random edges) is detected by the buffer comparison", differs)
    # Coverage: how many edge sets the delivery checks covered
    print(f"edge sets exercised: {len(body['graphs'])}")
    # ---- 3. batch isolation ----
    ei, ew = model.batch_edges(4, gid, x.device)
    # Block of every endpoint
    blk = ei // M
    # Both endpoints in the same block
    blocks_ok = bool((blk[0] == blk[1]).all())
    # Each example's count
    for b in range(4):
        # Count in block b
        blocks_ok &= int((blk[0] == b).sum()) == 2 * len(body["graphs"][int(gid[b])]["edges"])
    # Check
    ok &= report("batch edges: every edge joins two nodes of the same example and each example carries its own edge set's count", blocks_ok, f"{ei.shape[1]} directed edges over 4 examples")
    # Perturb example 1
    x2 = x.clone()
    # New random history for every meter of example 1
    x2[1] = redraw(x2[1], C)
    # Corrections
    y2 = correction(model, x2, gid)
    # Positive control inside the batch case: the perturbed example's own corrections must change
    batch_resp = max_change(ya[1], y2[1])
    # One control attempted
    n_att += 1
    # Count
    n_pos += int(batch_resp > args.tolerance)
    # Check: the perturbed example responds and the others are unchanged
    ok &= report("batch isolation: the perturbed example's corrections change and the other examples' corrections are unchanged", batch_resp > args.tolerance and max_change(ya[[0, 2, 3]], y2[[0, 2, 3]]) <= args.tolerance, f"perturbed change {batch_resp:.2e}, largest change elsewhere {max_change(ya[[0, 2, 3]], y2[[0, 2, 3]]):.2e}")
    # ---- 4. message reach on the controlled fixture ----
    fman, fbody, ftl = fixture_manifest_and_timeline(F)
    # Counters
    n_resp_nb, n_resp_two, n_own_fix = 0, 0, 0
    # Draws
    for s in range(args.reach_seeds):
        # Model
        fm = sm.make_model("gcn", fman, edges="true", edge_weights="admittance", seed=args.seed + s, timeline=ftl)
        # Random heads, as above
        randomize_heads(fm, args.seed + s)
        # Evaluation
        fm.eval()
        # Inputs: three meters, one channel
        torch.manual_seed(args.seed + s)
        # Base input with valid flags; the fixture has one channel
        xf = draw_inputs((1, args.history, 3, 1 * F), 1)
        # Base correction
        yf = correction(fm, xf, torch.tensor([0]))
        # Own-history positive control on this fixture model: a's earlier history, base held fixed
        f_same, f_resp = own_history_control(fm, xf, torch.tensor([0]), 0, 0)
        # Counted only when the base stayed fixed and the correction changed
        n_own_fix += int(f_same and f_resp > args.tolerance)
        # Perturb b (a's neighbor)
        xb = xf.clone()
        # Random history for b
        xb[:, :, 1, :] = redraw(xb[:, :, 1, :], 1)
        # Response of a's correction
        n_resp_nb += int(max_change(yf[0, 0], correction(fm, xb, torch.tensor([0]))[0, 0]) > args.tolerance)
        # Perturb c (two hops from a)
        xc = xf.clone()
        # Random history for c
        xc[:, :, 2, :] = redraw(xc[:, :, 2, :], 1)
        # a's correction must not respond
        n_resp_two += int(max_change(yf[0, 0], correction(fm, xc, torch.tensor([0]))[0, 0]) > args.tolerance)
    # Positive controls on the fixture models: own history must respond in every initialization
    ok &= report(f"positive control, fixture models: a's correction responded to its own earlier history with the persistence base held fixed in {n_own_fix} of {args.reach_seeds} initializations (all required)", n_own_fix == args.reach_seeds)
    # One own-history control attempted per initialization
    n_att += args.reach_seeds
    # Count
    n_pos += n_own_fix
    # Neighbor response
    ok &= report(f"fixture a-b-c: a's correction responded to its neighbor b in {n_resp_nb} of {args.reach_seeds} initializations (at least one required)", n_resp_nb >= 1)
    # One neighbor control attempted per initialization
    n_att += args.reach_seeds
    # Count
    n_pos += n_resp_nb
    # Two hops never
    ok &= report(f"fixture a-b-c: a's correction never responded to the two-hop meter c ({n_resp_two} of {args.reach_seeds} responded; zero required)", n_resp_two == 0)
    # Isolated meters on the real graph: unchanged by every other meter, in every edge set where they are isolated
    n_iso_cases = 0
    # Each edge set
    for g, graph in enumerate(body["graphs"]):
        # Degrees
        deg = np.zeros(M, dtype=int)
        # Count
        for e in graph["edges"]:
            # Both ends
            deg[e["source"]] += 1
            # Target
            deg[e["target"]] += 1
        # Base input and correction for this edge set, shared by the isolated cases and the positive control
        x1 = x[:1].clone()
        # Base
        y1 = correction(model, x1, torch.tensor([g]))
        # Isolated
        for i in np.flatnonzero(deg == 0):
            # Perturb all others
            xp = x1.clone()
            # Others
            others = [j for j in range(M) if j != i]
            # Random
            xp[:, :, others, :] = redraw(xp[:, :, others, :], C)
            # Correction
            yp = correction(model, xp, torch.tensor([g]))
            # Count
            n_iso_cases += 1
            # Check
            ok &= report(f"edge set {g}: isolated {names[i]} unchanged when every other meter changes", max_change(y1[0, i], yp[0, i]) <= args.tolerance, f"largest change {max_change(y1[0, i], yp[0, i]):.2e}")
        # Real-graph positive control: in the same model and input, at least one connected meter's correction responds to a neighbor
        nb_sets = {}
        # Neighbor lists from this edge set
        for e in graph["edges"]:
            # Both ways
            nb_sets.setdefault(e["source"], set()).add(e["target"])
            # Reverse
            nb_sets.setdefault(e["target"], set()).add(e["source"])
        # Try connected meters in order until one responds
        responded = None
        # Neighbor pairs tried in this edge set
        tried = 0
        # Loop
        for i in sorted(nb_sets):
            # Its first neighbor
            j = sorted(nb_sets[i])[0]
            # One more pair tried
            tried += 1
            # Perturb the neighbor only
            xn = x1.clone()
            # Random history for j
            xn[:, :, j, :] = redraw(xn[:, :, j, :], C)
            # Response of i's correction
            r = max_change(y1[0, i], correction(model, xn, torch.tensor([g]))[0, i])
            # Found
            if r > args.tolerance:
                # Record
                responded = (i, j, r)
                # Stop
                break
        # One control attempted for this edge set
        n_att += 1
        # Pairs tried, added to the total
        n_pairs += tried
        # Count
        n_pos += int(responded is not None)
        # Check
        ok &= report(f"positive control, edge set {g}: a connected meter's correction responds to a neighbor in the same model and input as the isolated cases", responded is not None, f"{names[responded[0]]} responded to {names[responded[1]]} by {responded[2]:.2e}; neighbor pairs tried {tried}" if responded else f"no connected meter responded; neighbor pairs tried {tried}")
    # Coverage
    print(f"reach cases exercised: fixture neighbor {args.reach_seeds} draws, fixture two-hop {args.reach_seeds} draws, isolated-meter cases on the real graph {n_iso_cases}, real-graph neighbor positive controls {len(body['graphs'])}")
    # ---- 5. no-edge invariance ----
    model_n = sm.make_model("gcn", manifest, edges="none", edge_weights="admittance", seed=args.seed, timeline=timeline)
    # Random heads, as above
    randomize_heads(model_n, args.seed)
    # Evaluation
    model_n.eval()
    # Base
    x1 = x[:1].clone()
    # Base correction under graph 0
    y1 = correction(model_n, x1, torch.tensor([0]))
    # Positive control on the no-edge model: own earlier history, base held fixed
    same_n, own_n = own_history_control(model_n, x1, torch.tensor([0]), 0, 0)
    # Passed only when the base stayed fixed and the correction changed
    passed_n = same_n and own_n > args.tolerance
    # One control attempted
    n_att += 1
    # Count
    n_pos += int(passed_n)
    # Check
    ok &= report("positive control, no-edge model: the learned correction responds to the meter's own earlier history with the persistence base held fixed", passed_n, f"base fixed {same_n}, correction change {own_n:.2e}")
    # Worst change over meters
    worst = 0.0
    # Each meter
    for i in range(M):
        # Copy
        xp = x1.clone()
        # Others
        others = [j for j in range(M) if j != i]
        # Random
        xp[:, :, others, :] = redraw(xp[:, :, others, :], C)
        # Correction
        yp = correction(model_n, xp, torch.tensor([0]))
        # Largest change on meter i
        worst = max(worst, max_change(y1[0, i], yp[0, i]))
    # Check
    ok &= report("no-edge model: no meter's correction changes when every other meter changes", worst <= args.tolerance, f"largest change {worst:.2e}")
    # Positive controls attempted, with what they are made of
    print(f"positive controls attempted: {n_att} (main model own history 1, batch perturbed example 1, fixture own history {args.reach_seeds}, fixture neighbor {args.reach_seeds}, real-graph neighbor controls {len(body['graphs'])}, no-edge own history 1)")
    # Positive controls responding, a separate count; the fixture neighbor control needs at least one response, every other control needs all
    print(f"positive controls responding: {n_pos} of {n_att} attempted (fixture neighbor responses {n_resp_nb} of {args.reach_seeds}, at least one required)")
    # Real-graph neighbor pairs tried, since each edge set's control tries pairs until one responds
    print(f"real-graph neighbor pairs tried: {n_pairs} over {len(body['graphs'])} edge sets")
    # ---- 6. a history-valid sender stays in the model input when its target is missing ----
    T = args.history + 1
    # Fixture arrays: meter b has valid history and an invalid target slot
    Xs = np.ones((T, 3, 1, F), dtype=np.float32)
    # Validity
    vs = np.ones((T, 3, 1), dtype=bool)
    # b's target invalid
    vs[T - 1, 1, 0] = False
    # Dataset
    ds = sm.GraphWindowDataset(Xs, vs, vs, args.history, 1, 12, 1, t=SLOT * np.arange(T), timeline=ftl)
    # One window; its x must carry b's history and its y_valid must exclude b
    xw, yw, yv, extra = ds[0]
    # Check
    ok &= report("sender with a missing future target: its history is in the model input and only its loss term is masked", len(ds) == 1 and bool(torch.as_tensor(xw[:, 1, :]).eq(1.0).all().item()) and not bool(yv[1].any()) and bool(yv[0].all()) and bool(ds.usable[0, 1] == False))
    # Verdict
    print("\nall checks passed" if ok else "\nsome checks FAILED")
    # Exit
    sys.exit(0 if ok else 1)


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
