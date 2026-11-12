# =============================================================================
# File:        main_maskedleakcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Leakage test for a trained masked run on the time-indexed
#              graph, as agreed in the masked design note (version 3,
#              sections 1 and 6). A masked model forecasts a hidden meter
#              without seeing that meter's readings. It keeps what it
#              learned about the meter (its embedding), the mask token and
#              the shared weights, and it receives the messages of the
#              meters connected to it. "Leakage" means the forecast can see
#              something it must not see: the hidden meter's own readings,
#              a meter the edges do not connect to it, the edges of another
#              period, another window in the same batch, or the added input
#              features, which are computed from other meters' readings.
#              checks/main_maskcheck.py covers a fixed graph and refuses a
#              time-indexed run, because it reads one edge set.
#
#              The test gives two separate verdicts.
#
#              Leakage invariance (a failure is a leak, exit code 1):
#                R1. real windows sampled from the data, eligibility
#                    computed once before anything is changed and then held
#                    fixed. For eligible (window, meter) pairs the hidden
#                    meter's history is replaced three ways: by zeros, by
#                    the same meter's history from another sampled window
#                    (a donor), and by random values in range. The masked
#                    forecast must not change, through the plain forward
#                    pass and through masked_forecast, the routine used for
#                    evaluation and scoring
#                R2. on the same windows, masked_forecast agrees with the
#                    plain forward pass
#                S1. made-up windows, every edge set, every meter that may
#                    be hidden: the hidden meter's own history does not
#                    reach its forecast
#                S3. changing every meter the saved edges do not connect to
#                    the hidden one, all at once, changes nothing
#                S4. two windows in one batch with different edge sets each
#                    follow their own edge set
#                S5. a batch mixing every edge set matches each window run
#                    alone, and changing one window changes no other
#                S6. masked_forecast flags the meters that may never be
#                    hidden as invalid, matches one pass per meter, and
#                    refuses to run without the edge set of each window
#                A1. the added input features change nothing in a masked
#                    forward, with two controls so that this cannot pass
#                    for a trivial reason: without a mask the same change
#                    does move the output, and a visible meter's own
#                    readings do move its forecast. A run with added
#                    features is tested itself. A run without them, as
#                    every run of this study, has nothing to perturb, so
#                    the code path is tested on a separate freshly built
#                    model with added features on the same graph, and the
#                    output says so
#              Neighbor response (no response is unresolved, exit code 2):
#                N.  changing a connected meter that has usable history
#                    changes the hidden meter's forecast, on the real
#                    windows and on the made-up ones. A model with
#                    connections that never responds is not a leak and not
#                    a reason to drop it. It means messages may not be
#                    arriving, which has to be investigated before the
#                    result is used. A model trained with no connections
#                    is exempt.
#
#              "Connected" means within as many edges as the run has
#              message-passing layers, counted on the edges the model
#              saved (after --edges true, none or random was applied).
#              Every forecast and every difference must be a finite
#              number: a NaN fails its case by name and cannot pass as "no
#              change". Every mandatory test must cover at least one case.
#              The made-up windows must leave every meter's history usable,
#              which is asserted. With sender gating, a meter two edges
#              away that loses its usable history can change the size of a
#              neighbor's message, because the layer rescales messages by
#              how many arrive. That is not a reading leaking, and the
#              assertion keeps it out of these tests.
#
#              The tests run on the CPU, where the same input gives the
#              same output bit for bit. With --time-batches the script also
#              measures, on real batches and on --time-device, the seconds
#              per forward call of masked_forecast, the forward calls per
#              window, the seconds per training step, and the largest GPU
#              memory this process allocated and reserved for them. The
#              training steps run on a copy of the model that is never
#              saved.
#
#              Exit codes: 0 both verdicts good, 1 a leak or a nonfinite
#              value, 2 the neighbor response is unresolved, 3 a required
#              part could not be run.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root.
#
# Usage:
#   python checks/main_maskedleakcheck.py --run-dir /content/drive/MyDrive/socal_runs/gcn_mask_dyn --graph graph_meters_periods.json --features-dir /content/features --splits splits_v2.json
#   With timing: add --time-batches 20
#   Options: python checks/main_maskedleakcheck.py --help
#
# Functions:
#   parse_args()            command-line options with defaults
#   report(name, ok, note)  print one PASS or FAIL line, return ok
#   graph_identity(path)    the graph file's name and content hash, as main_gnn.py records it
#   Tally                   largest difference over a set of cases, with nonfinite cases kept by name
#   Response                how often a connected meter moved the forecast
#   load_run(args)          options, manifest, timeline, scale and the model with the best weights
#   synthetic(B, H, M, C, seed, drop, extra)
#                           a made-up input window
#   usable(x, C, max_gap, what)
#                           stop unless every meter's history in x is usable
#   reach_sets(model, k, M, hops)
#                           meter index -> meters whose history may reach it under saved edge set k
#   hide(B, M, i)           a mask that hides meter i in every window
#   changed(x, other, meters)
#                           a copy of x with the listed meters' histories taken from other
#   real_pieces(args, options, manifest, timeline, scale)
#                           the datasets of the chosen real blocks, one per month piece
#   check_real(...)         R1, R2 and the neighbor response on real windows
#   check_synthetic(...)    S1, S3 to S6 and the neighbor response on made-up windows
#   check_added(...)        A1, the added input features
#   timing(...)             seconds per forward call and per training step, and this process's GPU memory
#   main()                  run the checks and give the two verdicts
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: a copy of the model for the timed training steps
import copy
# Standard library: content hash of the graph file
import hashlib
# Standard library: JSON reading
import json
# Standard library: whole forward passes per window
import math
# Standard library: repo root on the import path
import os
# Standard library: exit code
import sys
# Standard library: elapsed time
import time
# Third party: window sampling
import numpy as np
# Third party: deep learning
import torch

# The repo root, one level above checks/, so the project modules import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project: models
import socal_models as sm
# Project: data loading and constants
import socal_modeldata as sd
# Project: the history rule, shared with sender gating and the rule for hidden meters
import socal_sendergate as sg


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Leakage test for a trained masked run on the time-indexed graph")
    # Run folder
    parser.add_argument("--run-dir", required=True, help="Run folder holding checkpoint.pt, checkpoint_best.pt and scale.json")
    # Graph file the run was trained on
    parser.add_argument("--graph", default="graph_meters_periods.json", help="Time-indexed graph file the run used")
    # Feature files
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with <YYYY-MM>.npz and manifest.json")
    # Splits
    parser.add_argument("--splits", default="splits_v2.json", help="Split definition the run used")
    # Which split the real windows come from
    parser.add_argument("--split", default="validation", help="Split the real windows are sampled from")
    # Which of its blocks
    parser.add_argument("--blocks", type=int, nargs="+", default=[0], help="Indices of the blocks of --split to sample real windows from")
    # Real windows per block piece
    parser.add_argument("--windows", type=int, default=64, help="Real windows sampled per block piece")
    # Eligible pairs per block piece
    parser.add_argument("--pairs", type=int, default=256, help="At most this many eligible (window, meter) pairs tested per block piece")
    # Rows per forward pass in the real-window tests
    parser.add_argument("--rows", type=int, default=64, help="Rows per forward pass in the real-window tests")
    # Made-up windows per check
    parser.add_argument("--batch", type=int, default=2, help="Made-up windows per check")
    # Tolerance of the agreed test
    parser.add_argument("--tol", type=float, default=1e-5, help="Largest change that still counts as unchanged (1e-5 in the agreed test)")
    # Share of absent readings in the second kind of made-up window
    parser.add_argument("--drop", type=float, default=0.2, help="Share of readings marked absent in the second kind of made-up window")
    # Added features of the separate model
    parser.add_argument("--extra-features", type=int, default=4, help="Added features per channel of the separate model used for the added-feature test (4 = the physics inputs)")
    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Seed of the window sample, the donors and the made-up windows")
    # Timing
    parser.add_argument("--time-batches", type=int, default=0, help="Real batches timed for forward calls and training steps; 0 = no timing")
    # Device of the timing
    parser.add_argument("--time-device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device of the timing; the tests themselves run on the CPU")
    # Parse
    args = parser.parse_args()
    # Sizes that must be positive, and two windows at least so that a donor exists
    if args.windows < 2 or args.pairs < 1 or args.rows < 1 or args.batch < 1 or args.extra_features < 1 or args.time_batches < 0:
        # Refuse
        parser.error("--windows must be at least 2; --pairs, --rows, --batch and --extra-features at least 1; --time-batches at least 0")
    # A tolerance and a share that make sense
    if not (np.isfinite(args.tol) and args.tol >= 0 and 0 <= args.drop < 1):
        # Refuse
        parser.error("--tol must be finite and nonnegative, --drop at least 0 and below 1")
    # Done
    return args


def report(name, ok, note=""):
    """Print one PASS or FAIL line and return ok."""
    # Line
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + note if note else ''}", flush=True)
    # Pass through
    return bool(ok)


def graph_identity(path):
    """The graph file's name and the first 16 hex digits of the SHA-256 of its bytes, as main_gnn.py records it."""
    # Read the bytes
    with open(path, "rb") as f:
        # Digest
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    # Name and digest
    return f"{os.path.basename(path)}:{digest}"


class Tally:
    """
    The largest difference over a set of cases that must show no change.
    A case with a value that is not a finite number is kept by name and
    fails the tally: max(0.0, nan) is 0.0 in Python, so a NaN must never
    reach a running maximum.
    """

    def __init__(self):
        # Largest finite difference, number of cases, names of the cases with a nonfinite value
        self.worst, self.n, self.bad = 0.0, 0, []

    def add(self, a, b, case):
        """Add one case: tensors a and b must be finite; their largest absolute difference is kept."""
        # One more case
        self.n += 1
        # The differences, which can overflow even when both sides are finite
        d = (a - b).abs()
        # A nonfinite value on either side, or a nonfinite difference, fails the case by name
        if not (bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and bool(torch.isfinite(d).all())):
            # Keep the name
            self.bad.append(case)
            # Nothing to compare
            return
        # Largest finite difference so far
        self.worst = max(self.worst, float(d.max()))

    def ok(self, tol):
        """True when at least one case was covered, none was nonfinite, and no difference exceeds tol."""
        # All three conditions
        return self.n > 0 and not self.bad and self.worst <= tol

    def note(self):
        """One line: the largest difference, the number of cases, and the nonfinite cases."""
        # No case at all is said plainly, since it fails a mandatory test
        if self.n == 0:
            # Say so
            return "no case was covered"
        # Difference and count, then up to three nonfinite cases by name
        return f"largest change {self.worst:.2e} over {self.n} cases" + (f"; NONFINITE values in {len(self.bad)} cases, first: {self.bad[:3]}" if self.bad else "")


class Response:
    """How often changing a connected meter with usable history moved the hidden meter's forecast."""

    def __init__(self):
        # Cases tried, cases that moved, smallest and largest change, cases with a nonfinite value
        self.tried, self.moved, self.low, self.high, self.bad = 0, 0, float("inf"), 0.0, []

    def add(self, a, b, tol, case):
        """Add one batch of cases: rows of a and b are cases; a row moved when its largest change exceeds tol."""
        # The differences, which can overflow even when both sides are finite
        diff = (a - b).abs()
        # A nonfinite value on either side, or a nonfinite difference, fails by name and counts as no response
        if not (bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and bool(torch.isfinite(diff).all())):
            # Keep the name
            self.bad.append(case)
            # Nothing to compare
            return
        # Largest change of every row
        d = diff.reshape(a.shape[0], -1).max(dim=1).values
        # Counts
        self.tried, self.moved = self.tried + len(d), self.moved + int((d > tol).sum())
        # Range
        self.low, self.high = min(self.low, float(d.min())), max(self.high, float(d.max()))

    def note(self):
        """One line: how many cases moved, the range of the changes, and the nonfinite cases by name."""
        # The nonfinite cases, when there are any
        bad = f"; NONFINITE values in {len(self.bad)} cases, first: {self.bad[:3]}" if self.bad else ""
        # Nothing tried
        if self.tried == 0:
            # Say so
            return "no connected meter with usable history to change" + bad
        # Counts and range
        return f"moved in {self.moved} of {self.tried} cases, change from {self.low:.2e} to {self.high:.2e}" + bad


def load_run(args):
    """Options from the checkpoint, the manifest, the timeline, the scale, and the model with its best weights on the CPU, after the refusals and the graph identity check."""
    # Checkpoint
    body = torch.load(os.path.join(args.run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
    # Options
    options = body["options"]
    # Must be a masked run
    if not options.get("masked"):
        # Raise
        raise ValueError("this run was not trained with --masked")
    # A fixed-graph run has its own check
    if options.get("graph_selection", "fixed") == "fixed":
        # Raise
        raise ValueError("this check handles masked runs on a time-indexed graph only; use checks/main_maskcheck.py for a fixed graph")
    # The graph file must be the one the run used, else the rebuilt edge sets would not be the run's
    if options.get("graph_id") != graph_identity(args.graph):
        # Raise
        raise ValueError(f"graph identity mismatch: run has {options.get('graph_id')}, {args.graph} is {graph_identity(args.graph)}")
    # Graph file
    file_body = sm.load_graph(args.graph)
    # It must be a time-indexed file
    if sm.graph_kind(file_body) != "timeline":
        # Raise
        raise ValueError(f"{args.graph} is not a time-indexed graph file")
    # Say so
    print(f"PASS  0. graph file matches the run: {options['graph_id']}", flush=True)
    # Edge sets and the period table, validated
    timeline = sm.load_timeline(file_body)
    # Manifest
    with open(os.path.join(args.features_dir, "manifest.json")) as f:
        # Parse
        manifest = json.load(f)
    # Model as the run built it: same control, weights, sizes, input features, sender gating and rule for hidden meters
    model = sm.make_model(options["kind"], manifest, timeline["graphs"][0], edges=options["edges"], edge_weights=options["edge_weights"], hidden=options["hidden"], embed=options["embed"], layers=options["layers"], dropout=options["dropout"], heads=options.get("heads", 4), seed=options["seed"], gnn_layers=options["gnn_layers"], predict_delta=options["predict_delta"], masked=True, maskable_meters=options["maskable_meters"], timeline=timeline, n_features=int(options.get("n_features", sd.F)), sender_gate=bool(options.get("sender_gate", False)), gate_max_gap=options["max_gap"], mask_policy=options.get("mask_policy", "legacy"))
    # Best weights
    model.load_state_dict(torch.load(os.path.join(args.run_dir, "checkpoint_best.pt"), map_location="cpu"))
    # Evaluation mode
    model.eval()
    # The run's scale, for the real windows
    scale = sd.load_scale(os.path.join(args.run_dir, "scale.json"), manifest)
    # Done
    return options, manifest, timeline, scale, model


def synthetic(B, H, M, C, seed, drop=0.0, extra=0):
    """
    A made-up input window (B, H, M, C * (F + extra)): magnitudes near 1,
    unit (cos, sin), validity flags on, then `extra` added features per
    channel with random values. With drop > 0, that share of the readings
    is marked absent: both flags off and the values zero, as an absent
    reading is stored.
    """
    # Generator
    g = torch.Generator().manual_seed(seed)
    # Magnitude around the scaled level
    mag = 1.0 + 0.1 * torch.randn(B, H, M, C, generator=g)
    # Angles
    theta = 2 * torch.pi * torch.rand(B, H, M, C, generator=g)
    # Which readings are present
    present = (torch.rand(B, H, M, C, generator=g) >= drop).to(mag.dtype)
    # The five standard features: mag, cos, sin, mag_valid, ang_valid, with absent readings zeroed
    parts = [mag * present, torch.cos(theta) * present, torch.sin(theta) * present, present, present]
    # The added features, random values
    parts += [torch.randn(B, H, M, C, generator=g) for _ in range(extra)]
    # Flatten channels and features
    return torch.stack(parts, dim=-1).reshape(B, H, M, C * (sd.F + extra))


def usable(x, C, max_gap, what):
    """Stop unless every meter's history in x is usable by the history rule, so that no test depends on a meter falling silent."""
    # The rule of the pipeline
    if not bool(sg.history_available(x, C, max_gap).all()):
        # Stop: the tests below assume it
        raise RuntimeError(f"{what}: a meter's history is not usable; lower --drop or change --seed")


def reach_sets(model, k, M, hops):
    """
    Meter index -> sorted list of the other meters whose history may reach
    its forecast under saved edge set k: meters on the same node, or on a
    node at most `hops` saved edges away. One message-passing layer moves
    information across one edge, so hops is the run's number of layers.
    """
    # Node per meter
    node_of = model.meter_node.tolist()
    # Saved edges of this set, both directions already stored
    src, dst = model.edge_set(k)[0].tolist()
    # Nodes that send a message into each node
    senders = {}
    # Loop over edges
    for a, b in zip(src, dst):
        # a sends to b
        senders.setdefault(b, set()).add(a)
    # Result
    out = {}
    # Loop over meters
    for i in range(M):
        # Nodes found so far: the meter's own node
        seen = {node_of[i]}
        # Nodes added in the last round
        frontier = {node_of[i]}
        # One round per message-passing layer
        for _ in range(hops):
            # Nodes that send into the last round's nodes and are new
            frontier = {v for u in frontier for v in senders.get(u, ())} - seen
            # Keep them
            seen |= frontier
        # Meters on those nodes, not i
        out[i] = sorted(j for j in range(M) if j != i and node_of[j] in seen)
    # Done
    return out


def hide(B, M, i):
    """A mask (B, M) that hides meter i in every window."""
    # Nothing hidden
    mask = torch.zeros(B, M, dtype=torch.bool)
    # Hide meter i
    mask[:, i] = True
    # Done
    return mask


def changed(x, other, meters):
    """A copy of x in which the listed meters' whole histories, validity flags included, are taken from other."""
    # Copy
    out = x.clone()
    # Replace the listed meters
    out[:, :, meters] = other[:, :, meters]
    # Done
    return out


def real_pieces(args, options, manifest, timeline, scale):
    """
    The datasets of the chosen real blocks, one per month piece, built as
    training and scoring build them for this run: time-indexed, with the
    unfiltered target validity the rule for hidden meters needs. Yields
    (label, dataset).
    """
    # Splits
    splits = sd.load_splits(args.splits)
    # Each chosen block
    for index in args.blocks:
        # The block
        block = splits[args.split][index]
        # Each month file it touches
        for month in sd.months_in_block(block):
            # The part of the month inside the block, scaled
            piece = sd.load_piece(args.features_dir, block, month, scale, allow_missing=True, manifest=manifest)
            # Nothing there
            if piece is None:
                # Next
                continue
            # Slot times, inputs, validity
            t, X, mag_valid, ang_valid = piece
            # The dataset, with min_meters 1 as scoring uses
            ds = sm.GraphWindowDataset(X, mag_valid, ang_valid, options["history"], options["horizon"], options["max_gap"], 1, t=t, timeline=timeline, raw_targets=True)
            # Hand it out with a name
            yield f"{args.split}[{index}] {month}", ds


def check_real(model, options, manifest, timeline, scale, reach, args):
    """
    R1, R2 and the neighbor response on real windows. Returns
    (ok, response, pairs tested, edge sets covered).
    """
    # Sizes
    M, C, chunk = len(manifest["meters"]), len(manifest["channels"]), options.get("mask_chunk", 7)
    # One tally per replacement through the plain forward pass, one for masked_forecast, one for their agreement
    forward = {"zeros": Tally(), "donor": Tally(), "random": Tally()}
    # Through the scoring routine, all three replacements together
    helper, agree = Tally(), Tally()
    # Neighbor response
    response = Response()
    # Pairs tested and edge sets seen
    n_pairs, sets_seen = 0, set()
    # Generator of the window sample
    rng = np.random.default_rng(args.seed)
    # Generator of the random replacement values
    g = torch.Generator().manual_seed(args.seed)
    # Each block piece
    for label, ds in real_pieces(args, options, manifest, timeline, scale):
        # A donor needs a second window
        if len(ds) < 2:
            # Next
            continue
        # Sampled windows, in time order
        idx = np.sort(rng.choice(len(ds), size=min(args.windows, len(ds)), replace=False))
        # The batch as tensors: inputs, targets, unfiltered target validity, [target slot, period]
        x, _, y_valid, extra = [torch.from_numpy(a) for a in ds.get_batch(idx)]
        # The edge set of each window, through the model's own table
        gid = model.graph_ids(extra[:, 1])
        # Eligibility, computed once from the untouched inputs and held fixed for every change below
        eligible, _ = model.mask_eligible(x, y_valid, gid)
        # Eligible (window, meter) pairs
        pairs = torch.nonzero(eligible)
        # No eligible pair in this piece
        if len(pairs) == 0:
            # Next
            continue
        # At most --pairs of them, drawn without replacement, in order
        pairs = pairs[torch.from_numpy(np.sort(rng.choice(len(pairs), size=min(args.pairs, len(pairs)), replace=False)))]
        # Count and edge sets
        n_pairs, sets_seen = n_pairs + len(pairs), sets_seen | set(gid[pairs[:, 0]].tolist())
        # Range of the real magnitudes, for the random replacement
        mags = x.reshape(x.shape[0], x.shape[1], M, C, sd.F)[..., 0]
        # Smallest and largest
        lo, hi = float(mags.min()), float(mags.max())
        # Rows per forward pass
        for start in range(0, len(pairs), args.rows):
            # Window and meter of each row
            b, m = pairs[start:start + args.rows, 0], pairs[start:start + args.rows, 1]
            # Row numbers
            r = torch.arange(len(b))
            # Another sampled window for each row, the donor
            donor = (b + 1) % len(idx)
            # The rows' windows, their edge sets, and a mask hiding each row's meter
            xb, gb, mask = x[b], gid[b], torch.zeros(len(b), M, dtype=torch.bool)
            # Hide
            mask[r, m] = True
            # The masked forecast of each row's hidden meter
            base = model(xb, mask, graph_ids=gb)[r, m]
            # Random values in range: magnitude between the smallest and largest real one, a random direction, flags on
            theta = 2 * torch.pi * torch.rand(len(b), xb.shape[1], C, generator=g)
            # As the five features of every channel
            noise = torch.stack([lo + (hi - lo) * torch.rand(len(b), xb.shape[1], C, generator=g), torch.cos(theta), torch.sin(theta), torch.ones_like(theta), torch.ones_like(theta)], dim=-1).reshape(len(b), xb.shape[1], C * sd.F)
            # The three replacements of the hidden meter's whole history
            replacements = {"zeros": torch.zeros_like(xb[r, :, m]), "donor": x[donor, :, m], "random": noise.to(xb.dtype)}
            # Each replacement
            for name, value in replacements.items():
                # Copy
                xc = xb.clone()
                # Replace the hidden meter's history
                xc[r, :, m] = value
                # The forecast must not change
                forward[name].add(model(xc, mask, graph_ids=gb)[r, m], base, f"{label}, {name}, rows {start} on")
            # Neighbor response: for each row, the first connected meter that has usable history in that window
            available = sg.history_available(xb, C, options["max_gap"])
            # The chosen neighbor per row, or -1
            pick = torch.tensor([next((j for j in reach[int(gb[p])][int(m[p])] if bool(available[p, j])), -1) for p in range(len(b))])
            # Rows that have one
            has = pick >= 0
            # Only when some row has
            if bool(has.any()):
                # Copy
                xn = xb.clone()
                # That neighbor's history, from the donor window
                xn[r[has], :, pick[has]] = x[donor[has], :, pick[has]]
                # The hidden meter's forecast should change
                response.add(model(xn, mask, graph_ids=gb)[r, m][has], base[has], args.tol, f"{label}, neighbor, rows {start} on")
        # Through masked_forecast: one eligible meter per window, the first of its pairs
        first = {}
        # Loop over the pairs in order
        for bb, mm in pairs.tolist():
            # Keep the first meter of each window
            first.setdefault(bb, mm)
        # Windows and their meters as tensors
        wb, wm = torch.tensor(list(first.keys())), torch.tensor(list(first.values()))
        # Rows per pass, smaller since the routine repeats the batch for every meter of a chunk
        step = max(1, args.rows // chunk)
        # Loop
        for start in range(0, len(wb), step):
            # Windows and meters of this pass
            b, m = wb[start:start + step], wm[start:start + step]
            # Row numbers and donors
            r, donor = torch.arange(len(b)), (b + 1) % len(idx)
            # Inputs and edge sets
            xb, gb = x[b], gid[b]
            # The routine's forecast of each window's chosen meter
            base = sm.masked_forecast(model, xb, chunk, gb)[0][r, m]
            # A mask hiding the chosen meter
            mask = torch.zeros(len(b), M, dtype=torch.bool)
            # Hide
            mask[r, m] = True
            # R2: the routine agrees with the plain forward pass
            agree.add(base, model(xb, mask, graph_ids=gb)[r, m], f"{label}, agreement, windows {start} on")
            # The three replacements again; the random one reuses the real range with new draws
            theta = 2 * torch.pi * torch.rand(len(b), xb.shape[1], C, generator=g)
            # As the five features of every channel
            noise = torch.stack([lo + (hi - lo) * torch.rand(len(b), xb.shape[1], C, generator=g), torch.cos(theta), torch.sin(theta), torch.ones_like(theta), torch.ones_like(theta)], dim=-1).reshape(len(b), xb.shape[1], C * sd.F)
            # Each replacement
            for name, value in {"zeros": torch.zeros_like(xb[r, :, m]), "donor": x[donor, :, m], "random": noise.to(xb.dtype)}.items():
                # Copy
                xc = xb.clone()
                # Replace the chosen meter's history
                xc[r, :, m] = value
                # The routine's forecast of that meter must not change
                helper.add(sm.masked_forecast(model, xc, chunk, gb)[0][r, m], base, f"{label}, masked_forecast, {name}, windows {start} on")
    # What was covered
    print(f"real windows: {n_pairs} eligible (window, meter) pairs from {args.split} blocks {args.blocks}, edge sets covered {sorted(sets_seen)}", flush=True)
    # R1, plain forward pass, each replacement
    ok = all([report(f"R1. real windows, hidden history replaced by {name}, plain forward pass", t.ok(args.tol), t.note()) for name, t in forward.items()])
    # R1, the scoring routine
    ok &= report("R1. real windows, the three replacements through masked_forecast", helper.ok(args.tol), helper.note())
    # R2
    ok &= report("R2. real windows, masked_forecast agrees with the plain forward pass", agree.ok(args.tol), agree.note())
    # Done
    return ok, response, n_pairs, sorted(sets_seen)


def check_synthetic(model, options, manifest, reach, extra, args):
    """S1 and S3 to S6 on made-up windows, and the neighbor response. Returns (ok, response)."""
    # Names
    meters = manifest["meters"]
    # Sizes
    M, C, H, B, G = len(meters), len(manifest["channels"]), options["history"], args.batch, model.n_graphs
    # Meters that may be hidden, as indices
    maskable = [meters.index(m) for m in options["maskable_meters"]]
    # The two kinds of made-up window: every reading present, and a share absent
    kinds = [("every reading present", 0.0), (f"{args.drop:.0%} of readings absent", args.drop)]
    # Tallies of S1 and S3, and the neighbor response
    own, outside_t, response = Tally(), Tally(), Response()
    # Loop over the two kinds of window
    for kind_index, (kind_name, drop) in enumerate(kinds):
        # Base windows
        x = synthetic(B, H, M, C, args.seed + 10 * kind_index, drop, extra)
        # A different draw to take replacement histories from
        other = synthetic(B, H, M, C, args.seed + 10 * kind_index + 1, drop, extra)
        # Every history must be usable in both, so that no meter falls silent
        usable(x, C, options["max_gap"], f"made-up windows, {kind_name}")
        # The replacement draw too
        usable(other, C, options["max_gap"], f"made-up replacement windows, {kind_name}")
        # Loop over edge sets
        for k in range(G):
            # Every window uses edge set k
            gid = torch.full((B,), k, dtype=torch.long)
            # Loop over the meters that may be hidden
            for i in maskable:
                # Hide meter i
                mask = hide(B, M, i)
                # Its forecast
                base = model(x, mask, graph_ids=gid)[:, i]
                # The case, for a failure message
                case = f"edge set {k}, {meters[i]}, {kind_name}"
                # S1. replace the hidden meter's own history
                own.add(model(changed(x, other, [i]), mask, graph_ids=gid)[:, i], base, case)
                # S3. replace every meter the saved edges do not connect to it, all at once
                outside = [j for j in range(M) if j != i and j not in reach[k][i]]
                # Only when there is such a meter
                if outside:
                    # Change
                    outside_t.add(model(changed(x, other, outside), mask, graph_ids=gid)[:, i], base, case)
                # Neighbor response: replace one connected meter, on the windows with every reading present only
                if kind_index == 0 and reach[k][i]:
                    # Change, with the windows pooled into one case
                    response.add(model(changed(x, other, reach[k][i][:1]), mask, graph_ids=gid)[:, i].reshape(1, -1), base.reshape(1, -1), args.tol, case)
    # S1
    ok = report("S1. made-up windows, a hidden meter's own history does not reach its forecast", own.ok(args.tol), own.note())
    # S3: mandatory whenever some meter is unconnected to another, which holds on any graph that is not complete
    if outside_t.n:
        # Report
        ok &= report("S3. meters the saved edges do not connect to the hidden meter do not reach its forecast", outside_t.ok(args.tol), outside_t.note())
    else:
        # Nothing to test, said plainly
        report("S3. unconnected-meter test", True, "not applicable, every meter is connected to every other under the saved edges")

    # S4. two windows in one batch, each with its own edge set
    x = synthetic(1, H, M, C, args.seed + 100, 0.0, extra).repeat(2, 1, 1, 1)
    # Replacement histories
    other = synthetic(1, H, M, C, args.seed + 101, 0.0, extra).repeat(2, 1, 1, 1)
    # The window that must not change, and the response of the window that may
    apart, apart_response = Tally(), Response()
    # Ordered pairs of edge sets
    for k in range(G):
        # Second set
        for k2 in range(G):
            # A hidden meter i and a meter j that is connected to i under set k and not under set k2
            found = next(((i, j) for i in maskable for j in reach[k][i] if j not in reach[k2][i]), None)
            # No such pair of meters for these two sets
            if k == k2 or found is None:
                # Next
                continue
            # Names
            i, j = found
            # Window 0 uses set k, window 1 uses set k2
            gid = torch.tensor([k, k2], dtype=torch.long)
            # Hide meter i in both
            mask = hide(2, M, i)
            # Before and after meter j's history is replaced
            before, after = model(x, mask, graph_ids=gid)[:, i], model(changed(x, other, [j]), mask, graph_ids=gid)[:, i]
            # The window on set k2 must not change
            apart.add(after[1], before[1], f"edge sets {k} and {k2}, hidden {meters[i]}, changed {meters[j]}")
            # The window on set k may
            apart_response.add(after[0:1], before[0:1], args.tol, f"edge sets {k} and {k2}")
    # S4: mandatory whenever two saved edge sets differ in this way
    if apart.n:
        # A nonfinite forecast in the window on the connecting set fails the test too, and its note names the cases
        ok &= report("S4. in a mixed batch each window follows its own edge set", apart.ok(args.tol) and not apart_response.bad, apart.note() + f" (ordered pairs of edge sets); the window on the connecting set {apart_response.note()}")
    else:
        # Nothing to test, said plainly
        report("S4. edge-set separation", True, "not applicable, no meter is connected under one saved edge set and not under another")

    # S5. one window per edge set in one batch, each with its own content
    x = synthetic(G, H, M, C, args.seed + 200, 0.0, extra)
    # Window b uses edge set b
    gid = torch.arange(G, dtype=torch.long)
    # Hide the first meter that may be hidden
    mask = hide(G, M, maskable[0])
    # The batch
    together = model(x, mask, graph_ids=gid)
    # Each window alone
    alone = torch.cat([model(x[b:b + 1], mask[b:b + 1], graph_ids=gid[b:b + 1]) for b in range(G)], dim=0)
    # Tally
    mixed = Tally()
    # One case
    mixed.add(together, alone, "batch against windows alone")
    # Report
    ok &= report("S5a. a batch mixing every edge set matches each window run alone", mixed.ok(args.tol), mixed.note())
    # Only with more than one window
    if G > 1:
        # Replace the whole of window 0
        x5 = x.clone()
        # New content
        x5[0] = synthetic(1, H, M, C, args.seed + 201, 0.0, extra)[0]
        # Tally
        cross = Tally()
        # Every other window must be unchanged
        cross.add(model(x5, mask, graph_ids=gid)[1:], together[1:], "window 0 changed")
        # Report
        ok &= report("S5b. changing one window changes no other window's forecast", cross.ok(args.tol), cross.note())

    # S6. the routine used for evaluation and scoring, on the batch mixing every edge set
    y_all, keep = sm.masked_forecast(model, x, options.get("mask_chunk", 7), graph_ids=gid)
    # Names flagged invalid
    invalid = [meters[m] for m in range(M) if not bool(keep[m])]
    # Expected: every meter that may never be hidden
    expected = [m for m in meters if m not in options["maskable_meters"]]
    # Report
    ok &= report("S6a. meters that may never be hidden are flagged invalid", invalid == expected, f"invalid: {invalid}")
    # Tally
    passes = Tally()
    # One plain pass per meter
    for m in maskable:
        # Compare
        passes.add(model(x, hide(G, M, m), graph_ids=gid)[:, m], y_all[:, m], f"{meters[m]}")
    # Report
    ok &= report("S6b. masked_forecast matches one plain pass per meter", passes.ok(args.tol), passes.note())
    # Without the edge set of each window it must refuse
    try:
        # Call without graph ids
        sm.masked_forecast(model, x, options.get("mask_chunk", 7))
        # Reached only when it did not refuse
        refused = False
    except ValueError:
        # Refused
        refused = True
    # Report; a model with one edge set has nothing to refuse
    ok &= report("S6c. masked_forecast refuses to run without the edge set of each window", refused or G == 1)
    # Done
    return ok, response


def check_added(model, options, manifest, timeline, extra, args):
    """
    A1. The added input features change nothing in a masked forward.
    With extra > 0 the trained model has added features and is tested
    itself. With extra == 0 it has none, so a separate model is built with
    --extra-features on the same graph and settings; its weights are
    freshly drawn, which is enough, since the stripping happens before any
    weight is used. Returns ok.
    """
    # Names
    meters = manifest["meters"]
    # Sizes
    M, C, H, G = len(meters), len(manifest["channels"]), options["history"], model.n_graphs
    # The trained model when it has added features
    if extra > 0:
        # Itself
        subject, n_extra, label = model, extra, "the trained run, which has added features"
    else:
        # The separate model's weights are drawn at random, so the draw is seeded and the test repeats exactly
        torch.manual_seed(args.seed)
        # A separate model with added features: same graph, control, sizes, sender gating and rule for hidden meters
        subject = sm.make_model(options["kind"], manifest, timeline["graphs"][0], edges=options["edges"], edge_weights=options["edge_weights"], hidden=options["hidden"], embed=options["embed"], layers=options["layers"], dropout=options["dropout"], heads=options.get("heads", 4), seed=options["seed"], gnn_layers=options["gnn_layers"], predict_delta=options["predict_delta"], masked=True, maskable_meters=options["maskable_meters"], timeline=timeline, n_features=sd.F + args.extra_features, sender_gate=bool(options.get("sender_gate", False)), gate_max_gap=options["max_gap"], mask_policy=options.get("mask_policy", "legacy")).eval()
        # Its label says what it is
        n_extra, label = args.extra_features, f"a SEPARATE freshly built model with {args.extra_features} added features per channel; not applicable to the trained run, which has none"
    # One window per edge set
    x = synthetic(G, H, M, C, args.seed + 300, 0.0, n_extra)
    # The same windows with different values in the added features only
    x2 = x.reshape(G, H, M, C, sd.F + n_extra).clone()
    # New random values there
    x2[..., sd.F:] = torch.randn(G, H, M, C, n_extra, generator=torch.Generator().manual_seed(args.seed + 301))
    # Back to the flat layout
    x2 = x2.reshape(x.shape)
    # Window b uses edge set b
    gid = torch.arange(G, dtype=torch.long)
    # The first meter that may be hidden, and a visible meter
    i = meters.index(options["maskable_meters"][0])
    # Any other meter
    j = next(m for m in range(M) if m != i)
    # Hide meter i
    mask = hide(G, M, i)
    # Masked forward before and after the added features change
    before, after = subject(x, mask, graph_ids=gid), subject(x2, mask, graph_ids=gid)
    # Invariance: the whole output, every meter
    same = Tally()
    # One case
    same.add(after, before, "added features changed")
    # Control 1: without a mask the same change moves the output, so the features are real inputs
    live = Response()
    # One case
    live.add(subject(x2, graph_ids=gid).reshape(1, -1), subject(x, graph_ids=gid).reshape(1, -1), args.tol, "no mask")
    # Control 2: in the masked forward, a visible meter's own readings move its own forecast, so the output is not constant
    moves = Response()
    # One case
    moves.add(subject(changed(x, synthetic(G, H, M, C, args.seed + 302, 0.0, n_extra), [j]), mask, graph_ids=gid)[:, j].reshape(1, -1), before[:, j].reshape(1, -1), args.tol, "visible meter")
    # Both controls must respond and neither may be nonfinite
    controls = live.moved == 1 and moves.moved == 1 and not live.bad and not moves.bad
    # Report
    return report(f"A1. added input features change nothing in a masked forward, on {label}", same.ok(args.tol) and controls, f"{same.note()}; control without a mask {live.note()}; control on a visible meter {moves.note()}")


def timing(model, options, manifest, timeline, scale, args):
    """
    Seconds per forward call of masked_forecast and per training step on
    real batches of the run's batch size, the forward calls per window,
    and the largest GPU memory this process allocated and reserved for
    each. Training steps run on a copy of the model that is never saved.
    """
    # Device
    device = torch.device(args.time_device)
    # Whether GPU memory can be read
    cuda = device.type == "cuda"
    # Sizes
    chunk, size = options.get("mask_chunk", 7), options["batch_size"]
    # The first real block piece that can fill the batches
    ds = next((d for _, d in real_pieces(args, options, manifest, timeline, scale) if len(d) >= size), None)
    # None found
    if ds is None:
        # Say so and stop the timing only
        print("TIMING not run: no block piece holds a full batch", flush=True)
        # Done
        return
    # Windows of the batches, drawn without replacement
    idx = np.random.default_rng(args.seed).choice(len(ds), size=min(len(ds), size * args.time_batches) // size * size, replace=False)
    # The batches as tensors on the device, as the training loop makes them
    batches = [tuple(torch.from_numpy(a).to(device) for a in ds.get_batch(np.sort(idx[s:s + size]))) for s in range(0, len(idx), size)]
    # A copy on the device, so that the model of the tests is untouched
    subject = copy.deepcopy(model).to(device)

    def clock():
        """The time now, after the device has finished its queued work."""
        # Wait for the GPU
        if cuda:
            # Finish queued work
            torch.cuda.synchronize(device)
        # Now
        return time.time()

    def peak():
        """The largest memory this process allocated and reserved on the device since the last reset, in MiB."""
        # Only on a GPU
        return f"allocated {torch.cuda.max_memory_allocated(device) / 2**20:.0f} MiB, reserved {torch.cuda.max_memory_reserved(device) / 2**20:.0f} MiB" if cuda else "not a GPU"

    # Forward calls of the routine per batch: one per chunk of meters that may be hidden
    calls = math.ceil(int(subject.maskable.sum()) / chunk)
    # Evaluation mode, no gradient
    subject.eval()
    # No gradient
    with torch.no_grad():
        # One call first, so that start-up work is not timed
        sm.masked_forecast(subject, batches[0][0], chunk, subject.graph_ids(batches[0][3][:, 1]))
        # Fresh memory count
        if cuda:
            # Reset
            torch.cuda.reset_peak_memory_stats(device)
        # Start
        start = clock()
        # Each batch
        for x, _, _, extra in batches:
            # The routine, over each window's own edge set
            sm.masked_forecast(subject, x, chunk, subject.graph_ids(extra[:, 1]))
        # Seconds
        seconds = clock() - start
    # Report
    print(f"TIMING masked_forecast on {device}: {len(batches)} batches of {size} windows; {calls} forward calls per batch, each on {size} x {chunk} rows at most; {seconds / (len(batches) * calls):.4f} s per forward call, {seconds / len(batches):.4f} s per batch, {1000 * seconds / (len(batches) * size):.3f} ms per window; peak memory of this process for these calls: {peak()}", flush=True)
    # Training steps as the training loop runs them, with the run's optimizer settings
    optimizer = torch.optim.Adam(subject.parameters(), lr=options["lr"], weight_decay=options.get("weight_decay", 0.0))
    # One step first, so that start-up work is not timed
    sm.run_batches(subject, batches[:1], True, options["angle_weight"], sm.new_sums(), optimizer)
    # Fresh memory count
    if cuda:
        # Reset
        torch.cuda.reset_peak_memory_stats(device)
    # Start
    start = clock()
    # The steps
    sm.run_batches(subject, batches, True, options["angle_weight"], sm.new_sums(), optimizer)
    # Seconds
    seconds = clock() - start
    # Report
    print(f"TIMING training step on {device}: {len(batches)} steps of {size} windows, one forward and one backward call each; {seconds / len(batches):.4f} s per step, {1000 * seconds / (len(batches) * size):.3f} ms per window; peak memory of this process for these steps: {peak()}; the stepped copy of the model is discarded", flush=True)


def main():
    """Run the checks and give the two verdicts."""
    # Options
    args = parse_args()
    # Load
    options, manifest, timeline, scale, model = load_run(args)
    # Sizes
    M, G = len(manifest["meters"]), model.n_graphs
    # Added features per channel in the trained run
    extra = int(options.get("n_features", sd.F)) - sd.F
    # Whether the model was trained with connections; without them no neighbor can respond
    wired = options["edges"] != "none"
    # Reach under every saved edge set
    reach = [reach_sets(model, k, M, options["gnn_layers"]) for k in range(G)]
    # Report what is checked
    print(f"edges={options['edges']}, sender_gate={bool(options.get('sender_gate', False))}, mask_policy={options.get('mask_policy', 'legacy')}, n_features={sd.F + extra}: {G} edge sets, {len(options['maskable_meters'])} meters that may be hidden, {options['gnn_layers']} message-passing layer(s), tolerance {args.tol:g}", flush=True)
    # Whether a required part could not be run
    incomplete = False
    # No gradient
    with torch.no_grad():
        # Real windows need the five standard features, which are the ones the feature files hold
        if extra == 0:
            # R1, R2 and the neighbor response on real windows
            ok, real_response, _, _ = check_real(model, options, manifest, timeline, scale, reach, args)
        else:
            # A run with added features needs its added-feature files, which this check does not load
            print("NOT RUN  R1, R2: real windows of a run with added input features need its added-feature files", flush=True)
            # Mark it
            ok, real_response, incomplete = True, Response(), True
        # Made-up windows
        ok_s, synthetic_response = check_synthetic(model, options, manifest, reach, extra, args)
        # Added features
        ok_a = check_added(model, options, manifest, timeline, extra, args)
    # Leakage invariance over every part
    ok = ok and ok_s and ok_a
    # Neighbor response: nonfinite values there fail like any other
    ok = ok and not real_response.bad and not synthetic_response.bad
    # The two response lines
    print(f"N. neighbor response on real windows: {real_response.note()}", flush=True)
    # Made-up windows
    print(f"N. neighbor response on made-up windows (edge set, meter): {synthetic_response.note()}", flush=True)
    # A wired model must respond on the made-up windows, and on the real ones when they were run
    responds = synthetic_response.moved > 0 and (real_response.moved > 0 or incomplete)
    # Timing, when asked for
    if args.time_batches > 0:
        # Measure
        timing(model, options, manifest, timeline, scale, args)
    # First verdict
    print("leakage invariance: PASSED" if ok else "leakage invariance: FAILED", flush=True)
    # Second verdict
    print("neighbor response: NOT APPLICABLE, the model was trained with no connections" if not wired else ("neighbor response: RESPONDS" if responds else "neighbor response: UNRESOLVED, no connected meter moved the forecast; this is not a leak, and it has to be investigated before the result is used"), flush=True)
    # Exit code: a leak first, then an unresolved response, then an incomplete run
    sys.exit(1 if not ok else (2 if wired and not responds else (3 if incomplete else 0)))


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
