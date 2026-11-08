# =============================================================================
# File:        main_neighborcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-07
#
# Purpose:     Does a trained graph model use its neighbors? For each
#              trained run given, on windows from the design block
#              (validation block 0, January 2024), hold a target meter's
#              own history, the graph, the meter identities, and the target
#              readings fixed, change only the histories of the meters
#              adjacent to it in the graph the window uses, and measure how
#              far the target meter's forecast moves and whether its error
#              gets better or worse. Two perturbations:
#                donor   the neighbors' complete histories (magnitude, cos,
#                        sin, both validity flags) are replaced together by
#                        the same neighbors' histories from one donor
#                        window of the same topology period whose history
#                        does not overlap the window's; repeated over
#                        several donor draws with a fixed seed, and the
#                        spread across draws is reported
#                zero    the neighbors' histories are set to zero with their
#                        validity flags off, the "neighbors missing" case,
#                        reported separately as a stress test
#              Measured per target meter and overall: the median absolute
#              change of the magnitude forecast in scaled units and in
#              physical units (volts for L channels, amps for S channels),
#              the median circular change of the angle forecast in degrees,
#              the change relative to the model's own median magnitude
#              residual on the same windows (floored at 1e-4 scaled units),
#              and the magnitude and angle error before and after, with the
#              share of window-channels whose error got worse.
#              Only windows that are actually perturbed count: a window
#              enters a meter's measurements when the meter has a neighbor
#              under that window's graph and a donor exists in every draw;
#              the reference residual and the --min-windows rule use the
#              same windows. Angle comparisons need both forecast vectors
#              to have a norm of at least --norm-floor; the number excluded
#              is reported. The per-meter spread of the donor medians across
#              draws is printed for every meter.
#              Controls: an unperturbed second forward must reproduce the
#              forecast exactly; in a run trained with --edges none,
#              replacing (and separately zeroing) every other meter's
#              history must leave the target forecast unchanged within
#              tolerance, over a reported nonzero number of perturbed
#              windows. Meters with no neighbor in any sampled window's
#              graph (egauge_24, and every meter in a no-edge run) are
#              listed and skipped; meters with too few eligible windows are
#              listed too. The diagnostic is restricted to a block that
#              lies in one feature month, so its windows are built exactly
#              as scoring builds them; a longer block raises.
#              For a time-indexed run the graph of each window is the one
#              its period names, selected from the window's actual last
#              history slot by the dataset, and donors come from the same
#              period.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root. Needs torch
#              and torch_geometric, the run folders (checkpoint.pt,
#              checkpoint_best.pt, scale.json), the feature files, and the
#              graph file each run was trained with.
#
# Usage:
#   python checks/main_neighborcheck.py --run-dir RUN [RUN ...] --graph graph_meters.json
#   python checks/main_neighborcheck.py --run-dir /content/drive/MyDrive/socal_runs/gcn_dyn_aw001 /content/drive/MyDrive/socal_runs/gcn_dyn_none --graph graph_meters_periods.json --features-dir /content/features
#
# Functions:
#   parse_args()                    command-line options with defaults
#   report(name, ok, note)          print one PASS or FAIL line, return ok
#   graph_identity(path)            file name and content hash, as main_gcn.py records it
#   load_run(run_dir, graph_path, manifest, device)
#                                   options, model with best weights, scale, timeline or None
#   design_windows(args, options, scale, manifest, timeline)
#                                   dataset over the design block, slot times, period per window
#   neighbor_sets(model, n_meters)  per edge set, the list of neighbor meters of every meter
#   pick_donors(ds, idx, rng, span) one non-overlapping donor window per sampled window, same period
#   circular_change(a, b, floor)    wrapped angle difference in degrees between two (cos, sin) vectors, and which are usable
#   perturb_and_measure(...)        forecasts and errors under one perturbation, for one target meter, on the windows given
#   summarize(rows)                 medians and shares over window-channels
#   main()                          run the check for every run given
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: hashing for the graph identity
import hashlib
# Standard library: JSON
import json
# Standard library: paths and exit
import os
import sys
# Third party: numerics
import numpy as np
# Third party: deep learning
import torch

# The repo root, one level above checks/, so the project modules import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project: data side and models
import socal_modeldata as sd
import socal_models as sm


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Does a trained graph model use its neighbors? Perturb neighbor histories and measure the target forecast")
    # Runs and their graph
    parser.add_argument("--run-dir", nargs="+", required=True, help="Run folders to test; all must have been trained with --graph")
    parser.add_argument("--graph", default="graph_meters.json", help="The graph file the runs were trained with, fixed or time-indexed")
    # Data
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with the monthly feature files and manifest.json")
    parser.add_argument("--splits", default="splits_v2.json", help="Splits file; the design block is its validation block 0")
    parser.add_argument("--block", type=int, default=0, help="Validation block index to draw windows from (0 is the design block)")
    # Sampling
    parser.add_argument("--windows", type=int, default=200, help="Windows sampled from the block")
    parser.add_argument("--draws", type=int, default=5, help="Donor draws per perturbation")
    parser.add_argument("--min-windows", type=int, default=20, help="A meter with fewer usable windows is listed and skipped")
    parser.add_argument("--seed", type=int, default=0, help="Diagnostic seed for the window sample and the donors")
    # Reference floor, angle norm floor, and tolerance
    parser.add_argument("--ref-floor", type=float, default=1e-4, help="Floor for the per-meter median residual used as the relative reference, scaled units")
    parser.add_argument("--norm-floor", type=float, default=1e-3, help="A (cos, sin) forecast with a smaller norm has no usable direction and is excluded from angle comparisons")
    parser.add_argument("--atol", type=float, default=1e-5, help="Tolerance for the two controls")
    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="torch device")
    # Parse
    args = parser.parse_args()
    # Basic checks
    if args.windows < 1 or args.draws < 1 or args.min_windows < 1:
        # Refuse
        parser.error("--windows, --draws, and --min-windows must be at least 1")
    # Floors and tolerance
    if not (np.isfinite(args.ref_floor) and args.ref_floor > 0 and np.isfinite(args.norm_floor) and args.norm_floor > 0 and np.isfinite(args.atol) and args.atol >= 0):
        # Refuse
        parser.error("--ref-floor and --norm-floor must be finite and positive, --atol finite and nonnegative")
    # Done
    return args


def report(name, ok, note=""):
    """Print one PASS or FAIL line and return ok."""
    # Line
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + note if note else ''}")
    # Pass through
    return ok


def graph_identity(path):
    """File name and the first 16 hex digits of the content hash, as main_gcn.py records it."""
    # Read
    with open(path, "rb") as f:
        # Hash
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    # Done
    return f"{os.path.basename(path)}:{digest}"


def load_run(run_dir, graph_path, manifest, device):
    """Options from checkpoint.pt, the model with its best weights in evaluation mode, the scale, and the timeline (None for a fixed graph), after the graph identity check."""
    # Options
    body = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
    # Saved options
    options = body["options"]
    # The graph must be the one the run used
    if options.get("graph_id") != graph_identity(graph_path):
        # Raise
        raise ValueError(f"{run_dir}: graph identity mismatch, run has {options.get('graph_id')}, {graph_path} is {graph_identity(graph_path)}")
    # Masked runs are out of scope here
    if options.get("masked"):
        # Raise
        raise ValueError(f"{run_dir}: this check is for ordinary forecasters, not masked runs")
    # Graph file
    gbody = sm.load_graph(graph_path)
    # Kind
    timeline = sm.load_timeline(gbody) if sm.graph_kind(gbody) == "timeline" else None
    # Graph for the fixed path
    graph = gbody if timeline is None else None
    # Model
    model = sm.make_model(options["kind"], manifest, graph, edges=options["edges"], edge_weights=options["edge_weights"], hidden=options["hidden"], embed=options["embed"], layers=options["layers"], dropout=options["dropout"], seed=options["seed"], gnn_layers=options["gnn_layers"], predict_delta=options["predict_delta"], timeline=timeline)
    # Best weights
    model.load_state_dict(torch.load(os.path.join(run_dir, "checkpoint_best.pt"), map_location="cpu"))
    # Evaluation mode, on the device
    model = model.to(device).eval()
    # Scale
    scale = sd.load_scale(os.path.join(run_dir, "scale.json"), manifest)
    # Done
    return options, model, scale, timeline


def design_windows(args, options, scale, manifest, timeline):
    """The synchronized dataset over the chosen validation block, built as scoring builds it, with its slot times."""
    # Splits
    splits = sd.load_splits(args.splits)
    # Block
    block = splits["validation"][args.block]
    # Months it touches
    months = sd.months_in_block(block)
    # Pieces
    pieces = [sd.load_piece(args.features_dir, block, m, scale, manifest=manifest) for m in months]
    # Drop missing
    pieces = [p for p in pieces if p is not None]
    # Need exactly one, so windows are built as scoring builds them, inside one month piece
    if not pieces:
        # Raise
        raise ValueError("no feature data for the chosen block")
    # More than one month
    if len(pieces) > 1:
        # Raise
        raise ValueError(f"the block spans {len(pieces)} feature months; this diagnostic is restricted to a block inside one month")
    # The piece
    t, X, mv, av = pieces[0]
    # Dataset as scoring builds it (min_meters 1, as score_split uses), time-indexed when the run is
    ds = sm.GraphWindowDataset(X, mv, av, options["history"], options["horizon"], options["max_gap"], 1, t=t if timeline is not None else None, timeline=timeline)
    # Done
    return ds, block


def neighbor_sets(model, n_meters):
    """For every edge set k, a list over meters of the sorted meter indices adjacent to that meter."""
    # Node of each meter
    meter_node = model.meter_node.cpu().numpy()
    # Meters on each node
    on_node = {}
    # Loop
    for m, n in enumerate(meter_node):
        # Add
        on_node.setdefault(int(n), []).append(m)
    # Per edge set
    out = []
    # Loop
    for k in range(model.n_graphs):
        # Edges
        ei, _ = model.edge_set(k)
        # As numpy
        ei = ei.cpu().numpy()
        # Adjacent nodes per node
        adj = {}
        # Loop over edges, both directions
        for a, b in zip(ei[0], ei[1]):
            # Add
            adj.setdefault(int(a), set()).add(int(b))
            # Reverse
            adj.setdefault(int(b), set()).add(int(a))
        # Neighbor meters per meter: meters on adjacent nodes, plus meters sharing the node
        sets = []
        # Loop
        for m in range(n_meters):
            # Own node
            n = int(meter_node[m])
            # Nodes reached
            nodes = adj.get(n, set()) | {n}
            # Meters there, not m itself
            sets.append(sorted(mm for nn in nodes for mm in on_node.get(nn, []) if mm != m))
        # Keep
        out.append(sets)
    # Done
    return out


def pick_donors(ds, idx, rng, span):
    """One donor window per sampled window: a window of the same topology period whose history does not overlap, chosen at random. Returns donor indices, or -1 where none exists."""
    # Starts of every window
    starts = ds.starts
    # Period of every window (zeros for a fixed graph)
    period = ds.period if ds.period is not None else np.zeros(len(starts), dtype=np.int64)
    # Output
    donors = np.full(len(idx), -1, dtype=np.int64)
    # Loop
    for j, i in enumerate(idx):
        # Candidates: same period, non-overlapping
        cand = np.nonzero((period == period[i]) & (np.abs(starts - starts[i]) >= span))[0]
        # Any
        if len(cand):
            # Pick
            donors[j] = cand[rng.integers(len(cand))]
    # Done
    return donors


def circular_change(a, b, floor):
    """
    Wrapped angle difference in degrees (0 to 180) between the (cos, sin)
    vectors a and b of shape (..., 2), from atan2 of each, and a bool of
    the same shape that is True where both norms reach floor, so the
    directions are usable. Entries that are not usable hold NaN.
    """
    # Angles
    ta, tb = torch.atan2(a[..., 1], a[..., 0]), torch.atan2(b[..., 1], b[..., 0])
    # Wrapped difference
    d = torch.remainder(ta - tb + torch.pi, 2 * torch.pi) - torch.pi
    # Degrees, absolute
    deg = torch.rad2deg(d.abs())
    # Usable
    ok = (a.norm(dim=-1) >= floor) & (b.norm(dim=-1) >= floor)
    # NaN where not
    return torch.where(ok, deg, torch.full_like(deg, float("nan"))), ok


def perturb_and_measure(model, x, graph_ids, y, y_valid, y0, m, neigh_mask, window_mask, replacement, scale_m, is_current, norm_floor):
    """
    Forecast for target meter m after replacing the histories of the
    meters flagged in neigh_mask (W, M) with replacement (W, H, M, C*F),
    everything else fixed. Only the windows flagged in window_mask (W,)
    enter the measurements. Returns a dict of flat arrays over the
    window-channels of meter m with a valid magnitude target in those
    windows: magnitude change (scaled, physical), angle change (degrees,
    NaN where a forecast direction is unusable), magnitude error before
    and after, angle error before and after (NaN where the angle target
    is invalid or a direction is unusable), the physical unit flags, and
    the count of angle comparisons excluded for an unusable direction.
    """
    # Perturbed input
    xp = torch.where(neigh_mask[:, None, :, None], replacement, x)
    # Forward
    with torch.no_grad():
        # Forecast
        y1 = model(xp, graph_ids=graph_ids)
    # Target meter slices (W, C, 3)
    a, b, yt = y0[:, m], y1[:, m], y[:, m]
    # Valid targets (W, C), restricted to the windows measured
    vm = y_valid[:, m, :, 0] & window_mask[:, None]
    # Angle target validity
    va = y_valid[:, m, :, 1]
    # Magnitude change, scaled
    d_mag = (a[..., 0] - b[..., 0]).abs()
    # Physical
    d_phys = d_mag * scale_m[None, :]
    # Angle change between the two forecasts
    d_ang, ok_change = circular_change(a[..., 1:3], b[..., 1:3], norm_floor)
    # Errors before and after
    e0_mag, e1_mag = (a[..., 0] - yt[..., 0]).abs(), (b[..., 0] - yt[..., 0]).abs()
    # Angle errors, usable only where the forecast and the target directions both exist
    e0_ang, ok0 = circular_change(a[..., 1:3], yt[..., 1:3], norm_floor)
    # After
    e1_ang, ok1 = circular_change(b[..., 1:3], yt[..., 1:3], norm_floor)
    # Angle error rows need a valid angle target and both forecasts usable
    ang_ok = va & ok0 & ok1
    # NaN elsewhere
    e0_ang = torch.where(ang_ok, e0_ang, torch.full_like(e0_ang, float("nan")))
    # After
    e1_ang = torch.where(ang_ok, e1_ang, torch.full_like(e1_ang, float("nan")))
    # Keep
    keep = vm
    # Current flag per window-channel
    cur = is_current[None, :].expand_as(keep)
    # Excluded angle comparisons among kept rows: a valid angle target whose forecasts were unusable
    excluded = int((va & ~(ok0 & ok1) & keep).sum()) + int((~ok_change & keep).sum())
    # Flatten
    out = {
        "d_mag": d_mag[keep].cpu().numpy(), "d_phys": d_phys[keep].cpu().numpy(), "d_ang": d_ang[keep].cpu().numpy(),
        "e0_mag": e0_mag[keep].cpu().numpy(), "e1_mag": e1_mag[keep].cpu().numpy(),
        "e0_ang": e0_ang[keep].cpu().numpy(), "e1_ang": e1_ang[keep].cpu().numpy(),
        "current": cur[keep].cpu().numpy(), "excluded": np.array([excluded]),
    }
    # Done
    return out


def summarize(rows):
    """Medians and shares over the concatenated window-channels of a list of perturb_and_measure outputs."""
    # Concatenate each key
    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    # Nothing
    if len(cat["d_mag"]) == 0:
        # Empty
        return None
    # Angle error rows that are usable
    ang = ~np.isnan(cat["e0_ang"])
    # Angle change rows that are usable
    chg = ~np.isnan(cat["d_ang"])
    # Summary
    return {
        "n": int(len(cat["d_mag"])), "n_ang": int(ang.sum()), "excluded": int(cat["excluded"].sum()),
        "d_mag": float(np.median(cat["d_mag"])),
        "d_volts": float(np.median(cat["d_phys"][~cat["current"]])) if (~cat["current"]).any() else float("nan"),
        "d_amps": float(np.median(cat["d_phys"][cat["current"]])) if cat["current"].any() else float("nan"),
        "d_ang": float(np.median(cat["d_ang"][chg])) if chg.any() else float("nan"),
        "e0_mag": float(np.median(cat["e0_mag"])), "e1_mag": float(np.median(cat["e1_mag"])),
        "worse_mag": float(np.mean(cat["e1_mag"] > cat["e0_mag"])),
        "e0_ang": float(np.median(cat["e0_ang"][ang])) if ang.any() else float("nan"), "e1_ang": float(np.median(cat["e1_ang"][ang])) if ang.any() else float("nan"),
        "worse_ang": float(np.mean(cat["e1_ang"][ang] > cat["e0_ang"][ang])) if ang.any() else float("nan"),
    }


def main():
    """Run the check for every run given."""
    # Options
    args = parse_args()
    # Device
    device = torch.device(args.device)
    # Manifest
    manifest = sd.load_manifest(args.features_dir)
    # Names
    meters, channels = manifest["meters"], manifest["channels"]
    # Sizes
    M, C = len(meters), len(channels)
    # Current channels are the S channels
    is_current = torch.tensor([ch.startswith("S") for ch in channels], device=device)
    # Every run
    all_ok = True
    # Loop
    for run_dir in args.run_dir:
        # Header
        print(f"\n===== {os.path.basename(run_dir.rstrip('/'))} =====")
        # Load
        options, model, scale, timeline = load_run(run_dir, args.graph, manifest, device)
        # Say
        print(f"edges={options['edges']}, graph_selection={options.get('graph_selection', 'fixed')}, {model.n_graphs} edge set(s), history {options['history']}, horizon {options['horizon']}")
        # Windows
        ds, block = design_windows(args, options, scale, manifest, timeline)
        # Say
        print(f"design block {block['start']} to {block['end']}: {len(ds)} usable windows")
        # Sample
        rng = np.random.default_rng(args.seed)
        # Indices
        idx = np.sort(rng.choice(len(ds), size=min(args.windows, len(ds)), replace=False))
        # Window span in slots
        span = options["history"] + options["horizon"]
        # Batch
        xb, yb, vb, extra = ds.get_batch(idx)
        # To tensors
        x = torch.as_tensor(xb, device=device)
        # Targets
        y = torch.as_tensor(yb, device=device)
        # Validity
        y_valid = torch.as_tensor(vb, device=device)
        # Graph ids
        graph_ids = model.graph_ids(torch.as_tensor(extra[:, 1], device=device)) if extra.ndim == 2 else None
        # Graph id per window as numpy (zeros for fixed)
        gid = graph_ids.cpu().numpy() if graph_ids is not None else np.zeros(len(idx), dtype=np.int64)
        # Base forecast
        with torch.no_grad():
            # Forecast
            y0 = model(x, graph_ids=graph_ids)
            # Again, for the reproduction control
            y0b = model(x, graph_ids=graph_ids)
        # Control 1: reproduction
        all_ok &= report("control 1: an unperturbed second forward reproduces the forecast", float((y0 - y0b).abs().max()) <= args.atol, f"max abs diff {float((y0 - y0b).abs().max()):.2e}")
        # Neighbor sets per edge set
        nsets = neighbor_sets(model, M)
        # Donors per draw
        donor_lists = [pick_donors(ds, idx, np.random.default_rng(args.seed + 1 + d), span) for d in range(args.draws)]
        # Windows without a donor in any draw
        no_donor = int(np.any(np.stack(donor_lists) < 0, axis=0).sum())
        # Say
        print(f"{len(idx)} windows sampled, {args.draws} donor draws, windows without a same-period non-overlapping donor: {no_donor}")
        # Donor inputs per draw
        donor_x = []
        # Loop
        for donors in donor_lists:
            # Valid donors
            ok = donors >= 0
            # Batch of donors (invalid ones take their own window, then are dropped by the mask below)
            xd, _, _, _ = ds.get_batch(np.where(ok, donors, idx))
            # Tensor
            donor_x.append((torch.as_tensor(xd, device=device), torch.as_tensor(ok, device=device)))
        # Donor availability in every draw, per window
        donor_all = torch.as_tensor(np.all(np.stack(donor_lists) >= 0, axis=0), device=device)
        # Scale as a tensor
        scale_t = torch.as_tensor(scale, device=device)
        # Column guide
        print("\ncolumns: nbrs = neighbors under the sampled windows' graphs (min-max); win = eligible windows (neighbor present, donor in every draw, valid target);")
        # More
        print("  dmag = median |forecast change| scaled; dV, dA = the same in volts (L channels) and amps (S channels); rel = dmag over the meter's own median residual (ref);")
        # More
        print("  dang = median angle change in degrees; emag 0>1 = median magnitude error before > after, worse = share of window-channels with a larger error after;")
        # More
        print("  eang 0>1 and worse likewise for the angle (usable directions only); n_ang = usable angle comparisons, excl = excluded for an unusable direction")
        # Skipped
        isolated, thin = [], []
        # Overall collections
        all_donor, all_zero = [], []
        # Per meter
        for m in range(M):
            # Neighbor mask per window under its graph
            neigh = np.zeros((len(idx), M), dtype=bool)
            # Degrees seen
            degrees = set()
            # Loop
            for w in range(len(idx)):
                # Neighbors under this window's graph
                nb = nsets[gid[w]][m]
                # Set
                neigh[w, nb] = True
                # Degree
                degrees.add(len(nb))
            # Meters with no neighbor anywhere
            if not neigh.any():
                # Record
                isolated.append(meters[m])
                # Next
                continue
            # Mask tensor
            nm = torch.as_tensor(neigh, device=device)
            # Eligible windows: a neighbor present under the window's graph, a donor in every draw, and a valid magnitude target for m
            eligible = nm.any(dim=1) & donor_all & y_valid[:, m, :, 0].any(dim=1)
            # Count
            n_el = int(eligible.sum())
            # Too few
            if n_el < args.min_windows:
                # Record
                thin.append(f"{meters[m]} ({n_el})")
                # Next
                continue
            # Reference: the model's own median magnitude residual on the eligible windows, floored
            r = (y0[:, m, :, 0] - y[:, m, :, 0]).abs()[y_valid[:, m, :, 0] & eligible[:, None]]
            # Value
            ref = max(float(r.median()), args.ref_floor)
            # Donor draws
            rows, per_draw = [], []
            # Loop
            for xd, ok in donor_x:
                # Measure on the eligible windows
                r_ = perturb_and_measure(model, x, graph_ids, y, y_valid, y0, m, nm, eligible, xd, scale_t[m], is_current, args.norm_floor)
                # Keep
                rows.append(r_)
                # Median of this draw
                per_draw.append(float(np.median(r_["d_mag"])) if len(r_["d_mag"]) else float("nan"))
            # Summary over draws
            sdn = summarize(rows)
            # Zero perturbation on the same windows
            sz = summarize([perturb_and_measure(model, x, graph_ids, y, y_valid, y0, m, nm, eligible, torch.zeros_like(x), scale_t[m], is_current, args.norm_floor)])
            # Collect
            all_donor.append(sdn); all_zero.append(sz)
            # Degree text
            deg = f"{min(degrees)}" if len(degrees) == 1 else f"{min(degrees)}-{max(degrees)}"
            # Header line for the meter
            print(f"\n{meters[m]} nbrs {deg}, win {n_el}, ref {ref:.5f}, n_ang {sdn['n_ang']}, excl {sdn['excluded']} (donor) / {sz['excluded']} (zero)")
            # Donor line
            print(f"  donor  dmag {sdn['d_mag']:.5f}  dV {sdn['d_volts']:.3f}  dA {sdn['d_amps']:.3f}  rel {sdn['d_mag'] / ref:.2f}  dang {sdn['d_ang']:.2f}  emag {sdn['e0_mag']:.5f}>{sdn['e1_mag']:.5f} worse {sdn['worse_mag']:.2f}  eang {sdn['e0_ang']:.2f}>{sdn['e1_ang']:.2f} worse {sdn['worse_ang']:.2f}  draws dmag {min(per_draw):.5f}-{max(per_draw):.5f}")
            # Zero line
            print(f"  zero   dmag {sz['d_mag']:.5f}  dV {sz['d_volts']:.3f}  dA {sz['d_amps']:.3f}  rel {sz['d_mag'] / ref:.2f}  dang {sz['d_ang']:.2f}  emag {sz['e0_mag']:.5f}>{sz['e1_mag']:.5f} worse {sz['worse_mag']:.2f}  eang {sz['e0_ang']:.2f}>{sz['e1_ang']:.2f} worse {sz['worse_ang']:.2f}")
        # Skipped meters
        print(f"\nno neighbor in any sampled window's graph (skipped): {isolated if isolated else 'none'}")
        # Thin
        print(f"fewer than {args.min_windows} eligible windows (skipped): {thin if thin else 'none'}")
        # Overall
        if all_donor:
            # Pool size
            n_tot = sum(s_["n"] for s_ in all_donor)
            # Median of per-meter medians
            print(f"\noverall, median over {len(all_donor)} meters: donor dmag {np.median([s_['d_mag'] for s_ in all_donor]):.5f} scaled, dang {np.nanmedian([s_['d_ang'] for s_ in all_donor]):.2f} deg, worse magnitude share {np.median([s_['worse_mag'] for s_ in all_donor]):.2f}, worse angle share {np.nanmedian([s_['worse_ang'] for s_ in all_donor]):.2f}; zero dmag {np.median([s_['d_mag'] for s_ in all_zero]):.5f}, dang {np.nanmedian([s_['d_ang'] for s_ in all_zero]):.2f} deg, worse magnitude share {np.median([s_['worse_mag'] for s_ in all_zero]):.2f}; {n_tot} window-channels")
        # Control 2: in a no-edge run, replacing or zeroing every other meter leaves the target unchanged
        if options["edges"] == "none":
            # Largest change and windows tested
            worst, tested = 0.0, 0
            # Donor 0
            xd, ok = donor_x[0]
            # Loop
            for m in range(M):
                # Everyone else
                others = torch.ones(len(idx), M, dtype=torch.bool, device=device)
                # Not m
                others[:, m] = False
                # Replacement by donor, where a donor exists
                xp = torch.where((others & ok[:, None])[:, None, :, None], xd, x)
                # Zeroing
                xz = torch.where(others[:, None, :, None], torch.zeros_like(x), x)
                # Forward both
                with torch.no_grad():
                    # Donor
                    y1 = model(xp, graph_ids=graph_ids)
                    # Zero
                    y2 = model(xz, graph_ids=graph_ids)
                # Change on m
                worst = max(worst, float((y1[:, m] - y0[:, m]).abs().max()), float((y2[:, m] - y0[:, m]).abs().max()))
                # Windows perturbed: donor replacements plus zeroings
                tested += int(ok.sum()) + len(idx)
            # Report
            all_ok &= report("control 2: in the no-edge run, replacing or zeroing every other meter leaves each target forecast unchanged", worst <= args.atol and tested > 0, f"max abs change {worst:.2e} over {tested} perturbed windows")
        else:
            # Not applicable
            print("SKIP  control 2 (applies to a run trained with --edges none)")
    # Exit
    sys.exit(0 if all_ok else 1)


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
