# =============================================================================
# File:        main_neighboravail.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Read-only count of how often a meter's neighbors have no
#              usable history in the windows the models actually see, by
#              meter, topology period and split block. The reason: a
#              neighbor with no usable history still sits on its node and
#              its edges during message passing, so it sends a learned
#              "absent" state and counts in the degree normalization. How
#              often that happens decides whether a message-passing change
#              for such neighbors (candidate A) can matter, and it decides
#              which meters and windows are eligible for the masked
#              forecasting objective (a hidden meter needs at least one
#              neighbor with usable history).
#
#              Windows are produced the way the pipeline produces them:
#              split by split, block by block, one piece per (block, month)
#              as socal_modeldata.iter_blocks does, so no window crosses a
#              block or a month boundary, and only the months the blocks
#              touch are read. Three things are kept apart per window and
#              meter, because they are different questions:
#                sender_ok     history-only availability: the history rule
#                              (no run of max_gap absent slots inside the
#                              history) with no condition on the target;
#                              this is what a neighbor needs to send a
#                              message, and it never looks at the future;
#                target_valid  the meter's own target slot has at least one
#                              valid magnitude (and, for the angle mode, at
#                              least one valid angle), per mode;
#                usable        sender_ok and the magnitude target valid:
#                              the pipeline's per-meter usability, which
#                              decides the retained windows (at least
#                              min_meters usable meters) and the loss terms.
#              Eligibility for masked evaluation is: a valid target in the
#              mode, degree at least 1 in the window's edge set, and at
#              least one neighbor with sender_ok. It does not require the
#              hidden meter's own history to be usable, since that history
#              is hidden anyway. Reported beside it: the pipeline's own
#              restriction, which keeps a window only when at least
#              min_meters meters are usable, so eligible pairs in windows
#              the pipeline drops are counted separately. Channel-level
#              counts of eligible targets (per channel, magnitude and angle)
#              go to a second CSV, since thresholds are per channel. Meter
#              names are matched between the manifest and the graph file
#              before any array is indexed. With --compare-dataset and torch
#              present, the retained starts and usability flags are compared
#              with socal_models.GraphWindowDataset on every piece: retained
#              starts, usability, periods, the raw channel-level target masks
#              and the masks the dataset returns per item (raw masks under
#              the usability restriction). A requested comparison that fails,
#              compares zero pieces, or meets a block or month with no data
#              (missing coverage) exits nonzero before any CSV is written;
#              the output reports pieces counted, pieces compared and missing
#              coverage.
#
#              Output: one CSV with a row per (split block, topology period,
#              meter), written to --out-dir, and printed summaries: per
#              block, the share of retained windows with at least one
#              unavailable neighbor anywhere, the per-receiver exposure
#              (receiver usable and at least one neighbor unavailable), the
#              number of affected directed messages, and the masked
#              eligibility shares. --self-test runs the same code on small
#              synthetic fixtures that put a block boundary inside a month,
#              a month boundary inside a block, and a topology change inside
#              a block, and checks the selection against hand counts. It
#              infers nothing about the wiring and adds no edge.
#
# Usage (from the repo root):
#   python checks/main_neighboravail.py --self-test
#   python checks/main_neighboravail.py --features-dir /content/features --compare-dataset --out-dir /content/drive/MyDrive/socal_runs/diagnostics --out neighbor_availability_2026-11-12.csv
#
# Functions:
#   parse_args()                          command-line options with defaults
#   load_timeline(path)                   graphs, periods and period-to-graph table from the graph file
#   check_alignment(manifest, timeline)   meter names of the manifest and the graph nodes must agree
#   neighbor_sets(timeline, M)            per graph, the neighbor index arrays of each meter
#   window_flags(piece, history, horizon, max_gap, min_meters)
#                                         retained starts and the three per-meter flags for one piece
#   count_piece(label, piece, args, timeline, nbs, rows, totals)
#                                         accumulate one piece's counts
#   write_csv(path, rows, meters)         the per (block, period, meter) table
#   write_channel_csv(path, crow, meters, channels)
#                                         the per (block, period, meter, channel) eligible-target counts
#   print_summaries(rows, totals, meters, blocks)
#                                         the printed tables
#   run(args, manifest, timeline, pieces, blocks, comparator)
#                                         the whole count over an iterator of pieces, with status
#   verify_and_write(args, manifest, rows, totals, crow, status)
#                                         enforce the verification outcome, then write the CSVs
#   self_test()                           synthetic fixtures with hand-checked expectations
#   main()                                entry point
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: CSV output
import csv
# Standard library: JSON
import json
# Standard library: paths
import os
# Standard library: import path and exit codes
import sys
# Standard library: fixture folders
import tempfile
# Third party: numerics
import numpy as np
# Third party: timestamps
import pandas as pd

# The repo root, one level above checks/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Import from the root
sys.path.insert(0, ROOT)

# Project: month loading, block slicing, the window rule
import socal_modeldata as sd
# Project: period lookup
import socal_metergraph as smg

# The splits in pipeline order
SPLITS = ("train", "validation", "test_events", "test_normal")


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Usable histories and usable neighbors by meter, topology period and split block, with the pipeline's own window selection")
    # Features
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with <YYYY-MM>.npz and manifest.json")
    # Graph
    parser.add_argument("--graph", default="graph_meters_periods.json", help="Time-indexed meter graph")
    # Splits
    parser.add_argument("--splits", default="splits_v2.json", help="Splits file with the block bounds")
    # Which splits
    parser.add_argument("--split-names", nargs="+", default=list(SPLITS), help="Splits to count")
    # Window rule, the training defaults
    parser.add_argument("--history", type=int, default=60, help="History slots")
    # Horizon
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in slots")
    # Gap
    parser.add_argument("--max-gap", type=int, default=12, help="A run of this many absent slots inside a history makes the meter unavailable in that window")
    # Minimum meters
    parser.add_argument("--min-meters", type=int, default=1, help="Usable meters a window needs to be retained")
    # Output folder, separate from the run folders and the repo
    parser.add_argument("--out-dir", default=os.path.join("data", "diagnostics"), help="Folder for the CSV (created if needed)")
    # Output name
    parser.add_argument("--out", default="neighbor_availability.csv", help="CSV file name inside --out-dir")
    # Self-test
    parser.add_argument("--self-test", action="store_true", help="Run the synthetic fixture checks and exit")
    # Dataset comparison
    parser.add_argument("--compare-dataset", action="store_true", help="Also compare retained starts and usability with socal_models.GraphWindowDataset on every piece (needs torch)")
    # Parse
    return parser.parse_args()


def load_timeline(path):
    """The time-indexed graph file as {"nodes", "graphs", "periods", "period_graph"}, read directly so this check needs no torch."""
    # Read
    with open(path) as f:
        # Parse
        body = json.load(f)
    # Must be the time-indexed layout
    if "graphs" not in body or "periods" not in body:
        # Stop
        raise SystemExit(f"{path} is not a time-indexed graph file (needs graphs and periods)")
    # Period to graph
    period_graph = np.array([int(p["graph"]) for p in body["periods"]], dtype=np.int64)
    # Done
    return {"nodes": body["nodes"], "graphs": body["graphs"], "periods": body["periods"], "period_graph": period_graph}


def check_alignment(manifest, timeline):
    """The manifest's meter names and the graph nodes' meter names must be the same list in the same order, since edges index nodes and arrays index manifest meters."""
    # Names
    a = list(manifest["meters"])
    # Node names
    b = [n["meter"] for n in timeline["nodes"]]
    # Must agree
    if a != b:
        # Stop
        raise SystemExit(f"meter order differs between the manifest {a} and the graph nodes {b}; arrays cannot be indexed with graph indices")
    # Say
    print(f"alignment: {len(a)} meters, manifest order equals graph node order")


def neighbor_sets(timeline, M):
    """Per graph index, a list of M integer arrays: the neighbor meter indices of each meter in that edge set (both directions)."""
    # Per graph
    out = []
    # Loop
    for g in timeline["graphs"]:
        # Lists
        nb = [[] for _ in range(M)]
        # Edges, both directions
        for e in g["edges"]:
            # Add
            nb[e["source"]].append(e["target"])
            # Reverse
            nb[e["target"]].append(e["source"])
        # Arrays
        out.append([np.array(sorted(set(x)), dtype=np.int64) for x in nb])
    # Done
    return out


def window_flags(piece, history, horizon, max_gap, min_meters):
    """
    For one piece (one block, one month): the retained window starts, and per
    window and meter the flags sender_ok (history-only rule), target_mag
    (valid magnitude at the target), target_ang (valid angle at the target)
    and usable (sender_ok and target_mag, the pipeline's rule), for every
    candidate window, plus the retained mask. Returns (all_starts, retained,
    sender_ok, target_mag, target_ang, usable, tmag_ch, tang_ch) where the
    last two are per-channel target validity (W, M, C).
    """
    # Validity arrays
    mag_valid, ang_valid = piece["mag_valid"], piece["ang_valid"]
    # Sizes
    T, M = mag_valid.shape[0], mag_valid.shape[1]
    # Candidate starts
    n_starts = max(T - history - horizon + 1, 0)
    # Flags
    sender_ok = np.zeros((n_starts, M), dtype=bool)
    # Targets
    target_mag = np.zeros((n_starts, M), dtype=bool)
    # Angle targets
    target_ang = np.zeros((n_starts, M), dtype=bool)
    # Target index of each start
    target = np.arange(n_starts) + history + horizon - 1
    # Per meter
    for mi in range(M):
        # Presence
        present_any = mag_valid[:, mi, :].any(axis=1)
        # History-only rule: the target condition is switched off
        starts = sd.window_starts(present_any, history, horizon, max_gap, np.ones(T, dtype=bool))
        # Flag
        sender_ok[starts, mi] = True
        # Target validity per mode
        target_mag[:, mi] = present_any[target]
        # Angle
        target_ang[:, mi] = ang_valid[:, mi, :].any(axis=1)[target]
    # The pipeline's usability
    usable = sender_ok & target_mag
    # Retained windows
    retained = usable.sum(axis=1) >= min_meters
    # Per-channel target validity
    tmag_ch = mag_valid[target]
    # Angle
    tang_ch = ang_valid[target]
    # Done
    return np.arange(n_starts), retained, sender_ok, target_mag, target_ang, usable, tmag_ch, tang_ch


def count_piece(label, piece, args, timeline, nbs, rows, totals, crow=None):
    """Accumulate one piece's counts into rows (per block, period, meter), totals (per block) and, when crow is given, per-channel eligible-target counts."""
    # Flags for every candidate window
    all_starts, retained, sender_ok_all, target_mag_all, target_ang_all, usable_all, tmag_ch, tang_ch = window_flags(piece, args.history, args.horizon, args.max_gap, args.min_meters)
    # Nothing to count
    if len(all_starts) == 0:
        # Done
        return 0
    # Period of every candidate window from its last history slot, the pipeline's rule
    period_all = smg.period_index_at(timeline["periods"], piece["t"][all_starts + args.history - 1])
    # Sizes
    M = usable_all.shape[1]
    # Block totals
    tot = totals.setdefault(label, {"windows": 0, "windows_any_unavailable": 0, "receiver_exposure": 0, "receiver_usable": 0, "messages": 0, "messages_unavailable": 0, "eligible_mag": 0, "eligible_ang": 0, "eligible_mag_dropped": 0, "eligible_ang_dropped": 0, "usable_deg1": 0})
    # Eligible pairs in windows the pipeline drops (not retained): counted per block, by mode
    for p in np.unique(period_all[~retained]):
        # Windows of the period among the dropped
        sel = (period_all == p) & ~retained
        # Graph
        g = int(timeline["period_graph"][p])
        # Each meter
        for mi in range(M):
            # Neighbors
            nb = nbs[g][mi]
            # None
            if not len(nb):
                # Next
                continue
            # Available senders
            n_ok = sender_ok_all[sel][:, nb].sum(axis=1)
            # Eligible: valid target and an available sender
            tot["eligible_mag_dropped"] += int((target_mag_all[sel][:, mi] & (n_ok > 0)).sum())
            # Angle
            tot["eligible_ang_dropped"] += int((target_ang_all[sel][:, mi] & (n_ok > 0)).sum())
    # From here on, retained windows only
    kept = all_starts[retained]
    # Flags of the retained windows
    sender_ok, target_mag, target_ang, usable = sender_ok_all[retained], target_mag_all[retained], target_ang_all[retained], usable_all[retained]
    # Channel flags
    tmag_ch, tang_ch = tmag_ch[retained], tang_ch[retained]
    # Periods
    period = period_all[retained]
    # Nothing retained
    if len(kept) == 0:
        # Done
        return 0
    # Per period in this piece
    for p in np.unique(period):
        # Windows of the period
        sel = period == p
        # Graph
        g = int(timeline["period_graph"][p])
        # Per-window flag: any directed message in the window comes from an unavailable sender
        any_unavailable = np.zeros(int(sel.sum()), dtype=bool)
        # Per meter as receiver
        for mi in range(M):
            # Neighbors
            nb = nbs[g][mi]
            # Senders available among them
            n_ok = sender_ok[sel][:, nb].sum(axis=1) if len(nb) else np.zeros(int(sel.sum()), dtype=np.int64)
            # Unavailable
            n_bad = len(nb) - n_ok
            # Own flags
            own = usable[sel][:, mi]
            # Angle target too
            own_ang = own & target_ang[sel][:, mi]
            # Window-level flag
            any_unavailable |= n_bad > 0
            # Row
            row = rows.setdefault((label, int(p), mi), {"graph": g, "degree": int(len(nb)), "windows": 0, "sender_ok": 0, "target_mag": 0, "target_ang": 0, "usable": 0, "usable_ang": 0, "usable_with_sender": 0, "usable_with_unavailable": 0, "eligible_mag": 0, "eligible_ang": 0, "sum_ok": 0, "sum_bad": 0})
            # Counts
            row["windows"] += int(sel.sum())
            # Target validity by mode
            row["target_mag"] += int(target_mag[sel][:, mi].sum())
            # Angle
            row["target_ang"] += int(target_ang[sel][:, mi].sum())
            # History-only availability of this meter
            row["sender_ok"] += int(sender_ok[sel][:, mi].sum())
            # Usable
            row["usable"] += int(own.sum())
            # Usable with angle target
            row["usable_ang"] += int(own_ang.sum())
            # Usable with at least one available sender
            row["usable_with_sender"] += int((own & (n_ok > 0)).sum())
            # Usable with at least one unavailable neighbor
            row["usable_with_unavailable"] += int((own & (n_bad > 0)).sum())
            # Eligible for masked evaluation, magnitude mode: valid magnitude target, degree at least 1, one available sender (the hidden meter's own history is not required)
            e_mag = int((target_mag[sel][:, mi] & (n_ok > 0)).sum()) if len(nb) else 0
            # Angle mode: valid angle target instead
            e_ang = int((target_ang[sel][:, mi] & (n_ok > 0)).sum()) if len(nb) else 0
            # Per-channel eligible targets, when asked
            if crow is not None and len(nb):
                # Each channel
                for ci in range(tmag_ch.shape[2]):
                    # Row
                    cr = crow.setdefault((label, int(p), mi, ci), {"eligible_mag": 0, "eligible_ang": 0})
                    # Magnitude
                    cr["eligible_mag"] += int((tmag_ch[sel][:, mi, ci] & (n_ok > 0)).sum())
                    # Angle
                    cr["eligible_ang"] += int((tang_ch[sel][:, mi, ci] & (n_ok > 0)).sum())
            # Row
            row["eligible_mag"] += e_mag
            # Angle
            row["eligible_ang"] += e_ang
            # Sums over usable windows
            row["sum_ok"] += int(n_ok[own].sum())
            # Unavailable
            row["sum_bad"] += int(n_bad[own].sum())
            # Block totals, receivers
            tot["receiver_usable"] += int(own.sum())
            # Exposure
            tot["receiver_exposure"] += int((own & (n_bad > 0)).sum())
            # Directed messages into this receiver over all windows (message passing runs for every node)
            tot["messages"] += int(len(nb)) * int(sel.sum())
            # Of which from unavailable senders
            tot["messages_unavailable"] += int(n_bad.sum())
            # Eligibility totals
            tot["eligible_mag"] += e_mag
            # Angle
            tot["eligible_ang"] += e_ang
            # Usable with degree at least 1
            tot["usable_deg1"] += int(own.sum()) if len(nb) else 0
        # Windows
        tot["windows"] += int(sel.sum())
        # Windows with any unavailable sender
        tot["windows_any_unavailable"] += int(any_unavailable.sum())
    # Done
    return len(kept)


def write_csv(path, rows, meters):
    """Write the per (block, period, meter) table."""
    # Open
    with open(path, "w", newline="") as f:
        # Writer
        w = csv.writer(f)
        # Header
        w.writerow(["block", "period", "graph", "meter", "degree", "windows", "sender_ok", "target_mag", "target_ang", "usable", "usable_ang", "usable_with_sender", "usable_with_unavailable", "eligible_mag", "eligible_ang", "mean_available_neighbors", "mean_unavailable_neighbors"])
        # Rows
        for (label, p, mi), r in sorted(rows.items()):
            # Means over usable windows
            mo = f"{r['sum_ok'] / r['usable']:.3f}" if r["usable"] else ""
            # Unavailable
            mb = f"{r['sum_bad'] / r['usable']:.3f}" if r["usable"] else ""
            # Row
            w.writerow([label, p, r["graph"], meters[mi], r["degree"], r["windows"], r["sender_ok"], r["target_mag"], r["target_ang"], r["usable"], r["usable_ang"], r["usable_with_sender"], r["usable_with_unavailable"], r["eligible_mag"], r["eligible_ang"], mo, mb])


def write_channel_csv(path, crow, meters, channels):
    """Write the per (block, period, meter, channel) eligible-target counts, magnitude and angle."""
    # Open
    with open(path, "w", newline="") as f:
        # Writer
        w = csv.writer(f)
        # Header
        w.writerow(["block", "period", "meter", "channel", "eligible_mag", "eligible_ang"])
        # Rows
        for (label, p, mi, ci), r in sorted(crow.items()):
            # Row
            w.writerow([label, p, meters[mi], channels[ci], r["eligible_mag"], r["eligible_ang"]])


def print_summaries(rows, totals, meters, blocks):
    """Print the per-block and per-meter summaries."""
    # Per block
    print(f"\n{'block':<44} {'windows':>8} {'any unavail':>11} {'share':>6} {'recv usable':>11} {'recv exposed':>12} {'share':>6} {'messages':>10} {'unavail msgs':>12} {'share':>6} {'elig mag':>9} {'elig ang':>9} {'elig dropped mag/ang':>21}")
    # Loop in split order
    for label in blocks:
        # Totals
        t = totals.get(label)
        # Absent block
        if t is None:
            # Line
            print(f"{label:<44} {'no retained window':>8}")
            # Next
            continue
        # Shares
        s1 = t["windows_any_unavailable"] / t["windows"] if t["windows"] else 0.0
        # Receivers
        s2 = t["receiver_exposure"] / t["receiver_usable"] if t["receiver_usable"] else 0.0
        # Messages
        s3 = t["messages_unavailable"] / t["messages"] if t["messages"] else 0.0
        # Line
        print(f"{label:<44} {t['windows']:>8} {t['windows_any_unavailable']:>11} {s1:>6.3f} {t['receiver_usable']:>11} {t['receiver_exposure']:>12} {s2:>6.3f} {t['messages']:>10} {t['messages_unavailable']:>12} {s3:>6.3f} {t['eligible_mag']:>9} {t['eligible_ang']:>9} {str(t['eligible_mag_dropped']) + '/' + str(t['eligible_ang_dropped']):>21}")
    # Per meter over everything counted
    print(f"\n{'meter':<10} {'sender_ok':>10} {'usable':>8} {'usable+sender':>13} {'share':>6} {'usable+unavail':>14} {'share':>6} {'mean avail nb':>13} {'mean unavail nb':>15}")
    # Loop
    for mi, m in enumerate(meters):
        # Rows
        rs = [r for (b, p, k), r in rows.items() if k == mi]
        # Sums
        so = sum(r["sender_ok"] for r in rs)
        # Usable
        u = sum(r["usable"] for r in rs)
        # With sender
        ws = sum(r["usable_with_sender"] for r in rs)
        # With unavailable
        wu = sum(r["usable_with_unavailable"] for r in rs)
        # Means
        mo = sum(r["sum_ok"] for r in rs) / u if u else 0.0
        # Unavailable
        mb = sum(r["sum_bad"] for r in rs) / u if u else 0.0
        # Line
        print(f"{m:<10} {so:>10} {u:>8} {ws:>13} {(ws / u if u else 0):>6.3f} {wu:>14} {(wu / u if u else 0):>6.3f} {mo:>13.2f} {mb:>15.2f}")


def compare_with_dataset(piece, args, timeline):
    """Compare this check's retained starts and usability with socal_models.GraphWindowDataset on one piece; returns (ok, note). Needs torch."""
    # Import here so the rest works without torch
    import socal_models as sm
    # Build the dataset exactly as training does, with the validity arrays as inputs
    T, M, C = piece["mag_valid"].shape
    # Dummy features of the right shape
    X = np.zeros((T, M, C, 1), dtype=np.float32)
    # Dataset
    ds = sm.GraphWindowDataset(X, piece["mag_valid"], piece["ang_valid"], args.history, args.horizon, args.max_gap, args.min_meters, t=piece["t"], timeline=timeline)
    # This check's flags
    all_starts, retained, sender_ok, target_mag, target_ang, usable, _, _ = window_flags(piece, args.history, args.horizon, args.max_gap, args.min_meters)
    # This check's per-channel target validity
    _, _, _, _, _, _, tmag_ch, tang_ch = window_flags(piece, args.history, args.horizon, args.max_gap, args.min_meters)
    # Same starts, same usability, same periods
    same_starts = np.array_equal(ds.starts, all_starts[retained])
    # Usability
    same_usable = np.array_equal(ds.usable, usable[retained])
    # Periods
    same_period = np.array_equal(np.asarray(ds.period), smg.period_index_at(timeline["periods"], piece["t"][all_starts[retained] + args.history - 1]))
    # Channel-level target validity: the dataset's raw masks at the target slots (y_valid[target], before the usability restriction)
    targets = ds.starts + args.history + args.horizon - 1
    # Raw masks (W, M, C, 2)
    raw = ds.y_valid[targets]
    # Magnitude and angle against this check's per-channel flags on the retained windows
    same_raw = np.array_equal(raw[..., 0], tmag_ch[retained]) and np.array_equal(raw[..., 1], tang_ch[retained])
    # The masks the dataset actually returns per item carry the usability restriction: y_valid[target] & usable; checked on a sample of items
    sample = range(0, len(ds), max(1, len(ds) // 50))
    # Compare each sampled item with raw & usable
    same_items = all(np.array_equal(ds[i][2], raw[i] & ds.usable[i][:, None, None]) for i in sample)
    # The restriction differs from raw validity wherever a meter has a valid target but an unusable history; count those for the report
    restricted = int((raw[..., 0] & ~ds.usable[:, :, None]).sum())
    # Done
    ok = same_starts and same_usable and same_period and same_raw and same_items
    # Note
    return ok, f"starts {same_starts}, usable {same_usable}, period {same_period}, raw channel targets {same_raw}, returned masks = raw & usable {same_items} ({len(sample)} items), channel targets removed by the usability restriction {restricted}"


def run(args, manifest, timeline, pieces, blocks, comparator=None):
    """
    The whole count over an iterator of (label, month, piece-or-None) and the
    block labels in order. Returns (rows, totals, crow, status) where status
    holds pieces counted, pieces compared, comparison failures and the list
    of (block, month) with missing coverage. comparator defaults to
    compare_with_dataset and can be replaced in tests.
    """
    # Alignment first
    check_alignment(manifest, timeline)
    # Neighbor sets
    nbs = neighbor_sets(timeline, len(manifest["meters"]))
    # Accumulators
    rows, totals, crow = {}, {}, {}
    # Status
    status = {"pieces": 0, "compared": 0, "failed": [], "missing": []}
    # Comparator
    compare = comparator or compare_with_dataset
    # Each piece
    for label, month, piece in pieces:
        # Missing coverage
        if piece is None:
            # Record
            status["missing"].append((label, month))
            # Say
            print(f"  {label} {month}: MISSING coverage (no feature file or no rows inside the block)")
            # Next
            continue
        # Count
        n = count_piece(label, piece, args, timeline, nbs, rows, totals, crow)
        # Pieces
        status["pieces"] += 1
        # Optional comparison with the model's dataset class
        note = ""
        # Asked
        if getattr(args, "compare_dataset", False):
            # Compare; an environment without torch cannot verify and is a failure, not a pass
            try:
                # Run
                same, detail = compare(piece, args, timeline)
            # Missing torch or torch_geometric
            except ImportError as e:
                # Failure
                same, detail = False, f"comparison impossible here: {e}"
            # Record
            status["compared"] += 1
            # Failure
            if not same:
                # Keep
                status["failed"].append((label, month))
            # Note
            note = f"; dataset comparison {'PASS' if same else 'FAIL'} ({detail})"
        # Say
        print(f"  {label} {month}: {n} retained windows{note}")
    # Report
    print(f"pieces counted: {status['pieces']}; pieces compared with the dataset: {status['compared']}; comparison failures: {len(status['failed'])}; missing coverage: {status['missing'] if status['missing'] else 'none'}")
    # Done
    return rows, totals, crow, status


def verify_and_write(args, manifest, rows, totals, crow, status):
    """Enforce the verification outcome, then write the CSVs. A requested comparison that failed, compared nothing, or met missing coverage exits nonzero before any CSV is written."""
    # Requested comparison
    if getattr(args, "compare_dataset", False):
        # Problems
        problems = []
        # Failures
        if status["failed"]:
            # Record
            problems.append(f"dataset comparison FAILED on {len(status['failed'])} piece(s): {status['failed']}")
        # Nothing compared
        if status["compared"] == 0:
            # Record
            problems.append("dataset comparison requested but zero pieces were compared")
        # Missing coverage makes the verification incomplete
        if status["missing"]:
            # Record
            problems.append(f"missing coverage in {len(status['missing'])} (block, month) pair(s): {status['missing']}")
        # Stop before writing
        if problems:
            # Each
            for item in problems:
                # Print
                print(f"FAIL  {item}")
            # Exit
            raise SystemExit("verification failed; no CSV written")
        # Say
        print(f"PASS  dataset comparison on {status['compared']} piece(s), no missing coverage")
    # Without a requested comparison, missing coverage is reported as a warning
    elif status["missing"]:
        # Say
        print(f"WARNING  missing coverage in {len(status['missing'])} (block, month) pair(s); the counts above exclude them")
    # Output folder
    os.makedirs(args.out_dir, exist_ok=True)
    # Path
    path = os.path.join(args.out_dir, args.out)
    # Write
    write_csv(path, rows, list(manifest["meters"]))
    # Channel table beside it
    cpath = os.path.join(args.out_dir, args.out.replace(".csv", "_channels.csv"))
    # Write
    write_channel_csv(cpath, crow, list(manifest["meters"]), list(manifest["channels"]))
    # Say
    print(f"\nwrote {path} with {len(rows)} rows and {cpath} with {len(crow)} rows")


def pipeline_pieces(args, manifest, splits):
    """Yield (label, month, piece) the way socal_modeldata.iter_blocks does: per split, block and month, sliced to the block; no scaling is needed here."""
    # Splits
    for split in args.split_names:
        # Blocks
        for block in splits[split]:
            # Label as the scoring names it
            label = f"{split}_{block['start']}_{block['end']}"
            # Bounds
            start_s, end_s = sd.block_seconds(block)
            # Months the block touches
            for month in sd.months_in_block(block):
                # Month arrays
                arrays = sd.load_month(args.features_dir, month, allow_missing=True, manifest=manifest)
                # Missing month: reported as missing coverage, never as a verified piece
                if arrays is None:
                    # Yield a marker
                    yield label, month, None
                    # Next
                    continue
                # Slice to the block
                part = sd.slice_block(arrays, start_s, end_s)
                # No rows inside the block: missing coverage too
                if part is None:
                    # Marker
                    yield label, month, None
                    # Next
                    continue
                # Yield
                yield label, month, part


def self_test():
    """Synthetic fixtures with hand-checked expectations; exit 1 on any failure."""
    # Short windows so the fixture stays small
    args = argparse.Namespace(history=6, horizon=1, max_gap=3, min_meters=1, compare_dataset=False)
    # Three meters, one channel
    meters = ["m1", "m2", "m3"]
    # Manifest
    manifest = {"meters": meters, "channels": ["L1"]}
    # Slot grid: 10 s, starting 2025-01-31 23:55:00 UTC so a month boundary (fixed PST month = UTC minus 8 h, so the PST month changes at 08:00 UTC on Feb 1) can be placed inside a block
    t0 = int(pd.Timestamp("2025-02-01T07:58:00Z").timestamp())
    # 48 slots: 07:58:00 to 08:05:50 UTC
    t = t0 + 10 * np.arange(48)
    # Timeline: a topology change at 08:02:00 UTC (period 0 before, period 1 after), graph 0 joins m1-m2, graph 1 joins m1-m2 and m2-m3
    start_utc = int(pd.Timestamp("2025-01-01T00:00:00Z").timestamp())
    # Change time
    change = int(pd.Timestamp("2025-02-01T08:02:00Z").timestamp())
    # End
    end_utc = int(pd.Timestamp("2025-03-01T00:00:00Z").timestamp())
    # Timeline dict
    timeline = {"nodes": [{"meter": m} for m in meters],
                "graphs": [{"index": 0, "edges": [{"source": 0, "target": 1}]}, {"index": 1, "edges": [{"source": 0, "target": 1}, {"source": 1, "target": 2}]}],
                "periods": [{"index": 0, "start_utc": start_utc, "end_utc_exclusive": change, "graph": 0}, {"index": 1, "start_utc": change, "end_utc_exclusive": end_utc, "graph": 1}],
                "period_graph": np.array([0, 1])}
    # Validity: m1 always valid; m2 absent for slots 10..14 (a run of 5 >= max_gap 3); m3 valid magnitude everywhere but angle invalid everywhere
    mag_valid = np.ones((48, 3, 1), dtype=bool)
    # Gap
    mag_valid[10:15, 1, 0] = False
    # Angle validity
    ang_valid = mag_valid.copy()
    # m3 no angle
    ang_valid[:, 2, 0] = False
    # All passing so far
    ok = True
    # Report helper
    def rep(name, cond, note=""):
        # Print
        print(f"{'PASS' if cond else 'FAIL'}  {name}{': ' + note if note else ''}")
        # Return
        return bool(cond)
    # Case 1: one piece, whole array; flags
    piece = {"t": t, "mag_valid": mag_valid, "ang_valid": ang_valid}
    # Flags
    all_starts, retained, sender_ok, target_mag, target_ang, usable, tmag_ch, tang_ch = window_flags(piece, 6, 1, 3, 1)
    # Kept starts
    kept = all_starts[retained]
    # Starts 0..41 all retained (m1 always usable)
    ok &= rep("fixture: every window retained because m1 is always usable", len(kept) == 42 and kept[0] == 0 and kept[-1] == 41)
    # m2's history rule: a window whose history [s, s+5] contains the full run 10..14 (needs 3 consecutive absent inside) fails; windows with s in 5..14 contain at least 3 consecutive absent slots within 10..14? history covers s..s+5: s=5 covers 5..10 (one absent) ok; s=8 covers 8..13 (10,11,12,13 absent: run 4) fail; by hand: fail when history includes 3 consecutive of 10..14: s from 7 (7..12 has 10,11,12) to 12 (12..17 has 12,13,14)
    bad = set(range(7, 13))
    # Check
    ok &= rep("fixture: m2's history-only availability fails exactly where its history holds a run of 3 absent slots (starts 7 to 12)", set(np.nonzero(~sender_ok[:, 1])[0].tolist()) == bad, str(np.nonzero(~sender_ok[:, 1])[0].tolist()))
    # m2's target validity fails where the target slot s+6 is in 10..14: s in 4..8
    ok &= rep("fixture: m2's magnitude target invalid exactly where the target slot is absent (starts 4 to 8)", set(np.nonzero(~target_mag[:, 1])[0].tolist()) == set(range(4, 9)))
    # usable = both
    ok &= rep("fixture: m2 usable is sender_ok and target valid", np.array_equal(usable[:, 1], sender_ok[:, 1] & target_mag[:, 1]))
    # m3: sender ok and magnitude target valid everywhere, angle target never
    ok &= rep("fixture: m3 has magnitude targets everywhere and no angle target (mode-specific target validity)", target_mag[:, 2].all() and not target_ang[:, 2].any() and sender_ok[:, 2].all())
    # Case 2: period by last history slot: last slot = t[s+5]; change at 08:02:00 = slot index 24; windows with s+5 >= 24, s >= 19, are period 1
    period = smg.period_index_at(timeline["periods"], t[kept + 5])
    # Check
    ok &= rep("fixture: windows switch period exactly when the last history slot reaches the change time (start 19)", period[18] == 0 and period[19] == 1 and (period[:19] == 0).all() and (period[19:] == 1).all())
    # Case 3: counts through count_piece for a block that covers the whole array
    rows, totals = {}, {}
    # Neighbor sets
    nbs = neighbor_sets(timeline, 3)
    # Count
    n = count_piece("blockA", piece, args, timeline, nbs, rows, totals)
    # m1 as receiver: neighbor m2 in both graphs; unavailable windows = windows where m2 sender_ok False = 6 (starts 7..12); all in period 0
    r = rows[("blockA", 0, 0)]
    # Check
    ok &= rep("fixture: receiver m1 in period 0 counts 6 windows with its only neighbor unavailable", r["usable_with_unavailable"] == 6 and r["usable_with_sender"] == 19 - 6 and r["windows"] == 19)
    # m3 as receiver: no neighbor in period 0, neighbor m2 in period 1 (m2 available there), so eligible for masked magnitude only in period 1 and never in angle
    ok &= rep("fixture: receiver m3 has no neighbor in period 0 and an available one in period 1; eligible in magnitude only there and never in angle", rows[("blockA", 0, 2)]["eligible_mag"] == 0 and rows[("blockA", 1, 2)]["eligible_mag"] == 23 and rows[("blockA", 1, 2)]["eligible_ang"] == 0 and rows[("blockA", 1, 2)]["degree"] == 1)
    # Eligibility does not require the hidden meter's own history: m2 (history unusable at starts 7..12, target valid there, neighbor m1 available) is eligible in those windows
    r2 = rows[("blockA", 0, 1)]
    # In period 0 (starts 0..18): m2 target valid except starts 4..8, so 14 valid targets, all with m1 available; usable only 19 - 6 - 5 + overlap(7,8) = 10
    ok &= rep("fixture: a meter with unusable history but a valid target and an available neighbor is eligible (m2: 14 eligible against 10 usable in period 0)", r2["eligible_mag"] == 14 and r2["usable"] == 10 and r2["target_mag"] == 14)
    # Channel-level counts exist and agree with the meter-level ones on this one-channel fixture
    crow = {}
    # Recount with channel rows
    rows_c, totals_c = {}, {}
    # Count
    count_piece("blockA", piece, args, timeline, nbs, rows_c, totals_c, crow)
    # Check
    ok &= rep("fixture: per-channel eligible counts agree with the meter-level counts on a one-channel fixture", crow[("blockA", 0, 1, 0)]["eligible_mag"] == 14 and crow[("blockA", 1, 2, 0)]["eligible_ang"] == 0 and crow[("blockA", 1, 2, 0)]["eligible_mag"] == 23)
    # Pipeline restriction: a window with no usable meter is dropped even if a pair is eligible; fixture: m1 absent too for starts 20..21 target slots makes no usable meter there? Build a tiny piece: two meters, m1 history unusable (gap) but target valid, m2 history fine but target invalid, edge m1-m2: eligible pair (m1 hidden, m2 sends) in a window the pipeline drops
    t2 = t0 + 10 * np.arange(7)
    # Validity: 7 slots, history 6, horizon 1: exactly one window (start 0, target 6)
    mv = np.ones((7, 3, 1), dtype=bool)
    # m1 gap of 3 inside the history
    mv[2:5, 0, 0] = False
    # m2 target invalid
    mv[6, 1, 0] = False
    # m3 absent entirely
    mv[:, 2, 0] = False
    # Piece
    piece2 = {"t": t2, "mag_valid": mv, "ang_valid": mv}
    # Count
    rows_d, totals_d = {}, {}
    # Period 0 graph joins m1-m2
    count_piece("blockD", piece2, args, timeline, nbs, rows_d, totals_d)
    # Check: no retained window, one eligible magnitude pair in a dropped window
    ok &= rep("fixture: an eligible pair inside a window the pipeline drops (no usable meter) is counted as dropped, not as eligible", "blockD" in totals_d and totals_d["blockD"]["windows"] == 0 and totals_d["blockD"]["eligible_mag_dropped"] == 1)
    # Optional comparison with the model's dataset class when torch is present
    try:
        # Import
        import socal_models  # noqa: F401
        # Compare on the main fixture
        same, detail = compare_with_dataset(piece, args, timeline)
        # Check
        ok &= rep("fixture: retained starts, usability and periods equal socal_models.GraphWindowDataset", same, detail)
    # No torch here
    except ImportError:
        # Say
        print("SKIP  dataset comparison (torch not available in this environment); run with --compare-dataset on Colab")
    # Block totals: windows 42; windows with any unavailable sender = 6 (only m2 unavailable, and it is a neighbor of m1 in period 0)
    ok &= rep("fixture: block totals count 42 windows, 6 with an unavailable sender, and the directed messages", totals["blockA"]["windows"] == 42 and totals["blockA"]["windows_any_unavailable"] == 6 and totals["blockA"]["messages"] == 19 * 2 + 23 * 4 and totals["blockA"]["messages_unavailable"] == 6)
    # Case 4: block and month boundaries through the pipeline path: write a two-month fixture and a block that starts inside January's file and ends inside February's
    root = tempfile.mkdtemp()
    # Manifest
    with open(os.path.join(root, "manifest.json"), "w") as f:
        # Write
        json.dump(manifest, f)
    # January (PST) file: slots up to 07:59:50 UTC on Feb 1 belong to January's file; use 07:50:00 to 07:59:50 (60 slots)
    tj = int(pd.Timestamp("2025-02-01T07:50:00Z").timestamp()) + 10 * np.arange(60)
    # February file: 08:00:00 onward, 60 slots
    tf = int(pd.Timestamp("2025-02-01T08:00:00Z").timestamp()) + 10 * np.arange(60)
    # Arrays all valid
    for month, tt in (("2025-01", tj), ("2025-02", tf)):
        # Write npz as load_month expects
        np.savez(os.path.join(root, f"{month}.npz"), t=tt, rms=np.ones((60, 3, 1), dtype=np.float32), ang=np.zeros((60, 3, 1), dtype=np.float32), mask=np.ones((60, 3, 1), dtype=bool), angle_ok=np.ones((60, 3, 1), dtype=bool))
    # A block from 07:55:00 to 08:05:00 UTC: 30 slots in January's file, 30 in February's
    block = {"start": "2025-02-01a", "end": "2025-02-01b", "start_utc": "2025-02-01T07:55:00+00:00", "end_utc_exclusive": "2025-02-01T08:05:00+00:00"}
    # Args for the pipeline path
    pargs = argparse.Namespace(features_dir=root, split_names=["validation"], history=6, horizon=1, max_gap=3, min_meters=1)
    # Pieces
    pieces = list(pipeline_pieces(pargs, manifest, {"validation": [block]}))
    # Two pieces, 30 rows each
    ok &= rep("pipeline path: a block spanning a month boundary gives one piece per month, each sliced to the block", len(pieces) == 2 and all(len(p[2]["t"]) == 30 for p in pieces))
    # Windows per piece: 30 - 7 + 1 = 24 each, so 48 in all, none crossing the month boundary (a single 60-slot piece would give 54)
    rows2, totals2 = {}, {}
    # Count
    for label, month, part in pieces:
        # Accumulate
        count_piece(label, part, pargs, timeline, nbs, rows2, totals2)
    # Enforcement cases: a deliberately failing comparator must exit nonzero and write nothing
    out_dir = os.path.join(root, "diag_fail")
    # Arguments with the comparison requested
    eargs = argparse.Namespace(features_dir=root, split_names=["validation"], history=6, horizon=1, max_gap=3, min_meters=1, compare_dataset=True, out_dir=out_dir, out="x.csv")
    # Run with a comparator that always fails
    rows3, totals3, crow3, status3 = run(eargs, manifest, timeline, pipeline_pieces(eargs, manifest, {"validation": [block]}), [], comparator=lambda piece, a, tl: (False, "injected failure"))
    # Exit code
    code = None
    # Verify
    try:
        # Must raise
        verify_and_write(eargs, manifest, rows3, totals3, crow3, status3)
    # Caught
    except SystemExit as e:
        # Code (a message means exit 1)
        code = 1 if not isinstance(e.code, int) else e.code
    # Check: nonzero and no output folder or file
    ok &= rep("enforcement: a failing dataset comparison exits nonzero before any CSV is written", code == 1 and status3["compared"] == 2 and len(status3["failed"]) == 2 and not os.path.exists(os.path.join(out_dir, "x.csv")) and not os.path.exists(os.path.join(out_dir, "x_channels.csv")))
    # Zero pieces compared: a split with a block that has no data
    far = {"start": "2030-01-01", "end": "2030-01-02", "start_utc": "2030-01-01T00:00:00+00:00", "end_utc_exclusive": "2030-01-02T00:00:00+00:00"}
    # Run
    rows4, totals4, crow4, status4 = run(eargs, manifest, timeline, pipeline_pieces(eargs, manifest, {"validation": [far]}), [], comparator=lambda piece, a, tl: (True, "never called"))
    # Verify
    code = None
    # Try
    try:
        # Must raise
        verify_and_write(eargs, manifest, rows4, totals4, crow4, status4)
    # Caught
    except SystemExit as e:
        # Code
        code = 1 if not isinstance(e.code, int) else e.code
    # Check: zero compared, missing coverage listed, nonzero exit, nothing written
    ok &= rep("enforcement: a requested comparison with zero pieces compared and missing coverage exits nonzero and writes nothing", code == 1 and status4["compared"] == 0 and len(status4["missing"]) >= 1 and not os.path.exists(out_dir))
    # Passing comparator: writes both CSVs and exits normally
    ok_dir = os.path.join(root, "diag_ok")
    # Arguments
    oargs = argparse.Namespace(**{**vars(eargs), "out_dir": ok_dir})
    # Run
    rows5, totals5, crow5, status5 = run(oargs, manifest, timeline, pipeline_pieces(oargs, manifest, {"validation": [block]}), [], comparator=lambda piece, a, tl: (True, "injected pass"))
    # Verify and write
    verify_and_write(oargs, manifest, rows5, totals5, crow5, status5)
    # Check
    ok &= rep("enforcement: a passing comparison writes both CSVs", os.path.isfile(os.path.join(ok_dir, "x.csv")) and os.path.isfile(os.path.join(ok_dir, "x_channels.csv")) and status5["compared"] == 2)
    # Check
    ok &= rep("pipeline path: windows never cross the month or block boundary (24 + 24, not 54)", totals2[f"validation_{block['start']}_{block['end']}"]["windows"] == 48)
    # Verdict
    print("\nself-test: " + ("all cases passed" if ok else "some cases FAILED"))
    # Exit
    sys.exit(0 if ok else 1)


def main():
    """Entry point: self-test or the full count."""
    # Options
    args = parse_args()
    # Self-test
    if args.self_test:
        # Run
        self_test()
    # Manifest
    manifest = sd.load_manifest(args.features_dir)
    # Timeline
    timeline = load_timeline(args.graph)
    # Splits
    splits = sd.load_splits(args.splits)
    # Block labels in order, for the printed table
    blocks = [f"{s}_{b['start']}_{b['end']}" for s in args.split_names for b in splits[s]]
    # Say
    print(f"splits {args.split_names}; window rule history {args.history}, horizon {args.horizon}, max_gap {args.max_gap}, min_meters {args.min_meters}")
    # Count
    rows, totals, crow, status = run(args, manifest, timeline, pipeline_pieces(args, manifest, splits), blocks)
    # Enforce the verification outcome, then write
    verify_and_write(args, manifest, rows, totals, crow, status)
    # Summaries
    print_summaries(rows, totals, list(manifest["meters"]), blocks)


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
