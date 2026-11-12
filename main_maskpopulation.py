# =============================================================================
# File:        main_maskpopulation.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Report, on the real feature data, how many targets masked
#              forecasting can use under the available_neighbor policy,
#              before any masked model is trained.
#
#              Why it exists: the course staff's milestone feedback asked
#              whether the model uses the graph. Masked forecasting is the
#              direct test, and under the agreed rule a meter is hidden
#              only when its target is valid and at least one neighbor has
#              usable history in the window. The diagnostics showed that
#              neighbors without data are common, so the rule removes a
#              real share of the targets. A masked result can only be
#              read correctly when it is known how many targets each
#              block and period actually offers, and how many were left
#              out and why. This script counts them. It trains nothing
#              and needs no model.
#
#              What it counts, for every block-month piece of every split
#              and, inside a piece, for every topology period:
#                - the windows, and how many of them the pipeline keeps
#                - in kept windows: the eligible meter-windows, the
#                  eligible magnitude and angle targets (in total and per
#                  channel), and the permitted meters with a valid target
#                  that are left out for want of a neighbor with usable
#                  history
#                - the same in the windows the pipeline drops. The
#                  pipeline drops a window when no meter has both a
#                  usable history and a valid target. A hidden meter
#                  needs no usable history of its own, so such a window
#                  can still hold an eligible target. These are reported
#                  here and are not added to training or scoring.
#
#              How the edge set of a window is chosen: by the period that
#              holds the window's last history slot, exactly as the model
#              chooses it. A window whose history crosses a change of
#              topology is kept whole and judged under that edge set. A
#              piece that spans several periods therefore gives one row
#              per period.
#
#              Every piece gets a row, also a piece with no window at all
#              or with no kept window, so that nothing is silently
#              missing from the report.
#
#              Permitted meters are the meters with an edge in at least
#              one edge set of the graph file, the same list the training
#              script records. The neighbors come from the graph file's
#              own edges.
#
# Usage (from the repo root):
#   python main_maskpopulation.py --features-dir /content/features --graph graph_meters_periods.json --splits splits_v2.json --out /content/drive/MyDrive/socal_runs/diagnostics/mask_population.csv
#   Options: python main_maskpopulation.py --help
#
# Output:      A printed table, and a CSV with one row per split, block,
#              month and period. The CSV is never written over: an
#              existing file stops the script before any work.
#
# Exit code:   0 when the report was written, nonzero with a message
#              otherwise.
#
# Functions:
#   parse_args()                       command-line options with defaults
#   graph_parts(body, meters)          edge sets, periods, the period-to-edge-set table and the permitted meters of a graph file
#   piece_rows(...)                    the report rows of one block-month piece
#   main()                             read, count, print, write
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: the report table
import csv
# Standard library: JSON
import json
# Standard library: paths
import os
# Standard library: exit code
import sys
# Third party: numerics
import numpy as np
# Project: feature files, splits and the window selection rule
import socal_modeldata as sd
# Project: the period lookup
import socal_metergraph as smg
# Project: the eligibility rule and its counts
import socal_maskpolicy as mp


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Report how many targets masked forecasting can use under the available_neighbor policy, per block, month and period")
    # Feature files
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with the monthly feature files and manifest.json")
    # Splits
    parser.add_argument("--splits", default="splits_v2.json", help="Splits file")
    # Which splits to report
    parser.add_argument("--split-names", nargs="+", default=["train", "validation", "test_events", "test_normal"], help="Splits to report")
    # Graph
    parser.add_argument("--graph", default="graph_meters_periods.json", help="Graph file whose edges define the neighbors: fixed or time-indexed")
    # Window, as trained
    parser.add_argument("--history", type=int, default=60, help="History slots, as trained")
    # Horizon
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in slots, as trained")
    # Gap rule
    parser.add_argument("--max-gap", type=int, default=12, help="Gap rule, as trained")
    # Window rule
    parser.add_argument("--min-meters", type=int, default=1, help="The pipeline keeps a window when at least this many meters pass the gap rule with a valid target, as trained")
    # Memory bound
    parser.add_argument("--chunk", type=int, default=4096, help="Windows counted at a time")
    # Output
    parser.add_argument("--out", default="mask_population.csv", help="Where to write the table; it must not exist")
    # Parse
    args = parser.parse_args()
    # The numbers must be at least 1
    for name in ("history", "horizon", "max_gap", "min_meters", "chunk"):
        # Refuse anything smaller
        if getattr(args, name) < 1:
            # Refuse
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    # Done
    return args


def graph_parts(body, meters):
    """
    What the count needs from a graph file, fixed or time-indexed:
      adjacency     (G, M, M) bool, the neighbor matrix of every edge set
      periods       the file's periods, or None for a fixed graph
      period_graph  (P,) int, the edge set of each period, or None
      permitted     (M,) bool, meters with a neighbor in some edge set
    meters is the manifest's meter order.
    """
    # Each meter's node
    nodes = [int(body["meters"][m]["node"]) for m in meters]
    # Time-indexed: one edge set per graph, and the periods
    if "graphs" in body and "periods" in body:
        # Edge sets
        graphs = body["graphs"]
        # Periods
        periods = body["periods"]
        # Edge set of each period
        period_graph = np.array([int(p["graph"]) for p in periods], dtype=np.int64)
    # Fixed: one edge set, no periods
    else:
        # One edge set
        graphs, periods, period_graph = [body], None, None
    # Neighbor matrices from the file's own edges
    adjacency = np.stack([mp.meter_adjacency(g, nodes) for g in graphs])
    # Permitted: a neighbor in some edge set
    return adjacency, periods, period_graph, adjacency.any(axis=(0, 2))


def piece_rows(split, label, month, part, adjacency, periods, period_graph, permitted, channels, args):
    """
    The report rows of one block-month piece: one row per topology period
    that holds the last history slot of some window, or a single row of
    zeros when the piece has no window. part is the sliced month (t,
    mag_valid, ang_valid), or None when the month has no rows in the block.
    """
    # What every row of the piece starts with
    base = {"split": split, "block": label, "month": month}
    # No rows of the month lie in the block
    if part is None:
        # One row saying so
        return [dict(base, period="", period_start="", edge_set="", note="no feature rows in this block and month", **mp.sum_counts({"kept": np.zeros(0, dtype=bool), "eligible_pairs": np.zeros(0, dtype=np.int64), "eligible_mag": np.zeros((0, len(channels)), dtype=np.int64), "eligible_ang": np.zeros((0, len(channels)), dtype=np.int64), "no_neighbor_pairs": np.zeros(0, dtype=np.int64)}))]
    # Slot times and validity
    t, mv, av = part["t"], part["mag_valid"], part["ang_valid"]
    # Number of window starts
    n = max(len(t) - (args.history + args.horizon) + 1, 0)
    # The windows the pipeline keeps, by its own rule
    kept_starts, _ = sd.retained_starts(mv, args.history, args.horizon, args.max_gap, args.min_meters)
    # The last history slot of every window
    last = t[np.arange(n) + args.history - 1]
    # The period holding it, and that period's edge set; a fixed graph has one edge set
    period = smg.period_index_at(periods, last) if (periods is not None and n) else np.zeros(n, dtype=np.int64)
    # Edge set per window
    window_graph = period_graph[period] if periods is not None else np.zeros(n, dtype=np.int64)
    # Counts for every window
    per = mp.eligibility_counts_np(mv, av, adjacency, window_graph, permitted, args.history, args.horizon, args.max_gap, kept_starts, args.chunk)
    # A piece too short for any window still gets a row
    if n == 0:
        # One row of zeros
        return [dict(base, period="", period_start="", edge_set="", note=f"{len(t)} slots, too few for a window", **mp.sum_counts(per))]
    # One row per period present
    rows = []
    # Each period, in order
    for p in np.unique(period):
        # Totals over that period's windows
        totals = mp.sum_counts(per, period == p)
        # Row
        rows.append(dict(base, period=(int(p) if periods is not None else "fixed"), period_start=(periods[int(p)].get("start", "") if periods is not None else ""), edge_set=int(window_graph[period == p][0]), note=("" if totals["kept_windows"] else "no window kept by the pipeline"), **totals))
    # Done
    return rows


def main():
    """Read the inputs, count every piece, print the report, write the table."""
    # Options
    args = parse_args()
    # Never write over an existing file; checked before any work
    if os.path.lexists(args.out):
        # Stop
        raise SystemExit(f"refused: {args.out} already exists and is not written over; move or remove it first")
    # Manifest: meter and channel order
    manifest = sd.load_manifest(args.features_dir)
    # Names
    meters, channels = manifest["meters"], manifest["channels"]
    # Splits
    splits = sd.load_splits(args.splits)
    # Graph file
    with open(args.graph) as f:
        # Parse
        body = json.load(f)
    # Edge sets, periods and permitted meters
    adjacency, periods, period_graph, permitted = graph_parts(body, meters)
    # Say what is counted
    print(f"masked target population under the available_neighbor policy: graph {args.graph} ({adjacency.shape[0]} edge set(s), {'time-indexed, edge set chosen by the last history slot' if periods is not None else 'fixed'}), history {args.history}, horizon {args.horizon}, gap rule {args.max_gap}, min meters {args.min_meters}")
    # Permitted meters
    print(f"permitted meters ({int(permitted.sum())}): {[m for m, ok in zip(meters, permitted) if ok]}; never permitted: {[m for m, ok in zip(meters, permitted) if not ok]}")
    # Rows of the report
    rows = []
    # Each split
    for split in args.split_names:
        # Each block of the split
        for block in splits.get(split, []):
            # Its label, as the scoring names it
            label = f"{split}_{block['start']}_{block['end']}"
            # Its bounds in seconds
            start_s, end_s = sd.block_seconds(block)
            # Each month the block touches
            for month in sd.months_in_block(block):
                # The month file; a missing file is reported, not skipped silently
                arrays = sd.load_month(args.features_dir, month, allow_missing=True, manifest=manifest)
                # Missing
                if arrays is None:
                    # One row saying so
                    rows.append(dict({"split": split, "block": label, "month": month, "period": "", "period_start": "", "edge_set": "", "note": "month file missing"}, **mp.sum_counts({"kept": np.zeros(0, dtype=bool), "eligible_pairs": np.zeros(0, dtype=np.int64), "eligible_mag": np.zeros((0, len(channels)), dtype=np.int64), "eligible_ang": np.zeros((0, len(channels)), dtype=np.int64), "no_neighbor_pairs": np.zeros(0, dtype=np.int64)})))
                    # Next month
                    continue
                # The rows of the month inside the block, then the piece's report rows
                rows.extend(piece_rows(split, label, month, sd.slice_block(arrays, start_s, end_s), adjacency, periods, period_graph, permitted, channels, args))
    # Printed table
    print(f"\n{'split':<12} {'block':<44} {'month':<8} {'period':>6} {'set':>4} {'windows':>9} {'kept':>9} {'eligible':>10} {'mag':>10} {'ang':>10} {'no nbr':>9} | {'dropped: eligible':>17} {'mag':>8} {'ang':>8} {'no nbr':>8}  note")
    # Each row
    for r in rows:
        # Line
        print(f"{r['split']:<12} {r['block']:<44} {r['month']:<8} {str(r['period']):>6} {str(r['edge_set']):>4} {r['windows']:>9} {r['kept_windows']:>9} {r['kept_eligible_pairs']:>10} {r['kept_eligible_mag']:>10} {r['kept_eligible_ang']:>10} {r['kept_no_neighbor_pairs']:>9} | {r['dropped_eligible_pairs']:>17} {r['dropped_eligible_mag']:>8} {r['dropped_eligible_ang']:>8} {r['dropped_no_neighbor_pairs']:>8}  {r['note']}")
    # Totals per split
    print("\ntotals per split (kept windows: eligible meter-windows, share of permitted meter-windows with a valid target that are eligible; dropped windows: eligible meter-windows, reported and not used)")
    # Each split
    for split in args.split_names:
        # Its rows
        mine = [r for r in rows if r["split"] == split]
        # Sums
        el, lonely, dropped = sum(r["kept_eligible_pairs"] for r in mine), sum(r["kept_no_neighbor_pairs"] for r in mine), sum(r["dropped_eligible_pairs"] for r in mine)
        # Line
        print(f"  {split:<12} windows {sum(r['windows'] for r in mine):>10}, kept {sum(r['kept_windows'] for r in mine):>10}; kept: eligible {el:>10}, left out for want of a neighbor {lonely:>9}, share eligible {(100.0 * el / (el + lonely) if el + lonely else float('nan')):.1f}%; dropped: eligible {dropped}")
    # The per-channel lists go to the table as text, with the channel order in the column name
    order = "|".join(channels)
    # Table rows
    table = []
    # Each row
    for r in rows:
        # Copy without the lists
        row = {k: v for k, v in r.items() if not k.endswith("_by_channel")}
        # The lists as text
        for name in ("kept_eligible_mag", "kept_eligible_ang", "dropped_eligible_mag", "dropped_eligible_ang"):
            # One value per channel, in channel order
            row[f"{name}_by_channel ({order})"] = "|".join(str(v) for v in r[f"{name}_by_channel"])
        # Keep
        table.append(row)
    # Create the file; mode "x" fails when it exists, so nothing can be written over
    with open(args.out, "x", newline="") as f:
        # Columns from the first row
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        # Header
        writer.writeheader()
        # Rows
        writer.writerows(table)
    # Say what was written
    print(f"\nwrote {args.out}: {len(table)} rows")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
