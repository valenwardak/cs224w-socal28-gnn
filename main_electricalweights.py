# =============================================================================
# File:        main_electricalweights.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Write a copy of the time-indexed meter graph in which every
#              edge carries an electrical weight instead of 1.0, and print
#              the report the reviewers asked for before any model is
#              trained on it.
#
#              Why it exists: the course staff's milestone feedback asked
#              whether the model uses the graph. The current graph says
#              which meters are connected but gives every connection the
#              same weight. This script computes how strongly each
#              connected pair is coupled through the electrical network,
#              period by period, so a model can be trained on a graph that
#              carries that information and compared with the current one.
#
#              What it does, in order:
#                1. Refuses to start when an output file already exists,
#                   when the two outputs are the same file, or when an
#                   output is one of the inputs under any path. No
#                   existing file is ever written over or changed.
#                2. Puts the complete terminal block of every line and
#                   transformer in graph.json per unit: shunt parts,
#                   transformer ratios and phase shifts included. An
#                   element that cannot be reduced cleanly stops the
#                   script: it is reported, not approximated.
#                3. Rebuilds the electrical network at the reference time
#                   from the network file and the switch records and
#                   requires the same nodes as graph.json.
#                4. For each period, rebuilds the network at the period's
#                   midpoint, requires every meter's node to hold the buses
#                   the timeline file records, reduces the network onto the
#                   metered nodes, and requires the coupled meter pairs to
#                   be exactly the edges of the period's edge set. A
#                   reduction that is badly conditioned, or whose solve
#                   does not check, is refused.
#                5. Compares the weights of the periods that share an edge
#                   set. When they agree the edge sets keep their numbers.
#                   When they differ, the edge set is split, so that every
#                   period keeps the weights of its own network.
#                6. Scales the weights of each edge set to (0, 1] by its
#                   largest weight. Two meters on one node get 1.0.
#                7. Repeats steps 2 to 6 with the series-only approximation
#                   (lines by their series part, transformers from the
#                   nameplate) and reports, for every period, how far its
#                   weights are from the complete-block weights. The
#                   approximation is never written to the graph file.
#                8. Writes the weighted graph file and a CSV with one row
#                   per edge. The layout is the timeline file's, so the
#                   model code reads it unchanged, with
#                   --edge-weights admittance.
#                9. Prints the report: the per-unit elements, the reduction
#                   of every period, the comparison with the approximation,
#                   the raw and scaled weights by edge set, the weights by
#                   the number of transformers on the path, the GCN
#                   coefficients under uniform and electrical weights for
#                   every edge set, and how the scaling rule behaves
#                   across edge sets.
#
#              The weight is not an electrical equivalent of the network.
#              See socal_electricalweights.py. Nothing is trained or scored
#              here.
#
# Usage (from the repo root):
#   python main_electricalweights.py
#   python main_electricalweights.py --dry-run
#   python main_electricalweights.py --s-base 1e6 --out graph_meters_periods_pu.json
#   Options: python main_electricalweights.py --help
#
# Output:
#   graph_meters_periods_pu.json         the timeline file with weighted edges:
#       admittance_abs    the scaled weight in (0, 1], read by the model code
#       weight_raw_pu     the magnitude of the reduced per-unit entry, null for two meters on one node
#       same_node         true for two meters on one node
#       plus "weights", a record of the settings, the inputs and what the weight leaves out
#   graph_meters_periods_pu_report.csv   one row per edge set and edge
#
# Exit code:   0 when the files were written (or the dry run finished),
#              nonzero with a message when anything was refused.
#
# Functions:
#   parse_args()                      command-line options with defaults, checked
#   file_identity(path)               file name and the first 16 hex digits of its SHA-256
#   switch_files(topology_dir)        the switch record files that are read
#   refuse_outputs(args, inputs)      stop when an output exists, equals the other output, or is an input
#   transformers_on(path_elements)    number of transformers among an edge's path elements
#   reduce_all(data, branches, args)  the reduction of every period, stopping on any problem
#   build(args)                       everything up to the weighted file body and the report rows
#   print_report(result, args)        the printed report
#   main()                            refuse, build, report, write
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: copy the timeline body before changing it
import copy
# Standard library: the edge table
import csv
# Standard library: file identities
import hashlib
# Standard library: JSON
import json
# Standard library: paths
import os
# Standard library: exit code
import sys
# Third party: numerics
import numpy as np
# Third party: the reference time
import pandas as pd
# Project: the switch records reader
import socal_metergraph as smg
# Project: the electrical weight helpers
import socal_electricalweights as ew


def parse_args():
    """Read the command-line options. Every setting has a default, and every number is checked."""
    # Parser
    parser = argparse.ArgumentParser(description="Write the time-indexed meter graph with electrical edge weights, and print the coefficient report")
    # The electrical graph with the terminal blocks
    parser.add_argument("--graph", default="graph.json", help="Electrical graph at the reference time, with the terminal blocks")
    # The time-indexed meter graph
    parser.add_argument("--timeline", default="graph_meters_periods.json", help="Time-indexed meter graph whose edges are weighted")
    # Caltech's topology folder
    parser.add_argument("--topology-dir", default="topology", help="Caltech's topology folder, with the switch status files")
    # The network file
    parser.add_argument("--network", default=os.path.join("topology", "network_files", "circuit3", "2023-08-01T00h00m00.000000s.json"), help="Network file")
    # Power base
    parser.add_argument("--s-base", type=float, default=1e6, help="Common power base in volt-amperes")
    # Coupling test
    parser.add_argument("--coupling-limit", type=float, default=0.01, help="Largest coupling between sequences for a block to count as balanced")
    # Conditioning
    parser.add_argument("--condition-limit", type=float, default=1e12, help="Largest condition number of a connected part of the eliminated block; above it the reduction is refused")
    # Solve check
    parser.add_argument("--residual-limit", type=float, default=1e-9, help="Largest relative residual of the reduction's solve; above it the reduction is refused")
    # Agreement of periods that share an edge set
    parser.add_argument("--weight-tolerance", type=float, default=1e-9, help="Relative difference below which two periods' weights count as equal")
    # Small weights
    parser.add_argument("--small-weight", type=float, default=1e-6, help="Scaled weights below this are listed in the report; they are kept as they are")
    # Approximation: transformer resistance
    parser.add_argument("--comparison-x-over-r", type=float, default=None, help="Series-only approximation: ratio of reactance to resistance for transformers; not given means pure reactance, as the nameplate gives no resistance")
    # Approximation: which rating
    parser.add_argument("--rating-index", type=int, default=0, help="Series-only approximation and nameplate cross-check: which listed rated power to use when a transformer lists several (0 is the first)")
    # Outputs
    parser.add_argument("--out", default="graph_meters_periods_pu.json", help="Where to write the weighted graph file; it must not exist")
    # Edge table
    parser.add_argument("--report", default="graph_meters_periods_pu_report.csv", help="Where to write the edge table; it must not exist")
    # Compute and report only
    parser.add_argument("--dry-run", action="store_true", help="Compute and print the report, write nothing")
    # Parse
    args = parser.parse_args()
    # Numbers that must be positive and finite
    for name in ("s_base", "coupling_limit", "condition_limit", "residual_limit", "weight_tolerance", "small_weight"):
        # The value
        value = getattr(args, name)
        # Refuse zero, negative, infinite and not-a-number
        if not np.isfinite(value) or not value > 0:
            # Refuse
            parser.error(f"--{name.replace('_', '-')} must be a positive finite number, got {value}")
    # The optional ratio, when given
    if args.comparison_x_over_r is not None and (not np.isfinite(args.comparison_x_over_r) or not args.comparison_x_over_r > 0):
        # Refuse
        parser.error(f"--comparison-x-over-r must be a positive finite number, got {args.comparison_x_over_r}")
    # The rating index
    if args.rating_index < 0:
        # Refuse
        parser.error(f"--rating-index must be 0 or more, got {args.rating_index}")
    # Done
    return args


def file_identity(path):
    """The file's name and the first 16 hex digits of the SHA-256 of its bytes, as the training script records graph files."""
    # Hash the bytes
    with open(path, "rb") as f:
        # Digest
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    # Name and digest
    return f"{os.path.basename(path)}:{digest}"


def switch_files(topology_dir):
    """The switch record files the construction reads: every status file in the topology folder's parameter_timeseries, in name order."""
    # The folder
    folder = os.path.join(topology_dir, "parameter_timeseries")
    # Its status files
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder)) if name.endswith("-tbus_status.csv")]


def refuse_outputs(args, inputs):
    """
    Stop (SystemExit with a message) unless both outputs are safe to
    create: they are two different files, neither is an input under any
    path (a relative path, a symbolic link and a hard link are all seen
    through), and neither exists yet. Nothing is deleted or replaced by
    this script. A file from an earlier run has to be moved or removed by
    hand, which keeps earlier diagnostic files safe.
    """
    # The two outputs, by their real paths
    out, report = os.path.realpath(args.out), os.path.realpath(args.report)
    # One file for both outputs: the table would replace the graph
    if out == report or (os.path.exists(args.out) and os.path.exists(args.report) and os.path.samefile(args.out, args.report)):
        # Refuse
        raise SystemExit(f"refused: --out and --report are the same file ({args.out}); nothing written")
    # Each output
    for path in (args.out, args.report):
        # Against each input
        for source in inputs:
            # The same file by real path, or the same file on disk through a link
            if os.path.realpath(path) == os.path.realpath(source) or (os.path.exists(path) and os.path.samefile(path, source)):
                # Refuse
                raise SystemExit(f"refused: the output {path} is the input file {source}; nothing written")
        # Anything already there, a broken link included
        if os.path.lexists(path):
            # Refuse
            raise SystemExit(f"refused: {path} already exists and is not written over; move or remove it first; nothing written")
        # The folder must be there
        if not os.path.isdir(os.path.dirname(os.path.abspath(path))):
            # Refuse
            raise SystemExit(f"refused: the folder of {path} does not exist; nothing written")


def transformers_on(path_elements):
    """How many transformers lie on an edge's recorded physical path."""
    # Transformer elements are named tr_<number>
    return sum(1 for e in path_elements if str(e).startswith("tr_"))


def reduce_all(data, branches, args):
    """The reduction of every period with one set of branches. Stops (SystemExit) with the list of problems when any check fails or a reduction is refused."""
    # Each period
    periods = [ew.period_reduction(data["net"], data["records"], data["timeline"], i, branches, args.weight_tolerance, args.condition_limit, args.residual_limit) for i in range(len(data["timeline"]["periods"]))]
    # Every problem
    problems = [t for p in periods for t in p["problems"]]
    # Any problem stops the build
    if problems:
        # Stop
        raise SystemExit("refused: the rebuilt networks do not pass their checks, nothing written:\n  " + "\n  ".join(problems))
    # Done
    return periods


def build(args):
    """
    Everything up to the weighted file body: per-unit terminal blocks, the
    reference check, the reduction of every period, the comparison with
    the series-only approximation, the grouping of periods into weighted
    edge sets, and the scaling. Returns a dictionary with the file body,
    the report rows and the intermediate results, or raises SystemExit
    with the reason when a check does not hold.
    """
    # Inputs
    data = {}
    # The electrical graph with the terminal blocks
    with open(args.graph) as f:
        # Parse
        data["graph"] = json.load(f)
    # The time-indexed meter graph
    with open(args.timeline) as f:
        # Parse
        data["timeline"] = json.load(f)
    # The network file
    with open(args.network) as f:
        # Parse
        data["net"] = json.load(f)
    # Switch records
    data["records"] = smg.read_switch_records(os.path.join(args.topology_dir, "parameter_timeseries"))
    # Shorthand
    graph, timeline, net = data["graph"], data["timeline"], data["net"]
    # Per-unit terminal blocks: the construction the weights come from
    branches, notes = ew.branches_per_unit(net, graph, args.s_base, args.rating_index, args.coupling_limit, "terminal")
    # An element that could not be reduced is reported and stops the build: leaving it out would change the network
    if notes["unresolved"]:
        # Stop
        raise SystemExit("refused: elements that could not be put per unit, nothing written:\n  " + "\n  ".join(notes["unresolved"]))
    # Reference check: the rebuilt nodes at the reference time must be graph.json's nodes
    root, unresolved = ew.merged_nodes(net, data["records"], pd.Timestamp(graph["reference_time"]), set(branches))
    # Bus sets of the rebuilt nodes
    mine = {}
    # Collect
    for bus, r in root.items():
        # Under its representative
        mine.setdefault(r, set()).add(bus)
    # Compare as sets of bus sets
    if unresolved or {frozenset(v) for v in mine.values()} != {frozenset(n["buses"]) for n in graph["nodes"]}:
        # Stop
        raise SystemExit(f"refused: the network rebuilt at the reference time {graph['reference_time']} does not have the nodes of {args.graph}; unreadable switch states: {unresolved}")
    # The reduction of every period
    periods = reduce_all(data, branches, args)
    # The series-only approximation, for the comparison only
    approx_branches, _ = ew.branches_per_unit(net, graph, args.s_base, args.rating_index, args.coupling_limit, "series", args.comparison_x_over_r)
    # Its reduction of every period
    approx = reduce_all(data, approx_branches, args)
    # Per period: how far the approximation's weights are from the complete-block weights
    comparison = []
    # Each period
    for p, q in zip(periods, approx):
        # Scaled weights of both
        sp, sq = ew.scale_weights(p["raw"])[0], ew.scale_weights(q["raw"])[0]
        # Largest relative difference of the raw and of the scaled weights, over the edges with a number
        comparison.append({"raw": max((abs(b - a) / a for a, b in zip(p["raw"], q["raw"]) if a is not None), default=0.0), "scaled": max((abs(b - a) / a for a, b, r in zip(sp, sq, p["raw"]) if r is not None), default=0.0)})
    # Weighted edge sets: each source edge set, split when its periods carry different weights
    sets, period_graph = ew.weighted_edge_sets([int(p["graph"]) for p in timeline["periods"]], [p["raw"] for p in periods], len(timeline["graphs"]), args.weight_tolerance)
    # File body: the timeline's, with weighted edges
    body = copy.deepcopy(timeline)
    # New graphs list
    graphs = []
    # Each weighted edge set
    for index, x in enumerate(sets):
        # The source edge set
        src = timeline["graphs"][x["source"]]
        # An edge set no period uses has no network to weigh it in: stop, do not invent weights
        if x["raw"] is None:
            # Stop
            raise SystemExit(f"refused: edge set {x['source']} is used by no period, so it has no weights; nothing written")
        # Scaled weights and the largest raw weight
        scaled, top = ew.scale_weights(x["raw"])
        # Edges with the weights: the scaled weight goes where the model code reads it
        edges = [dict(e, admittance_abs=float(s), weight_raw_pu=(None if r is None else float(r)), same_node=bool(r is None)) for e, r, s in zip(src["edges"], x["raw"], scaled)]
        # Graph entry
        graphs.append({"index": index, "built_at": src.get("built_at"), "n_edges": len(edges), "source_graph": x["source"], "variant": x["variant"], "weight_scale_pu": top, "weighted_from_periods": x["periods"], "edges": edges})
    # Replace the graphs
    body["graphs"] = graphs
    # Point every period at its weighted edge set
    for i, p in enumerate(body["periods"]):
        # Index
        p["graph"] = int(period_graph[i])
    # What this file is
    body["note"] = (timeline.get("note", "") + " Electrical weights (main_electricalweights.py): admittance_abs on every edge is the magnitude of the "
                    "pair's entry in the per-unit positive-sequence nodal admittance matrix of the period's network, assembled from the complete terminal "
                    "blocks of graph.json (shunt parts, transformer ratios and phase shifts included) and Kron-reduced onto the metered nodes, divided by "
                    "the largest value of its edge set; two meters on one node carry 1.0. It is a scalar coupling measure, not an electrical equivalent "
                    "of the network: it is positive-sequence only and a magnitude does not carry the angle of the reduced entry. Connections are "
                    "unchanged from the source file.")
    # Settings and inputs
    body["weights"] = {
        # What the weights are
        "kind": "kron_reduced_positive_sequence_terminal_blocks_per_unit",
        # Settings
        "s_base_va": float(args.s_base), "coupling_limit": float(args.coupling_limit), "condition_limit": float(args.condition_limit), "residual_limit": float(args.residual_limit), "weight_tolerance": float(args.weight_tolerance),
        # Scaling
        "scaling": "per edge set: divided by its largest weight; same-node pairs 1.0", "self_loop": "1.0, added by the GCN layer",
        # Inputs
        "inputs": {"graph": file_identity(args.graph), "timeline": file_identity(args.timeline), "network": file_identity(args.network), "switch_records": [file_identity(p) for p in switch_files(args.topology_dir)]},
        # What a scalar weight leaves out
        "left_out": ["negative and zero sequence", "the angle of the reduced entry"],
        # Transformers whose rated voltage differs from their bus; the ratio is inside their blocks
        "off_nominal": notes["off_nominal"],
        # The approximation is reported, never stored
        "comparison": {"what": "series-only approximation (lines by their series part, transformers from the nameplate with no ratio and no phase shift); not stored in this file", "transformer_x_over_r": args.comparison_x_over_r,
                       "rating_index": int(args.rating_index), "largest_relative_difference_raw": max(c["raw"] for c in comparison), "largest_relative_difference_scaled": max(c["scaled"] for c in comparison)},
    }
    # Edge table: one row per weighted edge set and edge, with the GCN coefficients under uniform and electrical weights
    names = [n["meter"] for n in timeline["nodes"]]
    # Rows
    rows = []
    # Each weighted edge set
    for gph in graphs:
        # Node pairs
        pairs = [(e["source"], e["target"]) for e in gph["edges"]]
        # Coefficients with uniform weights
        _, _, nb_uniform = ew.gcn_coefficients(len(names), pairs, [1.0] * len(pairs))
        # Coefficients with the electrical weights
        _, _, nb_weighted = ew.gcn_coefficients(len(names), pairs, [e["admittance_abs"] for e in gph["edges"]])
        # Each edge
        for e, cu, cw in zip(gph["edges"], nb_uniform, nb_weighted):
            # Row
            rows.append({"graph": gph["index"], "source_graph": gph["source_graph"], "meter_a": names[e["source"]], "meter_b": names[e["target"]], "hops": e.get("hops"), "transformers_on_path": transformers_on(e.get("path_elements", [])),
                         "path_elements": "|".join(e.get("path_elements", [])), "same_node": e["same_node"], "weight_raw_pu": e["weight_raw_pu"], "weight_scaled": e["admittance_abs"], "neighbor_coefficient_uniform": float(cu), "neighbor_coefficient_electrical": float(cw)})
    # Everything the report and the writer need
    return {"body": body, "rows": rows, "notes": notes, "periods": periods, "comparison": comparison, "sets": sets, "timeline": timeline, "names": names}


def print_report(result, args):
    """Print the report the reviewers asked for before any training on the weighted graph."""
    # Shorthand
    notes, body, names = result["notes"], result["body"], result["names"]
    # Settings
    print(f"electrical weights from the complete terminal blocks: power base {args.s_base:.0f} VA, coupling limit {args.coupling_limit}, condition limit {args.condition_limit:.0e}, residual limit {args.residual_limit:.0e}")
    # Lines
    print(f"\nlines with a block: {len(notes['lines'])}; bus voltages (line-to-neutral) {sorted({x['v_ln'] for x in notes['lines']})}; per-unit series admittance {min(x['y_series_abs'] for x in notes['lines']):.1f} to {max(x['y_series_abs'] for x in notes['lines']):.1f}; "
          f"X/R {min(x['x_over_r'] for x in notes['lines']):.2f} to {max(x['x_over_r'] for x in notes['lines']):.2f}; largest coupling between sequences {max(x['coupling'] for x in notes['lines']):.1e}; "
          f"largest shunt part relative to the series part {max(x['shunt_ratio'] for x in notes['lines']):.1e} (the shunt part is in the matrix)")
    # Transformers
    print(f"\ntransformers: {len(notes['transformers'])}. Columns: the block's own numbers per unit on the bus voltages (|y11| at the from end, |y22| at the to end), the ratio, phase shift and X/R they imply,")
    # Second line of the explanation
    print("the ideal-transformer test (0 means the block is a series admittance behind an ideal ratio and shift, with no further shunt part), and the nameplate magnitude beside the block's on the rated voltage")
    # Header
    print(f"  {'name':<7} {'connection':<12} {'kVA listed':<12} {'%z':>5} {'kV rated':<13} {'|y11|':>7} {'|y22|':>7} {'ratio':>7} {'shift':>6} {'X/R':>5} {'ideal test':>10} {'nameplate':>9} {'block':>7} {'difference':>10}")
    # Each
    for t in notes["transformers"]:
        # Line
        print(f"  {t['name']:<7} {'-'.join(t['conn']):<12} {str(t['kva_listed']):<12} {t['percent_z']:>5.2f} {str(t['kv']):<13} {t['y11_abs']:>7.3f} {t['y22_abs']:>7.3f} {t['ratio']:>7.4f} {t['shift_deg']:>6.1f} {t['x_over_r']:>5.2f} {t['residual']:>10.1e} "
              f"{t['nameplate_abs']:>9.3f} {abs(t['block_on_rated']):>7.3f} {abs(abs(t['block_on_rated']) - t['nameplate_abs']) / t['nameplate_abs']:>10.1e}")
    # The rating
    print(f"the nameplate column uses rating index {args.rating_index} where several ratings are listed. That choice is consistent with the supplied blocks. It is not a verified operating capacity, and the weights do not use it: they come from the blocks.")
    # Off-nominal ratios
    print("transformers whose rated voltage differs from the nominal voltage of their bus (the ratio is inside their block and enters the matrix): " + ("; ".join(notes["off_nominal"]) if notes["off_nominal"] else "none"))
    # Periods
    print("\nreduction per period (checks passed for every period: each meter's node holds the buses the timeline records, the coupled pairs are exactly the edges, the conditioning and the solve are within their limits)")
    # Header
    print(f"  {'period':>6} {'source set':>10} {'weighted set':>12} {'edges':>6} {'eliminated':>10} {'left out':>8} {'parts':>6} {'condition':>10} {'residual':>9} {'approx raw':>11} {'approx scaled':>13}")
    # Each
    for i, (p, c) in enumerate(zip(result["periods"], result["comparison"])):
        # Line
        print(f"  {i:>6} {int(result['timeline']['periods'][i]['graph']):>10} {int(body['periods'][i]['graph']):>12} {len(p['pairs']):>6} {p['info']['eliminated']:>10} {p['info']['left_out']:>8} {p['info']['parts']:>6} {p['info']['condition']:>10.1f} {p['info']['residual']:>9.1e} {100 * c['raw']:>10.2f}% {100 * c['scaled']:>12.2f}%")
    # What the last two columns are
    print(f"approx raw and approx scaled: the largest relative difference, over the period's edges, between the series-only approximation (transformers {'pure reactance' if args.comparison_x_over_r is None else 'X/R ' + str(args.comparison_x_over_r)}, no ratio, no phase shift, no shunt) and the complete-block weights. "
          f"Largest over all periods: raw {100 * max(c['raw'] for c in result['comparison']):.2f}%, scaled {100 * max(c['scaled'] for c in result['comparison']):.2f}%. The approximation is not written to the graph file.")
    # Splits
    split = [(k, x) for k, x in enumerate(result["sets"]) if x["variant"] > 0]
    # Say whether any edge set was split
    print(f"\nweighted edge sets: {len(body['graphs'])} from {len(result['timeline']['graphs'])} source edge sets; " + ("periods that share an edge set carry the same weights, so no edge set was split and the numbering is unchanged" if not split else "split edge sets (same connections, different weights): " + "; ".join(f"source {x['source']} variant {x['variant']} -> set {k} for periods {x['periods']}" for k, x in split)))
    # Weights by edge set
    print("\nweights by edge set (raw = magnitude of the reduced per-unit entry, scaled = raw divided by the set's largest raw weight)")
    # Header
    print(f"  {'set':>3} {'edges':>6} {'same node':>9} {'raw min':>8} {'raw median':>10} {'raw max':>8} {'scaled min':>10} {'scaled median':>13} {'below 0.1':>9} {'below 0.01':>10} {'below small':>11}")
    # Each
    for gph in body["graphs"]:
        # Raw weights that are numbers
        raw = [e["weight_raw_pu"] for e in gph["edges"] if e["weight_raw_pu"] is not None]
        # Scaled weights
        sc = [e["admittance_abs"] for e in gph["edges"]]
        # Line
        print(f"  {gph['index']:>3} {len(sc):>6} {sum(1 for e in gph['edges'] if e['same_node']):>9} {min(raw):>8.3f} {np.median(raw):>10.3f} {max(raw):>8.3f} {min(sc):>10.4f} {np.median(sc):>13.4f} {sum(1 for s in sc if s < 0.1):>9} {sum(1 for s in sc if s < 0.01):>10} {sum(1 for s in sc if s < args.small_weight):>11}")
    # Small weights, named
    small = [f"set {r['graph']}: {r['meter_a']} - {r['meter_b']} {r['weight_scaled']:.2e}" for r in result["rows"] if r["weight_scaled"] < args.small_weight]
    # Say
    print(f"edges with a scaled weight below {args.small_weight}: " + ("; ".join(small) if small else "none"))
    # Transformers on the path, reference edge set
    print("\nreference edge set (set 0): weights by the number of transformers on the edge's recorded path")
    # Header
    print(f"  {'transformers':>12} {'edges':>6} {'raw min':>8} {'raw median':>10} {'raw max':>8}")
    # Rows of set 0 with a number
    ref = [r for r in result["rows"] if r["graph"] == 0 and r["weight_raw_pu"] is not None]
    # Each count
    for k in sorted({r["transformers_on_path"] for r in ref}):
        # Raw weights of that group
        w = [r["weight_raw_pu"] for r in ref if r["transformers_on_path"] == k]
        # Line
        print(f"  {k:>12} {len(w):>6} {min(w):>8.3f} {np.median(w):>10.3f} {max(w):>8.3f}")
    # GCN coefficients, every edge set
    print("\nGCN coefficients under uniform weights (the current graph) and under electrical weights, every edge set")
    # What they are
    print("  self-loop coefficient = 1 / d_i. Sum of neighbor coefficients = the sum of w_ij / sqrt(d_i d_j) over a meter's edges. These are coefficients of the GCN operator, not shares: they do not add up to 1.")
    # What they do not say
    print("  Under the specified weighted GCN operator, many meters have larger self-loop coefficients and smaller sums of neighbor coefficients than under uniform weights. This does not measure the trained model's total dependence on neighbors:")
    # Second line
    print("  the model also has a residual connection and other paths for a meter's own information, which these coefficients leave out.")
    # Header
    print(f"  {'set':>3} {'meters with edges':>17} {'self-loop median':>24} {'self-loop larger':>16} {'neighbor sum median':>27} {'neighbor sum smaller':>20}")
    # Second header line
    print(f"  {'':>3} {'':>17} {'uniform':>11} {'electrical':>12} {'(meters)':>16} {'uniform':>13} {'electrical':>13} {'(meters)':>20}")
    # Per set, the per-meter coefficients are kept for the table of set 0
    per_set = []
    # Each
    for gph in body["graphs"]:
        # Node pairs
        pairs = [(e["source"], e["target"]) for e in gph["edges"]]
        # Uniform coefficients
        _, su, nu = ew.gcn_coefficients(len(names), pairs, [1.0] * len(pairs))
        # Electrical coefficients
        _, sw, nw = ew.gcn_coefficients(len(names), pairs, [e["admittance_abs"] for e in gph["edges"]])
        # Each meter's edges
        mine = [[k for k, (a, b) in enumerate(pairs) if i in (a, b)] for i in range(len(names))]
        # Sums of neighbor coefficients per meter
        tu, tw = [sum(nu[k] for k in m) for m in mine], [sum(nw[k] for k in m) for m in mine]
        # Meters with at least one edge
        on = [i for i in range(len(names)) if mine[i]]
        # Line
        print(f"  {gph['index']:>3} {len(on):>17} {np.median([su[i] for i in on]):>11.3f} {np.median([sw[i] for i in on]):>12.3f} {sum(1 for i in on if sw[i] > su[i] + 1e-12):>16} {np.median([tu[i] for i in on]):>13.3f} {np.median([tw[i] for i in on]):>13.3f} {sum(1 for i in on if tw[i] < tu[i] - 1e-12):>20}")
        # Keep
        per_set.append((pairs, mine, su, sw, tu, tw, nw))
    # Per meter, reference edge set
    print("\nreference edge set (set 0), per meter")
    # Header
    print(f"  {'meter':<10} {'edges':>5} {'self-loop uniform':>17} {'self-loop electrical':>20} {'neighbor sum uniform':>20} {'neighbor sum electrical':>23} {'largest neighbor coefficient (electrical)':<42} {'below 0.01':>10}")
    # The reference set's coefficients
    pairs, mine, su, sw, tu, tw, nw = per_set[0]
    # Each meter
    for i, m in enumerate(names):
        # No edge
        if not mine[i]:
            # Line
            print(f"  {m:<10} {0:>5} {su[i]:>17.3f} {sw[i]:>20.3f} {0.0:>20.3f} {0.0:>23.3f} {'no neighbor':<42} {0:>10}")
            # Next
            continue
        # Strongest neighbor under electrical weights
        best = max(mine[i], key=lambda k: nw[k])
        # The neighbor's name
        other = names[pairs[best][1] if pairs[best][0] == i else pairs[best][0]]
        # Line
        print(f"  {m:<10} {len(mine[i]):>5} {su[i]:>17.3f} {sw[i]:>20.3f} {tu[i]:>20.3f} {tw[i]:>23.3f} {other + ' ' + format(nw[best], '.3f'):<42} {sum(1 for k in mine[i] if nw[k] < 0.01):>10}")
    # Scaling across edge sets
    tops = [gph["weight_scale_pu"] for gph in body["graphs"]]
    # Say whether a global maximum would differ from the per-set maximum
    print("\nscaling across edge sets: largest raw weight per set " + ", ".join(f"{t:.3f}" for t in tops) + ". " + ("They are equal, so scaling by each set's own maximum and scaling by one global maximum give the same weights." if max(tops) - min(tops) <= 1e-9 * max(tops) else f"They differ, so a global maximum ({max(tops):.3f}) would give different weights in the sets below it."))


def main():
    """Refuse unsafe outputs, build the weighted graph, print the report, write the files."""
    # Options
    args = parse_args()
    # Every file the construction reads
    inputs = [args.graph, args.timeline, args.network] + switch_files(args.topology_dir)
    # Refuse before any work when an output is unsafe; a dry run writes nothing, so it has nothing to refuse
    if not args.dry_run:
        # Stop with a message, or continue
        refuse_outputs(args, inputs)
    # Build, stopping with a message when a check fails
    result = build(args)
    # Report
    print_report(result, args)
    # Compute and report only
    if args.dry_run:
        # Say so
        print("\ndry run: nothing written")
        # Done
        return
    # The graph file's text, complete in memory before anything is created, in the layout and indentation of the timeline file
    text = json.dumps(result["body"], indent=1)
    # Check the outputs again: the computation took time
    refuse_outputs(args, inputs)
    # Create the graph file; mode "x" fails when the file exists, so nothing can be written over
    with open(args.out, "x") as f:
        # Write
        f.write(text)
    # Create the edge table the same way
    with open(args.report, "x", newline="") as f:
        # Columns from the first row
        writer = csv.DictWriter(f, fieldnames=list(result["rows"][0].keys()))
        # Header
        writer.writeheader()
        # Rows
        writer.writerows(result["rows"])
    # Say what was written, with the identity the training script will record
    print(f"\nwrote {args.out} ({file_identity(args.out)}) and {args.report} ({len(result['rows'])} rows)")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
