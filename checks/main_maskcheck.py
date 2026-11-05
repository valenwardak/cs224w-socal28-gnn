# =============================================================================
# File:        main_maskcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-05
#
# Purpose:     One-off check that a masked run's model really forecasts a
#              hidden meter from its neighbors and not from its own history.
#              Loads the run's options and best weights, builds synthetic
#              input windows (random values with every validity flag on, so
#              no data files are needed), and tests, in evaluation mode:
#                0. the graph file given is the one the run was trained on
#                   (name and content hash recorded in the checkpoint), so
#                   the graph buffers rebuilt here match the saved weights
#                1. hiding a meter and then changing that meter's original
#                   history leaves its masked forecast unchanged: no shortcut
#                   carries the hidden history to the head
#                2. changing a visible neighbor's history changes the hidden
#                   meter's forecast: the neighbors reach it. Reported as a
#                   NOTE, not a failure, when nothing changes: that is a
#                   finding about the trained weights, not a code error
#                3. changing a meter that the saved edges do not connect to
#                   the hidden meter leaves its forecast unchanged
#                4. the non-maskable meters (egauge_24 on the meter graph) are
#                   flagged invalid by masked_forecast
#                5. masked_forecast agrees with one plain masked pass per
#                   meter: the chunked batching changes nothing
#              Neighbors and connectivity come from the edges the model
#              saved (after --edges true, none, or random was applied), not
#              from the graph file, so the expectations hold for the
#              controls too: with no edges, no other meter may influence the
#              hidden one; with random edges, influence follows the drawn
#              graph. Prints PASS, FAIL, or NOTE per check and exits with 1
#              on any failure. Checks 2 and 3 are about trained weights, so
#              run this on a trained run; check 1 holds for any weights.
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root.
#
# Usage:
#   python checks/main_maskcheck.py --run-dir data/processed/models/smoke_masked --graph graph_meters.json --manifest data/features/manifest.json
#   python checks/main_maskcheck.py --run-dir /content/drive/MyDrive/socal_runs/gcn_masked --graph graph_meters.json --manifest /content/features/manifest.json
#
# Functions:
#   parse_args()            command-line options with defaults
#   report(name, ok, note)  print one PASS or FAIL line, return ok
#   graph_identity(path)    the graph file's name and content hash, as main_gcn.py records it
#   load_run(args)          options, graph, manifest, model with the best weights, after the identity check
#   synthetic(B, H, M, C, seed)
#                           a random input window with every validity flag on
#   model_neighbors(model, M)
#                           meter index -> meter indices one saved edge away (or on the same node)
#   model_components(model, M)
#                           meter index -> component id under the saved edges
#   main()                  run the checks
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: content hash of the graph file
import hashlib
# Standard library: JSON reading
import json
# Standard library: repo root on the import path, exit code
import os
import sys
# Third party: deep learning
import torch

# The repo root, one level above checks/, so the project modules import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project: models
import socal_models as sm
# Project: data constants (feature count)
import socal_modeldata as sd


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Check that a masked run forecasts hidden meters from their neighbors")
    # Run folder
    parser.add_argument("--run-dir", required=True, help="Run folder holding checkpoint.pt and checkpoint_best.pt")
    # Graph file the run was trained on
    parser.add_argument("--graph", default="graph_meters.json", help="Graph file the run used")
    # Manifest for meter and channel order
    parser.add_argument("--manifest", default=os.path.join("data", "features", "manifest.json"), help="features manifest.json")
    # Window size
    parser.add_argument("--batch", type=int, default=4, help="Synthetic windows per check")
    # Tolerance
    parser.add_argument("--tol", type=float, default=1e-6, help="Largest change that still counts as unchanged")
    # Parse
    return parser.parse_args()


def report(name, ok, note=""):
    """Print one PASS or FAIL line and return ok."""
    # Line
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + note if note else ''}")
    # Pass through
    return ok


def graph_identity(path):
    """The graph file's name and the first 16 hex digits of the SHA-256 of its bytes, as main_gcn.py records it."""
    # Hash
    with open(path, "rb") as f:
        # Digest
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    # Name and digest
    return f"{os.path.basename(path)}:{digest}"


def load_run(args):
    """Options from the checkpoint, the graph, the manifest, and the model with its best weights, after the graph identity check."""
    # Checkpoint
    body = torch.load(os.path.join(args.run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
    # Options
    options = body["options"]
    # Must be a masked run
    if not options.get("masked"):
        # Raise
        raise ValueError("this run was not trained with --masked")
    # The graph file must be the one the run used, else the rebuilt buffers would not match the weights
    if options.get("graph_id") != graph_identity(args.graph):
        # Raise
        raise ValueError(f"graph identity mismatch: run has {options.get('graph_id')}, {args.graph} is {graph_identity(args.graph)}")
    # Say so
    print(f"PASS  0. graph file matches the run: {options['graph_id']}")
    # Graph
    graph = sm.load_graph(args.graph)
    # Manifest
    with open(args.manifest) as f:
        # Parse
        manifest = json.load(f)
    # Model as the run built it
    model = sm.make_model(options["kind"], manifest, graph, edges=options["edges"], edge_weights=options["edge_weights"], hidden=options["hidden"], embed=options["embed"], layers=options["layers"], dropout=options["dropout"], seed=options["seed"], gnn_layers=options["gnn_layers"], predict_delta=options["predict_delta"], masked=True, maskable_meters=options["maskable_meters"])
    # Best weights
    model.load_state_dict(torch.load(os.path.join(args.run_dir, "checkpoint_best.pt"), map_location="cpu"))
    # Evaluation mode
    model.eval()
    # Done
    return options, graph, manifest, model


def synthetic(B, H, M, C, seed):
    """A random input window (B, H, M, C * F): magnitudes near 1, unit (cos, sin), every validity flag on."""
    # Generator
    g = torch.Generator().manual_seed(seed)
    # Magnitude around the scaled level
    mag = 1.0 + 0.1 * torch.randn(B, H, M, C, generator=g)
    # Angles
    theta = 2 * torch.pi * torch.rand(B, H, M, C, generator=g)
    # Features: mag, cos, sin, mag_valid, ang_valid
    x = torch.stack([mag, torch.cos(theta), torch.sin(theta), torch.ones_like(mag), torch.ones_like(mag)], dim=-1)
    # Flatten channels and features
    return x.reshape(B, H, M, C * sd.F)


def model_neighbors(model, M):
    """
    Meter index -> sorted list of meter indices that can reach it in one
    message-passing hop under the model's saved edges: meters on a node
    joined by a saved edge, or on the same node (their states are averaged
    before message passing).
    """
    # Node per meter
    node_of = model.meter_node.tolist()
    # Saved edges, both directions already stored
    src, dst = model.edge_index.tolist()
    # Nodes joined to each node
    joined = {}
    # Loop
    for a, b in zip(src, dst):
        # Add
        joined.setdefault(a, set()).add(b)
    # Result
    out = {}
    # Loop over meters
    for i in range(M):
        # Nodes that reach meter i's node: itself (shared node) and its saved neighbors
        reach = {node_of[i]} | joined.get(node_of[i], set())
        # Meters on those nodes, not i
        out[i] = sorted(j for j in range(M) if j != i and node_of[j] in reach)
    # Done
    return out


def model_components(model, M):
    """Meter index -> component id under the model's saved edges (breadth-first over nodes)."""
    # Node per meter
    node_of = model.meter_node.tolist()
    # Saved edges
    src, dst = model.edge_index.tolist()
    # Adjacency
    adj = {}
    # Loop
    for a, b in zip(src, dst):
        # Add
        adj.setdefault(a, set()).add(b)
    # Component per node
    comp = {}
    # Next id
    next_id = 0
    # Loop over every node that carries a meter or an edge
    for start in sorted(set(node_of) | set(adj)):
        # Seen
        if start in comp:
            # Next
            continue
        # Flood
        stack = [start]
        # Mark
        comp[start] = next_id
        # Loop
        while stack:
            # Pop
            u = stack.pop()
            # Neighbors
            for v in adj.get(u, ()):
                # New
                if v not in comp:
                    # Mark
                    comp[v] = next_id
                    # Push
                    stack.append(v)
        # Next component
        next_id += 1
    # Per meter
    return {i: comp[node_of[i]] for i in range(M)}


def main():
    """Run the checks."""
    # Options
    args = parse_args()
    # Load
    options, graph, manifest, model = load_run(args)
    # Names
    meters = manifest["meters"]
    # Sizes
    M, C, H = len(meters), len(manifest["channels"]), options["history"]
    # Maskable meters
    maskable = options["maskable_meters"]
    # Reach under the saved edges
    nbrs = model_neighbors(model, M)
    # Components under the saved edges
    comps = model_components(model, M)
    # Hidden meter for the tests: the first maskable meter with a neighbor under the saved edges, else the first maskable meter
    hi = next((meters.index(m) for m in maskable if nbrs[meters.index(m)]), meters.index(maskable[0]))
    # Its name
    hidden = meters[hi]
    # A neighbor under the saved edges, if any
    ni = nbrs[hi][0] if nbrs[hi] else None
    # Meters the saved edges do not connect to the hidden one
    others = [i for i in range(M) if i != hi and comps[i] != comps[hi]]
    # Report what is used
    print(f"edges={options['edges']}: hidden meter {hidden}, neighbor {meters[ni] if ni is not None else 'none under the saved edges'}, unconnected meter {meters[others[0]] if others else 'none'}")
    # Every check's result
    ok = True
    # Mask that hides the chosen meter in every window
    mask = torch.zeros(args.batch, M, dtype=torch.bool)
    # Set
    mask[:, hi] = True
    # Base input
    x = synthetic(args.batch, H, M, C, 0)
    # No gradient
    with torch.no_grad():
        # Base forecast of the hidden meter
        base = model(x, mask)[:, hi]

        # 1. change the hidden meter's own history
        x1 = x.clone()
        # Replace its features with a different draw
        x1[:, :, hi] = synthetic(args.batch, H, M, C, 1)[:, :, hi]
        # Forecast
        y1 = model(x1, mask)[:, hi]
        # Difference
        d1 = float((y1 - base).abs().max())
        # Report
        ok &= report("1. hidden meter's own history does not reach its forecast", d1 <= args.tol, f"max change {d1:.2e}")

        # 2. change a neighbor's history; a finding about the weights, so a NOTE rather than a failure when nothing moves
        if ni is not None:
            # Copy
            x2 = x.clone()
            # Replace
            x2[:, :, ni] = synthetic(args.batch, H, M, C, 2)[:, :, ni]
            # Forecast
            y2 = model(x2, mask)[:, hi]
            # Difference
            d2 = float((y2 - base).abs().max())
            # Report
            if d2 > args.tol:
                # Reaches
                report("2. a visible neighbor's history changes the hidden meter's forecast", True, f"max change {d2:.2e}")
            else:
                # Does not, which is a finding about the trained weights
                print(f"NOTE  2. the neighbor's history did not change the hidden meter's forecast (max change {d2:.2e}): the trained weights are not using it")
        else:
            # No neighbor under these edges
            print("NOTE  2. no neighbor under the saved edges, nothing to test")

        # 3. change a meter the saved edges do not connect to the hidden one
        if others:
            # Index
            oi = others[0]
            # Copy
            x3 = x.clone()
            # Replace
            x3[:, :, oi] = synthetic(args.batch, H, M, C, 3)[:, :, oi]
            # Forecast
            y3 = model(x3, mask)[:, hi]
            # Difference
            d3 = float((y3 - base).abs().max())
            # Report
            ok &= report("3. an unconnected meter does not reach the hidden meter's forecast", d3 <= args.tol, f"max change {d3:.2e}")
        else:
            # Nothing to test
            report("3. unconnected-meter check", True, "skipped, every meter is connected under the saved edges")

        # 4. non-maskable meters flagged invalid
        y_all, keep = sm.masked_forecast(model, x, options.get("mask_chunk", 7))
        # Names flagged invalid
        invalid = [meters[i] for i in range(M) if not bool(keep[i])]
        # Expected: every meter not in the maskable list
        expected = [m for m in meters if m not in maskable]
        # Report
        ok &= report("4. non-maskable meters are flagged invalid", invalid == expected, f"invalid: {invalid}")

        # 5. chunked batching agrees with one pass per meter
        worst = 0.0
        # Loop over maskable meters
        for m in maskable:
            # Index
            mi = meters.index(m)
            # Mask hiding only this meter
            one = torch.zeros(args.batch, M, dtype=torch.bool)
            # Set
            one[:, mi] = True
            # Plain pass
            y_one = model(x, one)[:, mi]
            # Compare with the chunked result
            worst = max(worst, float((y_one - y_all[:, mi]).abs().max()))
        # Report
        ok &= report("5. chunked masked_forecast matches one pass per meter", worst <= 1e-5, f"max difference {worst:.2e}")
    # Exit code
    sys.exit(0 if ok else 1)


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
