# =============================================================================
# File:        main_sendergate.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Tests of sender gating before any model is trained with it.
#
#              Why it exists: the course staff's milestone feedback asked
#              whether nodes without readings hurt the graph model. The
#              diagnostics showed that meters without usable history stay
#              connected and keep sending messages. Sender gating is the
#              first alternative: such a meter sends no message in that
#              window, keeps its self-loop and still receives. A comparison
#              of trained models would mean little if the new rule were
#              wired wrongly, so this check runs the rule on small
#              fixtures and on the real graph first. Nothing is trained,
#              scored or written to a run folder.
#
#              The cases, with the names the reviewers agreed:
#                R  the availability rule: a hand-built fixture; the torch
#                   rule against its NumPy copy and against the pipeline's
#                   own rule (socal_modeldata.window_starts with the target
#                   condition off) on a synthetic piece built with the
#                   pipeline's build_features; optionally the same on real
#                   feature months (--compare-months).
#                T1 all available: the gated model equals the ungated
#                   model on every edge set of the real graph.
#                T2 an unavailable sender is silenced: changing its history
#                   changes no other meter, while the same change does move
#                   a neighbor in the ungated model, and an available
#                   sender still moves its neighbors, the unavailable
#                   receiver included.
#                T3 deletion equivalence: the gated model equals the
#                   ungated model run on an edge list from which only the
#                   outgoing directed edges of the unavailable sender are
#                   removed. Removing its incoming edges too gives a
#                   different result, which shows the distinction matters.
#                T4 masked mode: a hidden meter keeps its incoming
#                   messages, its own readings leak nowhere, and it is
#                   silenced as a sender.
#                T5 batch isolation with gating: stacked examples do not
#                   affect one another, and gating in one example leaves
#                   the other examples as they were.
#                T6 degrees: the weights the GCN layer normalizes equal the
#                   hand formula a_j / sqrt(d_i * d_j) with
#                   d_i = 1 + sum of a_j over the neighbors of i, and the
#                   layer's output equals a dense calculation of that
#                   formula. The fixture's stored weights are all 1.
#                B  boundaries of the same operator: a window in which
#                   every meter is unavailable equals the model with no
#                   edges; a model built with no edges runs with gating on
#                   and equals the ungated one; stored weights that are not
#                   1 keep their value for available senders, in the
#                   normalized weights and in the deletion equivalence.
#                C  compatibility: the option adds no parameter or buffer;
#                   the default model equals the model of a reference
#                   commit from before the change, weights and forecasts;
#                   weights saved by the reference code load into the new
#                   code; a checkpoint without the option is read as
#                   sender_gate False and refused for a gated run.
#
#              The project starts the forecast heads at zero, so an
#              untrained model's forecast is persistence and responds to
#              nothing. As in checks/main_graphdelivery.py, every model here
#              gets nonzero test heads, the test inputs are all-valid
#              synthetic inputs unless a case removes readings on purpose,
#              and the readout is the learned correction (the forecast
#              minus the persistence base). Masked models forecast the
#              value directly and already start with nonzero heads.
#
#              T4 tests the existing masked mode, which works on a fixed
#              graph only. Masked training on the time-indexed graph is not
#              part of this change.
#
#              Period selection at the switching boundaries belongs to the
#              dataset class, which sender gating does not touch. It stays
#              covered by checks/main_graphdelivery.py.
#
#              Status: syntax-checked only. The parts that need torch have
#              not been executed. The NumPy copy of the rule was executed
#              against the pipeline's rule. The first Colab run of this
#              file is the first execution of every torch case.
#
# Usage (from the repo root):
#   python checks/main_sendergate.py --features-dir /content/features
#   python checks/main_sendergate.py --features-dir /content/features --compare-months 2024-10 2025-02
#   python checks/main_sendergate.py --features-dir /content/features --reference-commit 098cdc0
#
# Functions:
#   parse_args()                     command-line options with defaults
#   fixture_graph(weights)           five-meter fixed graph a - b - c - d and e alone, two channels
#   remove_start(x, b, m, max_gap)   make meter m unavailable in example b: no readings in the first max_gap slots
#   redraw_kept_gap(x, b, m, C, max_gap)
#                                    new values for an unavailable meter that keep its gap and its last slot
#   masked_forecasts(model, x, mask) forecast of a masked model, no gradients
#   without_outgoing(model, node)    replace a fixed-graph model's edges by the list without the node's outgoing edges
#   without_all(model, node)         replace them by the list without any edge touching the node
#   without_edges(model)             replace them by an empty edge list
#   load_reference(commit)           the model code of an earlier commit as a separate module
#   pipeline_available(present_any, history, max_gap)
#                                    the pipeline's history-only rule for every window start of one meter
#   case_rule(args, manifest)        case R
#   case_all_available(...)          case T1
#   case_silenced(args)              cases T2 and T3
#   case_masked(args)                case T4
#   case_batch(...)                  case T5
#   case_degrees(args)               case T6
#   case_boundaries(args, manifest, timeline, x)
#                                    case B
#   case_compatibility(...)          case C
#   main()                           run every case, exit 0 when all pass
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: load the reference model code as a module
import importlib.util
# Standard library: paths
import os
# Standard library: read the reference model code from git
import subprocess
# Standard library: exit code
import sys
# Standard library: folder for the reference module and the test checkpoint
import tempfile
# Third party: numerics
import numpy as np
# Third party: tensors
import torch

# The folder of this file, where the delivery check lives
CHECKS = os.path.dirname(os.path.abspath(__file__))
# The repo root, one level above checks/
ROOT = os.path.dirname(CHECKS)
# Import the project modules from the root
sys.path.insert(0, ROOT)
# Import the delivery check's helpers from checks/
sys.path.insert(0, CHECKS)

# Project: model, graph loading, checkpoints
import socal_models as sm
# Project: features and the pipeline's window rule
import socal_modeldata as sd
# Project: the sender gating rule under test
import socal_sendergate as sg
# Project: the delivery check's helpers (report, nonzero test heads, learned correction, all-valid inputs), already run on Colab
import main_graphdelivery as gd


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Tests of sender gating on fixtures and on the real graph; nothing is trained")
    # Graph
    parser.add_argument("--graph", default="graph_meters_periods.json", help="Time-indexed meter graph")
    # Manifest source and, for --compare-months, the feature months
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with manifest.json and the feature months")
    # History
    parser.add_argument("--history", type=int, default=60, help="History slots, as trained")
    # Gap rule
    parser.add_argument("--max-gap", type=int, default=12, help="A run of this many absent slots in the history makes a meter unavailable, as trained")
    # Tolerance for "equal" and "unchanged", absolute
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Largest absolute difference that counts as equal")
    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random weights and inputs")
    # Draws
    parser.add_argument("--draws", type=int, default=5, help="Random initializations tried for the fixture cases")
    # Reference commit for the compatibility case
    parser.add_argument("--reference-commit", default="098cdc0", help="Commit whose socal_models.py is the model code from before sender gating")
    # Real months for the rule comparison
    parser.add_argument("--compare-months", nargs="*", default=[], help="Feature months YYYY-MM on which the rule is also compared with the pipeline's rule; none by default")
    # Windows per pass in the real-month comparison
    parser.add_argument("--chunk", type=int, default=5000, help="Windows per pass in the real-month comparison")
    # Parse
    return parser.parse_args()


def fixture_graph(weights=(1.0, 1.0, 1.0)):
    """
    A fixed graph for the fixture cases: five meters, a - b - c - d in a
    path and e with no edge, two channels per meter so the any-channel
    presence rule is exercised. weights are the stored weights of the
    edges a - b, b - c and c - d, all 1 unless a case asks otherwise.
    Returns (manifest, graph) in the layouts of manifest.json and
    graph.json.
    """
    # Meter names in node order
    names = ["a", "b", "c", "d", "e"]
    # Manifest with one voltage and one current channel
    manifest = {"meters": names, "channels": ["L1", "S1"]}
    # Graph: one node per meter, three edges in a path, e alone
    graph = {"nodes": [{"meter": n} for n in names], "meters": {n: {"node": i} for i, n in enumerate(names)},
             "edges": [{"source": 0, "target": 1, "admittance_abs": float(weights[0])}, {"source": 1, "target": 2, "admittance_abs": float(weights[1])}, {"source": 2, "target": 3, "admittance_abs": float(weights[2])}]}
    # Both
    return manifest, graph


def remove_start(x, b, m, max_gap):
    """
    Make meter m unavailable in example b: zero every feature of every
    channel in its first max_gap history slots, as a real outage looks in
    the inputs. The last slot keeps its reading, so the persistence base
    is unchanged. Returns a new tensor.
    """
    # Copy
    out = x.clone()
    # No readings in the first max_gap slots: values and validity flags all zero
    out[b, :max_gap, m, :] = 0.0
    # Done
    return out


def redraw_kept_gap(x, b, m, C, max_gap):
    """
    New random values for an unavailable meter m in example b that keep
    the meter unavailable: the first max_gap slots stay empty and the last
    slot stays as it was, so only the slots between them change.
    Returns a new tensor.
    """
    # Copy
    out = x.clone()
    # Redraw the slots after the gap and before the last slot, with valid flags
    out[b, max_gap:-1, m, :] = gd.redraw(out[b, max_gap:-1, m, :], C)
    # Done
    return out


def masked_forecasts(model, x, mask):
    """The forecast (B, M, C, 3) of a masked model for inputs x with the meters in mask hidden, no gradients."""
    # No gradients
    with torch.no_grad():
        # Forward on the fixed graph
        return model(x, mask=mask)


def without_outgoing(model, node):
    """
    Replace the edges of a fixed-graph model by the same list without the
    outgoing directed edges of one node. Edges into the node are kept.
    Returns the number of directed edges removed.
    """
    # The model's edge list
    ei, ew = model.edge_set(0)
    # Keep every edge whose sender is not the node
    keep = ei[0] != node
    # Store the shorter list in the model's buffers
    model.edge_index, model.edge_weight = ei[:, keep].clone(), ew[keep].clone()
    # Removed count
    return int((~keep).sum())


def without_all(model, node):
    """
    Replace the edges of a fixed-graph model by the same list without any
    edge that touches one node, in either direction. Returns the number
    of directed edges removed.
    """
    # The model's edge list
    ei, ew = model.edge_set(0)
    # Keep every edge that does not touch the node
    keep = (ei[0] != node) & (ei[1] != node)
    # Store the shorter list in the model's buffers
    model.edge_index, model.edge_weight = ei[:, keep].clone(), ew[keep].clone()
    # Removed count
    return int((~keep).sum())


def without_edges(model):
    """Replace the edges of a fixed-graph model by an empty list, so every node sees only its own self-loop. Returns the number of directed edges removed."""
    # The model's edge list
    ei, ew = model.edge_set(0)
    # Store an empty list of the same kind in the model's buffers
    model.edge_index, model.edge_weight = ei[:, :0].clone(), ew[:0].clone()
    # Removed count
    return int(ei.shape[1])


def load_reference(commit):
    """
    The model code of an earlier commit, loaded as a separate module so it
    can be compared with the current code. Reads socal_models.py of that
    commit from git. Returns (module, None), or (None, reason) when git
    cannot provide it.
    """
    # Ask git for the file as it was at the commit
    out = subprocess.run(["git", "-C", ROOT, "show", f"{commit}:socal_models.py"], capture_output=True, text=True)
    # Git could not provide it
    if out.returncode != 0:
        # Reason
        return None, out.stderr.strip() or f"git show failed with code {out.returncode}"
    # A folder for the copy
    folder = tempfile.mkdtemp()
    # The copy, under another module name
    path = os.path.join(folder, "socal_models_reference.py")
    # Write
    with open(path, "w") as f:
        # Text
        f.write(out.stdout)
    # Load it as a module of its own
    spec = importlib.util.spec_from_file_location("socal_models_reference", path)
    # Module
    module = importlib.util.module_from_spec(spec)
    # Run its top level
    spec.loader.exec_module(module)
    # Done
    return module, None


def pipeline_available(present_any, history, max_gap):
    """
    The pipeline's history-only rule for one meter: for every window start
    s in 0 .. T - history - 1, whether the history s .. s + history - 1
    holds no run of max_gap absent slots. It calls the pipeline's own
    window_starts with the target condition switched off (every target
    counted as valid). Returns a bool array (T - history,).
    """
    # Number of window starts for horizon 1
    n = len(present_any) - history
    # Starts the pipeline keeps when every target counts as valid
    starts = sd.window_starts(present_any, history, 1, max_gap, np.ones(len(present_any), dtype=bool))
    # As flags per start
    out = np.zeros(max(n, 0), dtype=bool)
    # Mark the kept starts
    out[starts] = True
    # Done
    return out


def case_rule(args, manifest):
    """Case R: the availability rule on a hand fixture, against NumPy and the pipeline on a synthetic piece, and optionally on real months."""
    # All pass so far
    ok = True
    # Sizes of the hand fixture: one window, four meters, two channels
    H, G, C, F = args.history, args.max_gap, 2, sd.n_features()
    # All-valid input
    x = gd.draw_inputs((1, H, 4, C * F), C)
    # Split into channels and features to edit the flags
    xf = x.reshape(1, H, 4, C, F).clone()
    # Meter 0: every angle flag off, magnitude flags on; angle validity must not matter
    xf[0, :, 0, :, 4] = 0.0
    # Meter 1: a run of exactly max_gap absent slots in the middle
    xf[0, 5:5 + G, 1, :, :] = 0.0
    # Meter 2: a run one slot shorter than max_gap
    xf[0, 5:5 + G - 1, 2, :, :] = 0.0
    # Meter 3: one channel absent throughout, the other present; one valid channel is enough
    xf[0, :, 3, 0, :] = 0.0
    # Back to the input layout
    x = xf.reshape(1, H, 4, C * F)
    # The rule
    got = sg.history_available(x, C, G)[0].tolist()
    # Expected: only meter 1 is unavailable
    ok &= gd.report("R hand fixture: angle validity ignored, a run of max_gap absent slots is unavailable, one slot shorter is available, one valid channel is enough", got == [True, False, True, True], f"available {got}")
    # A history shorter than the gap can never hold the run
    short = sg.available_from_presence(torch.zeros(1, G - 1, 2, dtype=torch.bool), G)
    # Every meter available
    ok &= gd.report("R a history shorter than max_gap is always available, as in the pipeline", bool(short.all()))
    # Synthetic piece: T slots, M meters, readings with gaps of many lengths around max_gap
    rng = np.random.default_rng(args.seed)
    # Sizes
    T, M = 6 * H, 6
    # Start with every reading valid
    mag_valid = np.ones((T, M, C), dtype=bool)
    # Each meter
    for mi in range(M):
        # Several gaps
        for _ in range(8):
            # Gap length from 1 to twice max_gap, and its start
            length, start = int(rng.integers(1, 2 * G + 1)), int(rng.integers(0, T))
            # Meter mi loses both channels there, except meter 0, which loses one channel only
            mag_valid[start:start + length, mi, :(1 if mi == 0 else C)] = False
    # Arrays in the layout build_features reads
    arrays = {"rms": rng.random((T, M, C)).astype(np.float32) * mag_valid, "ang": np.zeros((T, M, C), dtype=np.float32), "mag_valid": mag_valid, "ang_valid": mag_valid.copy()}
    # Model inputs from the pipeline's own function
    X, _, _ = sd.build_features(arrays, np.ones((M, C), dtype=np.float32))
    # Every window start
    n = T - H
    # Windows as a batch: (n, H, M, C * F)
    windows = torch.from_numpy(np.stack([X[s:s + H].reshape(H, M, -1) for s in range(n)]))
    # The torch rule on the inputs
    got_t = sg.history_available(windows, C, G).numpy()
    # The NumPy copy on the presence
    present = mag_valid.any(axis=2)
    # Windows of presence: (n, H, M)
    got_n = sg.available_from_presence_np(np.stack([present[s:s + H] for s in range(n)]), G)
    # The pipeline's rule per meter: (n, M)
    want = np.stack([pipeline_available(present[:, mi], H, G) for mi in range(M)], axis=1)
    # Both outcomes must occur, or the comparison shows little
    mixed = bool(want.any() and (~want).any())
    # Check
    ok &= gd.report("R synthetic piece: the torch rule on the model inputs equals the pipeline's history-only rule for every window and meter", bool(np.array_equal(got_t, want)) and mixed, f"{n * M} window-meter pairs, {int((~want).sum())} unavailable, mismatches {int((got_t != want).sum())}")
    # NumPy copy
    ok &= gd.report("R synthetic piece: the NumPy copy of the rule equals the pipeline's rule", bool(np.array_equal(got_n, want)), f"mismatches {int((got_n != want).sum())}")
    # Real months, when asked for
    for month in args.compare_months:
        # The month's arrays through the pipeline's loader
        arrays = sd.load_month(args.features_dir, month, manifest=manifest)
        # Presence per slot and meter
        present = arrays["mag_valid"].any(axis=2)
        # Sizes
        T, M = present.shape
        # Window starts
        n = T - H
        # The pipeline's rule per meter
        want = np.stack([pipeline_available(present[:, mi], H, G) for mi in range(M)], axis=1)
        # Presence as a tensor
        p = torch.from_numpy(present)
        # Every stretch of H slots as a view: (T - H + 1, M, H)
        view = p.unfold(0, H, 1)
        # Mismatch count
        bad = 0
        # Chunks of window starts
        for s in range(0, n, args.chunk):
            # Windows of this chunk as (windows, H, M)
            part = view[s:min(s + args.chunk, n)].permute(0, 2, 1)
            # The torch rule
            got = sg.available_from_presence(part, G).numpy()
            # Count mismatches
            bad += int((got != want[s:s + len(got)]).sum())
        # Check
        ok &= gd.report(f"R real month {month}: the torch rule equals the pipeline's history-only rule for every window and meter", bad == 0, f"{n * M} window-meter pairs, {int((~want).sum())} unavailable, mismatches {bad}")
    # Result
    return ok


def case_all_available(args, manifest, body, timeline, base, gated, x):
    """Case T1: with every sender available, the gated model equals the ungated model on every edge set."""
    # All pass so far
    ok = True
    # Largest difference over the edge sets
    worst = 0.0
    # Each edge set
    for g in range(len(body["graphs"])):
        # Graph ids: every example on edge set g
        gid = torch.full((x.shape[0],), g, dtype=torch.long)
        # Difference of the learned corrections
        worst = max(worst, gd.max_change(gd.correction(base, x, gid), gd.correction(gated, x, gid)))
    # Check
    ok &= gd.report(f"T1 all available: the gated model equals the ungated model on all {len(body['graphs'])} edge sets", worst <= args.tolerance, f"largest difference {worst:.2e}")
    # Result
    return ok


def case_silenced(args):
    """Cases T2 and T3 on the fixture a - b - c - d, e alone, with b unavailable."""
    # All pass so far
    ok = True
    # Fixture
    fman, fgraph = fixture_graph()
    # Sizes
    M, C, F, G = len(fman["meters"]), len(fman["channels"]), sd.n_features(), args.max_gap
    # Graph ids: None, since the fixture is a fixed graph with one edge set
    none = None
    # Counters over the draws
    silent, base_moves, avail_moves, recv_moves, deletion, differs = 0, 0, 0, 0, 0, 0
    # Largest values for the report
    worst_silent, worst_deletion = 0.0, 0.0
    # Draws
    for s in range(args.draws):
        # Gated model with nonzero test heads
        fm = gd.randomize_heads(sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed + s, sender_gate=True, gate_max_gap=G), args.seed + s)
        # Evaluation mode
        fm.eval()
        # Ungated model with the same weights and edges
        fb = sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed + s)
        # Same parameters and buffers
        fb.load_state_dict(fm.state_dict())
        # Evaluation mode
        fb.eval()
        # Inputs: all valid, then b (meter 1) made unavailable
        torch.manual_seed(args.seed + s)
        # Draw and remove b's first max_gap slots
        x = remove_start(gd.draw_inputs((1, args.history, M, C * F), C), 0, 1, G)
        # The rule must see exactly b as unavailable
        if sg.history_available(x, C, G)[0].tolist() != [True, False, True, True, True]:
            # Stop: the fixture is not what the cases assume
            raise SystemExit("fixture error: the rule does not see exactly meter b as unavailable")
        # Base corrections of both models
        yg, yb = gd.correction(fm, x, none), gd.correction(fb, x, none)
        # Change b's history but keep it unavailable
        xb = redraw_kept_gap(x, 0, 1, C, G)
        # Gated: no other meter may change
        ch = gd.max_change(yg[0, [0, 2, 3, 4]], gd.correction(fm, xb, none)[0, [0, 2, 3, 4]])
        # Record
        worst_silent = max(worst_silent, ch)
        # Count the draws where nothing changed
        silent += int(ch <= args.tolerance)
        # Ungated: the same change must move a neighbor, or the case above shows little
        base_moves += int(gd.max_change(yb[0, [0, 2]], gd.correction(fb, xb, none)[0, [0, 2]]) > args.tolerance)
        # Change an available sender, c (meter 2), keeping it available
        xc = x.clone()
        # New history for c with valid flags
        xc[0, :, 2, :] = gd.redraw(xc[0, :, 2, :], C)
        # Corrections after the change
        yc = gd.correction(fm, xc, none)
        # Its available neighbor d must respond
        avail_moves += int(gd.max_change(yg[0, 3], yc[0, 3]) > args.tolerance)
        # Its unavailable neighbor b must respond too: an unavailable receiver keeps its incoming messages
        recv_moves += int(gd.max_change(yg[0, 1], yc[0, 1]) > args.tolerance)
        # T3: the ungated model without b's outgoing directed edges
        fd = sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed + s)
        # Same weights
        fd.load_state_dict(fm.state_dict())
        # Evaluation mode
        fd.eval()
        # Remove b's outgoing edges only; it has two neighbors, so two directed edges go
        removed = without_outgoing(fd, 1)
        # The gated model must equal it on every meter
        d = gd.max_change(yg, gd.correction(fd, x, none))
        # Record
        worst_deletion = max(worst_deletion, d)
        # Count
        deletion += int(d <= args.tolerance and removed == 2)
        # Contrast: removing b's incoming edges too gives another result for b itself
        fa = sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed + s)
        # Same weights
        fa.load_state_dict(fm.state_dict())
        # Evaluation mode
        fa.eval()
        # Remove every edge touching b
        without_all(fa, 1)
        # b's correction must differ from the gated model's
        differs += int(gd.max_change(yg[0, 1], gd.correction(fa, x, none)[0, 1]) > args.tolerance)
    # T2 invariance, every draw
    ok &= gd.report(f"T2 silenced sender: changing the unavailable meter's history changed no other meter in {silent} of {args.draws} initializations (all required)", silent == args.draws, f"largest change {worst_silent:.2e}")
    # T2 positive control on the ungated model
    ok &= gd.report(f"T2 positive control: the same change moved a neighbor in the ungated model in {base_moves} of {args.draws} initializations (at least one required)", base_moves >= 1)
    # T2 available sender
    ok &= gd.report(f"T2 available sender: changing an available meter moved its available neighbor in {avail_moves} of {args.draws} initializations (at least one required)", avail_moves >= 1)
    # T2 unavailable receiver
    ok &= gd.report(f"T2 unavailable receiver keeps incoming messages: the same change moved the unavailable neighbor in {recv_moves} of {args.draws} initializations (at least one required)", recv_moves >= 1)
    # T3 equivalence, every draw
    ok &= gd.report(f"T3 deletion equivalence: the gated model equals the ungated model without the unavailable sender's outgoing edges in {deletion} of {args.draws} initializations (all required)", deletion == args.draws, f"largest difference {worst_deletion:.2e}")
    # T3 contrast
    ok &= gd.report(f"T3 contrast: removing the incoming edges too changed the unavailable meter's own result in {differs} of {args.draws} initializations (at least one required)", differs >= 1)
    # Result
    return ok


def case_masked(args):
    """Case T4: masked mode on the fixture with c hidden."""
    # All pass so far
    ok = True
    # Fixture
    fman, fgraph = fixture_graph()
    # Sizes
    M, C, F, G = len(fman["meters"]), len(fman["channels"]), sd.n_features(), args.max_gap
    # The meters with an edge may be hidden
    maskable = ["a", "b", "c", "d"]
    # Counters over the draws
    leak, incoming, as_deleted, fired = 0, 0, 0, 0
    # Largest values for the report
    worst_leak, worst_deleted = 0.0, 0.0
    # Draws
    for s in range(args.draws):
        # Gated masked model; masked models forecast the value directly and start with nonzero heads
        mm = sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed + s, masked=True, maskable_meters=maskable, sender_gate=True, gate_max_gap=G)
        # Evaluation mode
        mm.eval()
        # Inputs, all valid
        torch.manual_seed(args.seed + s)
        # Draw
        x = gd.draw_inputs((1, args.history, M, C * F), C)
        # Hide c (meter 2)
        mask = torch.zeros(1, M, dtype=torch.bool)
        # Set
        mask[0, 2] = True
        # Base forecast
        y = masked_forecasts(mm, x, mask)
        # Change the hidden meter's own readings
        xc = x.clone()
        # New history for c
        xc[0, :, 2, :] = gd.redraw(xc[0, :, 2, :], C)
        # No forecast anywhere may change: the hidden readings leak nowhere
        ch = gd.max_change(y, masked_forecasts(mm, xc, mask))
        # Record
        worst_leak = max(worst_leak, ch)
        # Count
        leak += int(ch <= args.tolerance)
        # Change an available neighbor of the hidden meter, b (meter 1)
        xb = x.clone()
        # New history for b
        xb[0, :, 1, :] = gd.redraw(xb[0, :, 1, :], C)
        # The hidden meter's forecast must respond: it keeps its incoming messages
        incoming += int(gd.max_change(y[0, 2], masked_forecasts(mm, xb, mask)[0, 2]) > args.tolerance)
        # Ungated masked model without the hidden meter's outgoing edges
        md = sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed + s, masked=True, maskable_meters=maskable)
        # Same weights
        md.load_state_dict(mm.state_dict())
        # Evaluation mode
        md.eval()
        # Remove c's outgoing edges only
        without_outgoing(md, 2)
        # The gated model must equal it: the hidden meter is silenced as a sender
        d = gd.max_change(y, masked_forecasts(md, x, mask))
        # Record
        worst_deleted = max(worst_deleted, d)
        # Count
        as_deleted += int(d <= args.tolerance)
        # Ungated masked model with every edge: its neighbors' forecasts must differ, so the gate did act
        mu = sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed + s, masked=True, maskable_meters=maskable)
        # Same weights
        mu.load_state_dict(mm.state_dict())
        # Evaluation mode
        mu.eval()
        # Neighbors b and d
        fired += int(gd.max_change(y[0, [1, 3]], masked_forecasts(mu, x, mask)[0, [1, 3]]) > args.tolerance)
    # Leakage, every draw
    ok &= gd.report(f"T4 no leakage: changing the hidden meter's own readings changed no forecast in {leak} of {args.draws} initializations (all required)", leak == args.draws, f"largest change {worst_leak:.2e}")
    # Incoming kept
    ok &= gd.report(f"T4 hidden receiver keeps incoming messages: changing an available neighbor moved the hidden meter's forecast in {incoming} of {args.draws} initializations (at least one required)", incoming >= 1)
    # Silenced as sender, every draw
    ok &= gd.report(f"T4 hidden meter silenced as a sender: the gated model equals the ungated model without its outgoing edges in {as_deleted} of {args.draws} initializations (all required)", as_deleted == args.draws, f"largest difference {worst_deleted:.2e}")
    # The gate acted
    ok &= gd.report(f"T4 positive control: the gated forecasts of the hidden meter's neighbors differ from the ungated model's in {fired} of {args.draws} initializations (at least one required)", fired >= 1)
    # Result
    return ok


def case_batch(args, body, base, gated, x):
    """Case T5: batch isolation with gating, on the real graph."""
    # All pass so far
    ok = True
    # Sizes
    C, G = gated.C, args.max_gap
    # Graph ids: two examples on graph 0, one on graph 1, one on the last graph
    gid = torch.tensor([0, 0, 1, len(body["graphs"]) - 1])
    # A meter with a neighbor in graph 0, and that neighbor
    j, i = int(body["graphs"][0]["edges"][0]["source"]), int(body["graphs"][0]["edges"][0]["target"])
    # Example 0: meter j unavailable; the other examples all valid
    xg = remove_start(x, 0, j, G)
    # Corrections of both models
    yg, yb = gd.correction(gated, xg, gid), gd.correction(base, xg, gid)
    # In example 0 the gate must act on j's neighbor i
    acted = gd.max_change(yg[0, i], yb[0, i])
    # In the other examples every sender is available, so the gated model equals the ungated one
    others = gd.max_change(yg[1:], yb[1:])
    # Check
    ok &= gd.report("T5 gating is per example: an unavailable sender in one example changes that example and leaves the others equal to the ungated model", acted > args.tolerance and others <= args.tolerance, f"change in the gated example {acted:.2e}, largest difference elsewhere {others:.2e}")
    # Change example 1 entirely
    x2 = xg.clone()
    # New history for every meter of example 1
    x2[1] = gd.redraw(x2[1], C)
    # Corrections
    y2 = gd.correction(gated, x2, gid)
    # Example 1 must change and the others must not
    moved, still = gd.max_change(yg[1], y2[1]), gd.max_change(yg[[0, 2, 3]], y2[[0, 2, 3]])
    # Check
    ok &= gd.report("T5 batch isolation with gating: changing one example's inputs changes that example and no other", moved > args.tolerance and still <= args.tolerance, f"perturbed change {moved:.2e}, largest change elsewhere {still:.2e}")
    # Result
    return ok


def case_degrees(args):
    """Case T6: the normalized weights and the layer's output against the hand formula, on the fixture with b silenced."""
    # The function the GCN layer calls to normalize its weights
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    # All pass so far
    ok = True
    # Fixture and a gated model, for its edge list and its layer
    fman, fgraph = fixture_graph()
    # Model
    fm = sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed, sender_gate=True, gate_max_gap=args.max_gap)
    # Evaluation mode
    fm.eval()
    # Nodes
    n = len(fman["meters"])
    # Edge list of the fixture: a - b - c - d in both directions
    ei, ew = fm.edge_set(0)
    # Senders: b (node 1) silenced, every other node may send
    a = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])
    # Gated weights
    gw = sg.gate_edge_weights(ei, ew, a[None, :] > 0.5)
    # Degrees by hand, d_i = 1 + sum of a_j over the neighbors of i:
    # a has neighbor b (silenced): 1. b has neighbors a and c: 3. c has neighbors b (silenced) and d: 2. d has neighbor c: 2. e has none: 1.
    d = torch.tensor([1.0, 3.0, 2.0, 2.0, 1.0])
    # The layer's normalized weights, self-loops added
    ni, nw = gcn_norm(ei, gw, n, add_self_loops=True)
    # Expected weight of every returned edge j -> i: 1 / d_i for a self-loop, a_j / sqrt(d_i * d_j) otherwise
    want = torch.where(ni[0] == ni[1], 1.0 / d[ni[1]], a[ni[0]] / torch.sqrt(d[ni[1]] * d[ni[0]]))
    # Largest difference
    diff = gd.max_change(nw, want)
    # Check, with the count of edges: 6 directed edges and 5 self-loops
    ok &= gd.report("T6 degrees: the weights the layer normalizes equal a_j / sqrt(d_i * d_j) with d_i = 1 + the available neighbors of i, self-loops 1 / d_i", diff <= args.tolerance and len(nw) == ei.shape[1] + n, f"{len(nw)} weights, largest difference {diff:.2e}, silenced sender's own degree {float(d[1]):.0f}")
    # Dense form of the same formula: row i, column j
    N = torch.zeros(n, n)
    # Each directed edge j -> i
    for j, i in ei.t().tolist():
        # Its normalized weight
        N[i, j] = a[j] / torch.sqrt(d[i] * d[j])
    # Self-loops
    N += torch.diag(1.0 / d)
    # The model's own layer
    conv = fm.convs[0]
    # Node states to pass through it
    torch.manual_seed(args.seed)
    # Random states
    h = torch.randn(n, fm.hidden)
    # No gradients
    with torch.no_grad():
        # The layer with the gated weights
        out = conv(h, ei, edge_weight=gw)
        # The dense calculation: the layer's linear map, the normalized sum, then its bias
        ref = N @ conv.lin(h) + conv.bias
    # Largest difference
    diff = gd.max_change(out, ref)
    # Check
    ok &= gd.report("T6 layer output: the GCN layer with gated weights equals the dense calculation of the hand formula", diff <= 10 * args.tolerance, f"largest difference {diff:.2e} (limit {10 * args.tolerance:.0e}, sums in another order)")
    # Result
    return ok


def case_boundaries(args, manifest, timeline, x):
    """Case B: every meter unavailable, a model with no edges, and stored weights that are not 1."""
    # The function the GCN layer calls to normalize its weights
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    # All pass so far
    ok = True
    # Fixture with unit weights
    fman, fgraph = fixture_graph()
    # Sizes
    M, C, F, G = len(fman["meters"]), len(fman["channels"]), sd.n_features(), args.max_gap
    # Counters over the draws
    as_empty, acted = 0, 0
    # Largest difference for the report
    worst_empty = 0.0
    # Draws
    for s in range(args.draws):
        # Gated model with nonzero test heads
        fm = gd.randomize_heads(sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed + s, sender_gate=True, gate_max_gap=G), args.seed + s)
        # Evaluation mode
        fm.eval()
        # Inputs, all valid
        torch.manual_seed(args.seed + s)
        # Draw
        x_all = gd.draw_inputs((1, args.history, M, C * F), C)
        # Make every meter unavailable
        for m in range(M):
            # Remove its first max_gap slots
            x_all = remove_start(x_all, 0, m, G)
        # The rule must see no available meter
        if bool(sg.history_available(x_all, C, G).any()):
            # Stop: the fixture is not what the case assumes
            raise SystemExit("fixture error: a meter is still available in the all-unavailable window")
        # Gated correction
        yg = gd.correction(fm, x_all, None)
        # Ungated model with the same weights and no edges at all
        fe = sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed + s)
        # Same weights
        fe.load_state_dict(fm.state_dict())
        # Evaluation mode
        fe.eval()
        # Empty edge list
        without_edges(fe)
        # The gated model must equal it
        d = gd.max_change(yg, gd.correction(fe, x_all, None))
        # Record
        worst_empty = max(worst_empty, d)
        # Count
        as_empty += int(d <= args.tolerance)
        # Ungated model with every edge: it must differ, so the gate did act
        fu = sm.make_model("gcn", fman, fgraph, edges="true", edge_weights="binary", seed=args.seed + s)
        # Same weights
        fu.load_state_dict(fm.state_dict())
        # Evaluation mode
        fu.eval()
        # Difference on the connected meters a to d
        acted += int(gd.max_change(yg[0, :4], gd.correction(fu, x_all, None)[0, :4]) > args.tolerance)
    # Every draw
    ok &= gd.report(f"B all unavailable: with every meter unavailable the gated model equals the model with no edges in {as_empty} of {args.draws} initializations (all required)", as_empty == args.draws, f"largest difference {worst_empty:.2e}")
    # The gate acted
    ok &= gd.report(f"B all unavailable, positive control: the gated result differs from the ungated model with every edge in {acted} of {args.draws} initializations (at least one required)", acted >= 1)
    # A model built with no edges, on the real graph, ungated, with nonzero test heads
    un = gd.randomize_heads(sm.make_model("gcn", manifest, edges="none", edge_weights="admittance", seed=args.seed, timeline=timeline), args.seed)
    # Evaluation mode
    un.eval()
    # The same model with gating on
    gn = sm.make_model("gcn", manifest, edges="none", edge_weights="admittance", seed=args.seed, timeline=timeline, sender_gate=True, gate_max_gap=G)
    # Same weights
    gn.load_state_dict(un.state_dict())
    # Evaluation mode
    gn.eval()
    # Graph ids over several edge sets, all empty in these models
    gid = torch.tensor([0, 0, 1, un.n_graphs - 1])
    # Inputs with meter 0 unavailable in example 0
    xn = remove_start(x, 0, 0, G)
    # Difference between gated and ungated
    d = gd.max_change(gd.correction(gn, xn, gid), gd.correction(un, xn, gid))
    # The gated no-edge model must still respond to a meter's own history, so the equality is not empty
    same, own = gd.own_history_control(gn, xn, gid, 1, 0)
    # Check
    ok &= gd.report("B no edges: a model built with no edges runs with gating on, equals the ungated one, and still responds to a meter's own history", d <= args.tolerance and same and own > args.tolerance, f"largest difference {d:.2e}, base fixed {same}, own-history change {own:.2e}")
    # Stored weights that are not 1: edges a - b, b - c, c - d with 1.0, 0.5, 0.25
    wman, wgraph = fixture_graph((1.0, 0.5, 0.25))
    # A gated model that keeps the stored weights (scaled by their maximum, which is 1)
    wm = sm.make_model("gcn", wman, wgraph, edges="true", edge_weights="admittance", seed=args.seed, sender_gate=True, gate_max_gap=G)
    # Evaluation mode
    wm.eval()
    # Its edge list and stored weights
    ei, ew = wm.edge_set(0)
    # Senders: b (node 1) silenced
    a = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])
    # Gated weights
    gw = sg.gate_edge_weights(ei, ew, a[None, :] > 0.5)
    # Stored weight per receiver i and sender j, as a table
    W = torch.zeros(M, M)
    # Fill from the edge list
    W[ei[1], ei[0]] = ew
    # The stored weights must be the three values asked for, in both directions
    stored = bool(torch.allclose(W, torch.tensor([[0, 1.0, 0, 0, 0], [1.0, 0, 0.5, 0, 0], [0, 0.5, 0, 0.25, 0], [0, 0, 0.25, 0, 0], [0, 0, 0, 0, 0]]), rtol=0, atol=1e-7))
    # Degrees by hand, d_i = 1 + sum of w_ij * a_j:
    # a: 1 + 1.0 * 0 = 1. b: 1 + 1.0 + 0.5 = 2.5. c: 1 + 0.5 * 0 + 0.25 = 1.25. d: 1 + 0.25 = 1.25. e: 1.
    d = torch.tensor([1.0, 2.5, 1.25, 1.25, 1.0])
    # The layer's normalized weights, self-loops added
    ni, nw = gcn_norm(ei, gw, M, add_self_loops=True)
    # Expected weight of every returned edge j -> i: 1 / d_i for a self-loop, w_ij * a_j / sqrt(d_i * d_j) otherwise
    want = torch.where(ni[0] == ni[1], 1.0 / d[ni[1]], W[ni[1], ni[0]] * a[ni[0]] / torch.sqrt(d[ni[1]] * d[ni[0]]))
    # Largest difference
    diff = gd.max_change(nw, want)
    # Check
    ok &= gd.report("B non-unit weights: stored weights 1.0, 0.5, 0.25 are kept for available senders and the normalized weights equal w_ij * a_j / sqrt(d_i * d_j)", stored and diff <= args.tolerance, f"stored weights as asked {stored}, largest difference {diff:.2e}")
    # Deletion equivalence with the non-unit weights, over the draws
    kept, worst_kept = 0, 0.0
    # Draws
    for s in range(args.draws):
        # Gated model with nonzero test heads
        fm = gd.randomize_heads(sm.make_model("gcn", wman, wgraph, edges="true", edge_weights="admittance", seed=args.seed + s, sender_gate=True, gate_max_gap=G), args.seed + s)
        # Evaluation mode
        fm.eval()
        # Inputs with b unavailable
        torch.manual_seed(args.seed + s)
        # Draw and remove b's first max_gap slots
        xw = remove_start(gd.draw_inputs((1, args.history, M, C * F), C), 0, 1, G)
        # Ungated model with the same weights, without b's outgoing edges
        fd = sm.make_model("gcn", wman, wgraph, edges="true", edge_weights="admittance", seed=args.seed + s)
        # Same weights
        fd.load_state_dict(fm.state_dict())
        # Evaluation mode
        fd.eval()
        # Remove b's outgoing edges only
        without_outgoing(fd, 1)
        # Difference
        dd = gd.max_change(gd.correction(fm, xw, None), gd.correction(fd, xw, None))
        # Record
        worst_kept = max(worst_kept, dd)
        # Count
        kept += int(dd <= args.tolerance)
    # Every draw
    ok &= gd.report(f"B non-unit weights, deletion equivalence: the gated model equals the ungated model without the unavailable sender's outgoing edges in {kept} of {args.draws} initializations (all required)", kept == args.draws, f"largest difference {worst_kept:.2e}")
    # Result
    return ok


def case_compatibility(args, manifest, body, timeline, base, gated, x):
    """Case C: no new parameter or buffer, the default model equals the reference commit's, old weights load, old checkpoints read as ungated."""
    # All pass so far
    ok = True
    # State of both models
    sb, sgd = base.state_dict(), gated.state_dict()
    # Same keys and shapes
    same = list(sb) == list(sgd) and all(sb[k].shape == sgd[k].shape for k in sb)
    # Check
    ok &= gd.report("C the option adds no parameter and no buffer: gated and ungated models have the same state_dict keys and shapes", same, f"{len(sb)} entries")
    # The model code from before the change
    ref, reason = load_reference(args.reference_commit)
    # Git could not provide it
    if ref is None:
        # A failure, not a skip: the default path must be shown unchanged
        ok &= gd.report(f"C reference commit {args.reference_commit}: its socal_models.py could be read from git", False, reason)
        # Result
        return ok
    # Reference model, built from a fixed random state
    torch.manual_seed(args.seed)
    # Build
    old = ref.make_model("gcn", manifest, edges="true", edge_weights="admittance", seed=args.seed, timeline=ref.load_timeline(body))
    # Current default model, built from the same random state
    torch.manual_seed(args.seed)
    # Build
    new = sm.make_model("gcn", manifest, edges="true", edge_weights="admittance", seed=args.seed, timeline=timeline)
    # States
    so, sn = old.state_dict(), new.state_dict()
    # Same keys, shapes and values
    equal = list(so) == list(sn) and all(torch.equal(so[k], sn[k]) for k in so)
    # Check
    ok &= gd.report(f"C default model: built from the same random state, the current code and commit {args.reference_commit} give identical weights and buffers", equal, f"{len(so)} entries")
    # Nonzero test heads on the reference model
    gd.randomize_heads(old, args.seed)
    # The same test heads on the current model
    gd.randomize_heads(new, args.seed)
    # Evaluation mode for the reference model
    old.eval()
    # Evaluation mode for the current model
    new.eval()
    # Graph ids over several edge sets
    gid = torch.tensor([0, 0, 1, len(body["graphs"]) - 1])
    # Full forecasts of both
    fo, fn = gd.forecasts(old, x, gid), gd.forecasts(new, x, gid)
    # Check: identical, not only close
    ok &= gd.report("C default forecasts: the current code with the option off gives forecasts identical to the reference commit's", bool(torch.equal(fo, fn)), f"largest difference {gd.max_change(fo, fn):.2e}")
    # Weights saved by the reference code load into the current code, gated or not
    loaded = True
    # Try both
    try:
        # Ungated
        new.load_state_dict(so, strict=True)
        # Gated
        gated_copy = sm.make_model("gcn", manifest, edges="true", edge_weights="admittance", seed=args.seed, timeline=timeline, sender_gate=True, gate_max_gap=args.max_gap)
        # Load
        gated_copy.load_state_dict(so, strict=True)
    # A missing or unexpected key
    except RuntimeError as e:
        # Record
        loaded = False
        # Say why
        print(f"  load_state_dict refused: {e}")
    # Check
    ok &= gd.report("C old weights: a state_dict from the reference code loads into the current code with the option off and on", loaded)
    # Checkpoint options: a checkpoint saved before the option existed has no sender_gate entry
    folder = tempfile.mkdtemp()
    # Path
    path = os.path.join(folder, "checkpoint.pt")
    # Save weights with options that lack the key
    torch.save({"model": so, "options": {}}, path)
    # An ungated run must load it
    try:
        # Load with the option off
        sm.load_checkpoint(path, new, {"sender_gate": False})
        # Loaded
        off_ok = True
    # Refused
    except ValueError:
        # Wrong
        off_ok = False
    # A gated run must be refused
    try:
        # Load with the option on
        sm.load_checkpoint(path, new, {"sender_gate": True})
        # Loaded, which is wrong
        on_refused = False
    # Refused
    except ValueError:
        # Right
        on_refused = True
    # A checkpoint saved with the option on must be refused for an ungated run
    torch.save({"model": so, "options": {"sender_gate": True}}, path)
    # Try
    try:
        # Load with the option off
        sm.load_checkpoint(path, new, {"sender_gate": False})
        # Loaded, which is wrong
        gated_refused = False
    # Refused
    except ValueError:
        # Right
        gated_refused = True
    # Check
    ok &= gd.report("C checkpoints: one without the option loads for an ungated run and is refused for a gated run; one saved gated is refused for an ungated run", off_ok and on_refused and gated_refused, f"ungated loads {off_ok}, gated refused {on_refused}, gated checkpoint refused {gated_refused}")
    # Result
    return ok


def main():
    """Run every case and exit 0 when all pass."""
    # Options
    args = parse_args()
    # Seed
    torch.manual_seed(args.seed)
    # Manifest
    manifest = sd.load_manifest(args.features_dir)
    # Graph file
    body = sm.load_graph(args.graph)
    # Timeline
    timeline = sm.load_timeline(body)
    # Sizes
    M, C, F = len(manifest["meters"]), len(manifest["channels"]), sd.n_features()
    # Say what is tested and how
    print(f"sender gating tests: history {args.history}, max_gap {args.max_gap}, tolerance {args.tolerance}, {args.draws} initializations per fixture case, reference commit {args.reference_commit}")
    # Say how the models are read
    print("note: models get nonzero test heads and all-valid synthetic inputs unless a case removes readings on purpose; the readout is the learned correction, or the forecast itself for masked models")
    # Ungated model on the real graph, with nonzero test heads
    base = gd.randomize_heads(sm.make_model("gcn", manifest, edges="true", edge_weights="admittance", seed=args.seed, timeline=timeline), args.seed)
    # Evaluation mode
    base.eval()
    # Gated model on the real graph
    gated = sm.make_model("gcn", manifest, edges="true", edge_weights="admittance", seed=args.seed, timeline=timeline, sender_gate=True, gate_max_gap=args.max_gap)
    # Same weights and edges as the ungated model
    gated.load_state_dict(base.state_dict())
    # Evaluation mode
    gated.eval()
    # Inputs for the real-graph cases: four examples, all valid
    x = gd.draw_inputs((4, args.history, M, C * F), C)
    # All pass so far
    ok = True
    # Case R
    ok &= case_rule(args, manifest)
    # Case T1
    ok &= case_all_available(args, manifest, body, timeline, base, gated, x)
    # Cases T2 and T3
    ok &= case_silenced(args)
    # Case T4
    ok &= case_masked(args)
    # Case T5
    ok &= case_batch(args, body, base, gated, x)
    # Case T6
    ok &= case_degrees(args)
    # Case B
    ok &= case_boundaries(args, manifest, timeline, x)
    # Case C
    ok &= case_compatibility(args, manifest, body, timeline, base, gated, x)
    # Verdict
    print("\nall checks passed" if ok else "\nsome checks FAILED")
    # Exit
    sys.exit(0 if ok else 1)


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
