# =============================================================================
# File:        checks/main_maskedtimeline.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Test masked forecasting on the time-indexed meter graph, and
#              the rule that decides which meters may be hidden, before any
#              masked model is trained.
#
#              Why it exists: the course staff's milestone feedback asked
#              whether the model uses the graph. Masked forecasting is the
#              direct test: a meter's own history is hidden and the model
#              must forecast it anyway. The comparison between a wired
#              model and a model with no edges only means something when
#              (1) the hidden meter's own readings cannot reach its
#              forecast by any route, (2) every model in the comparison
#              hides and scores exactly the same targets, and (3) a hidden
#              meter really has a neighbor with data. If any of the three
#              fails, a training result would look normal and still be
#              wrong. This check tests all three on small made-up cases
#              where the right answer is known in advance, and it tests
#              that nothing changed for the runs that already exist.
#
#              Hiding a meter does not make its neighbors the only source
#              of information. Its embedding and the shared weights
#              remain. That is why the no-edge model is in every case
#              below: it is the comparison, not an afterthought.
#
#              The cases:
#              R   Nothing changed for existing runs. The model code of
#                  the reference commit is loaded beside the present code.
#                  With the same weights, an ordinary model on the
#                  time-indexed graph, a sender-gated one, and a masked
#                  model on a fixed graph under the legacy policy give
#                  identical results, with identical saved keys. The
#                  window dataset hands out the same arrays by default.
#              E   Eligibility, each against an answer worked by hand:
#                  E1  every reading present: permitted meters with a
#                      neighbor are eligible, a meter that is not
#                      permitted never is
#                  E2  a meter with no neighbor in the window's edge set
#                      is left out and counted, and the same meter is
#                      eligible in a period where it has a neighbor
#                  E3  a meter whose own history is unusable, with a valid
#                      target and an available neighbor, is eligible
#                  E4  a meter whose only neighbor has no usable history
#                      is left out and counted
#                  E5  a neighbor with usable history and a missing future
#                      target still counts as available, and is itself
#                      not eligible
#                  E6  a meter with one valid magnitude target and no
#                      valid angle target is eligible, and only that one
#                      entry is scored
#                  E7  no eligible meter in a window: nothing is hidden;
#                      a whole batch of such windows takes no training
#                      step
#                  E8  the four arms (current graph, no edges, sender
#                      gating, non-unit weights) agree on every eligible
#                      meter and draw the same hidden meters
#                  E9  on random patterns over mixed periods the model's
#                      eligibility equals a separate plain calculation
#                  E10 the usual per-meter filter rebuilt from unfiltered
#                      validity equals what the dataset hands out by
#                      default, so the persistence baseline is unchanged
#                  E11 the NumPy count of eligible targets equals the
#                      count from the model path across a change of
#                      topology, and counts eligible targets inside
#                      windows the pipeline drops
#                  E12 evaluation scores exactly the eligible valid
#                      targets, counted by hand
#              L   Leakage, with eligibility held fixed. For every arm, a
#                  hidden meter's forecast does not change when its own
#                  history is replaced by zeros, by another window's
#                  history or by random values, validity flags included,
#                  and when the added features of every meter change. The
#                  scoring path is tested too. Controls: where a usable
#                  message path exists, a neighbor's history does change
#                  the forecast; in the no-edge arm no other meter's
#                  history changes it.
#              G   Sender gating with a hidden meter: it sends nothing and
#                  still receives.
#              M   Mixed periods in one batch: the batched masked forecast
#                  equals one pass per example and meter.
#              S   The residual tables scoring writes, end to end, on
#                  made-up feature files with a change of topology. The
#                  persistence residuals and flags in the masked table
#                  equal the ordinary run's entry by entry. A kept window
#                  with valid persistence and no eligible hidden meter
#                  keeps its row. The masked validity written equals
#                  eligibility worked by hand at every slot. A hidden
#                  meter with a gap in its own history is scored. An
#                  ordinary run and a legacy masked run write the same
#                  tables as the reference commit.
#              X   Refusals: the legacy policy on a time-indexed graph,
#                  the new policy without the gap length, an unknown
#                  policy, a masked forecast on several edge sets without
#                  their ids.
#              C   Checkpoints: the saved keys are the same under either
#                  policy, a checkpoint without the option loads as
#                  legacy and is refused for the new policy, and the
#                  other policy is refused by name.
#              T   The real graph file, when present: its neighbor
#                  matrices hold exactly its edges for every edge set.
#
#              The made-up network has four meters a, b, c, d. Edge set 0
#              joins a-b and b-c. Edge set 1 joins a-b only. Meter d never
#              has an edge. Period 0 uses edge set 0, period 1 edge set 1.
#
#              Nothing is trained. Needs torch and torch_geometric, and
#              git for case R.
#
# Usage (from the repo root):
#   python checks/main_maskedtimeline.py
#   python checks/main_maskedtimeline.py --graph graph_meters_periods.json --reference-commit 25a6b52
#   Options: python checks/main_maskedtimeline.py --help
#
# Exit code:   0 when every case passes and none was skipped, 1 otherwise.
#
# Functions:
#   parse_args()                     command-line options with defaults
#   Tally                            counts and prints PASS, FAIL and NOT RUN lines
#   fixture(weights)                 the made-up manifest, timeline body and fixed graph
#   build(arm, args, policy, ...)    one masked model of the made-up network
#   load_reference(commit)           the model code of the reference commit, as a module
#   blank(x, present, b, m, slots)   remove a meter's readings from some history slots
#   usable_by_hand(present, max_gap) history-only usability by a plain loop
#   eligible_by_hand(...)            eligible and left-out meters by a plain loop
#   case_reference(t, args)          R
#   case_eligibility(t, args)        E1 to E9, E12
#   case_dataset(t, args)            E10, E11
#   case_leakage(t, args)            L, G, M
#   case_score(t, args)              S
#   case_refusals(t, args)           X, C
#   case_real_graph(t, args)         T
#   main()                           run everything, print the tally, exit code
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: load the reference commit's code as a module
import importlib.util
# Standard library: the made-up manifest and splits files
import json
# Standard library: paths
import os
# Standard library: read the reference commit's code from git
import subprocess
# Standard library: exit code and import path
import sys
# Standard library: temporary folder
import tempfile
# Third party: numerics
import numpy as np
# Third party: the residual tables
import pandas as pd
# Third party: tensors
import torch

# This folder, so the accepted delivery check can be imported for its helpers
HERE = os.path.dirname(os.path.abspath(__file__))
# The repo root, one level above checks/
ROOT = os.path.dirname(HERE)
# Import from the root first
sys.path.insert(0, ROOT)
# Then from this folder
sys.path.insert(1, HERE)

# Project: models
import socal_models as sm
# Project: data side
import socal_modeldata as sd
# Project: the mask policies under test
import socal_maskpolicy as mp
# Project: the period lookup, for the separate calculation of each window's edge set
import socal_metergraph as smg
# Project: helpers of the accepted graph delivery check (test heads, all-valid inputs)
import main_graphdelivery as gd

# Meter names of the made-up network
METERS = ["a", "b", "c", "d"]
# Its channels: one voltage, one current
CHANNELS = ["L1", "S1"]
# Meters the caller permits to be hidden: d never has an edge
PERMITTED = ["a", "b", "c"]
# Neighbors by hand, per edge set: set 0 joins a-b and b-c, set 1 joins a-b only
NEIGHBORS = [{0: [1], 1: [0, 2], 2: [1], 3: []}, {0: [1], 1: [0], 2: [], 3: []}]
# The four arms of the comparison
ARMS = ("current", "none", "gate", "weights")


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Test masked forecasting on the time-indexed graph and the rule for which meters may be hidden")
    # History length of the made-up windows
    parser.add_argument("--history", type=int, default=12, help="History slots of the made-up windows")
    # Gap rule of the made-up windows
    parser.add_argument("--max-gap", type=int, default=3, help="Gap rule of the made-up windows: a run of this many absent slots makes a history unusable")
    # Model size
    parser.add_argument("--hidden", type=int, default=16, help="State size of the test models")
    # Batch size
    parser.add_argument("--batch", type=int, default=6, help="Windows per made-up batch")
    # Random patterns
    parser.add_argument("--patterns", type=int, default=200, help="E9: random windows compared with the plain calculation")
    # Tolerance
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Largest absolute change of a forecast that counts as unchanged")
    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random weights and inputs")
    # Reference commit
    parser.add_argument("--reference-commit", default="25a6b52", help="Commit whose model code the present code must reproduce for existing runs")
    # Real graph file
    parser.add_argument("--graph", default="graph_meters_periods.json", help="Time-indexed meter graph for case T; the case is counted as not run when the file is absent")
    # Done
    return parser.parse_args()


class Tally:
    """Counts cases and prints one line per case."""

    def __init__(self):
        """Start with nothing counted."""
        # Passed, failed, not run
        self.passed, self.failed, self.skipped = 0, 0, 0

    def check(self, ok, text):
        """Record one case and print it."""
        # Count a pass
        self.passed += int(bool(ok))
        # Count a failure
        self.failed += int(not bool(ok))
        # Print
        print(f"  {'PASS' if ok else 'FAIL'}  {text}", flush=True)

    def skip(self, text):
        """Record a case that could not run."""
        # Count
        self.skipped += 1
        # Print
        print(f"  NOT RUN  {text}", flush=True)


def fixture(weights=(1.0, 1.0, 1.0), boundary=10000):
    """
    The made-up network: (manifest, timeline body, fixed graph). weights
    are the stored weights of edge a-b in set 0, edge b-c in set 0 and
    edge a-b in set 1. Period 0 runs up to `boundary` (UTC seconds,
    10,000 by default) and uses edge set 0. Period 1 follows and uses
    edge set 1.
    """
    # Manifest: meter and channel order
    manifest = {"meters": list(METERS), "channels": list(CHANNELS)}
    # One node per meter
    nodes = [{"meter": m} for m in METERS]
    # Meter placement
    meters = {m: {"node": i} for i, m in enumerate(METERS)}
    # Edge set 0: a-b and b-c
    set0 = [{"source": 0, "target": 1, "admittance_abs": weights[0]}, {"source": 1, "target": 2, "admittance_abs": weights[1]}]
    # Edge set 1: a-b only
    set1 = [{"source": 0, "target": 1, "admittance_abs": weights[2]}]
    # The time-indexed file body
    body = {"nodes": nodes, "meters": meters, "graphs": [{"index": 0, "edges": set0}, {"index": 1, "edges": set1}],
            "periods": [{"index": 0, "start_utc": 0, "end_utc_exclusive": int(boundary), "graph": 0}, {"index": 1, "start_utc": int(boundary), "end_utc_exclusive": 10 ** 10, "graph": 1}]}
    # A fixed graph with edge set 0, for the legacy path
    fixed = {"nodes": nodes, "meters": meters, "edges": set0}
    # Done
    return manifest, body, fixed


def build(arm, args, policy=mp.AVAILABLE_NEIGHBOR, module=sm, n_features=None, masked=True, fixed=False, boundary=10000):
    """
    One model of the made-up network, in evaluation mode with nonzero test
    heads and a nonzero mask token. arm is one of ARMS: "current" (unit
    weights), "none" (no edges), "gate" (sender gating), "weights"
    (non-unit weights). module is the model code to build with: the
    present code, or the reference commit's. fixed builds on the fixed
    graph instead of the timeline. boundary is the time at which period
    1 starts.
    """
    # The made-up network, with non-unit weights for that arm
    manifest, body, fixed_graph = fixture((0.3, 1.0, 0.6) if arm == "weights" else (1.0, 1.0, 1.0), boundary)
    # Settings shared by every arm
    common = dict(edges="none" if arm == "none" else "true", edge_weights="admittance", hidden=args.hidden, seed=args.seed, n_features=n_features)
    # Masked settings
    if masked:
        # Permitted meters, always the same list
        common.update(masked=True, maskable_meters=list(PERMITTED))
    # The reference commit knows no mask policy
    if module is sm:
        # The policy and the gap length it needs
        common.update(mask_policy=policy)
    # Sender gating for its arm, and the gap length for gating and for the policy
    if arm == "gate" or (module is sm and policy == mp.AVAILABLE_NEIGHBOR):
        # Gap length
        common.update(gate_max_gap=args.max_gap)
    # Gating on
    if arm == "gate":
        # Flag
        common.update(sender_gate=True)
    # Fixed graph or timeline
    model = module.make_model("gcn", manifest, fixed_graph, **common) if fixed else module.make_model("gcn", manifest, None, timeline=module.load_timeline(body), **common)
    # Nonzero test heads, so every forecast depends on the states it reads
    gd.randomize_heads(model, args.seed)
    # A nonzero mask token, so a hidden meter's state differs from an absent meter's
    if masked:
        # No gradients
        with torch.no_grad():
            # Random values
            model.mask_token.copy_(torch.randn(model.mask_token.shape, generator=torch.Generator().manual_seed(args.seed + 1)) * 0.1)
    # Evaluation mode
    return model.eval()


def load_reference(commit):
    """The model code (socal_models.py) of the reference commit, loaded as a separate module beside the present code. Returns the module, or None with the reason printed when git cannot supply it."""
    # Ask git for the file as it was at that commit
    done = subprocess.run(["git", "show", f"{commit}:socal_models.py"], cwd=ROOT, capture_output=True, text=True)
    # Not available
    if done.returncode != 0:
        # Say why
        print(f"  git show {commit}:socal_models.py failed: {done.stderr.strip()[:200]}")
        # Nothing
        return None
    # A folder that lives as long as the module is used
    folder = tempfile.mkdtemp(prefix="maskref_")
    # Path of the copy
    path = os.path.join(folder, "socal_models_reference.py")
    # Write it
    with open(path, "w") as f:
        # The file's text
        f.write(done.stdout)
    # Load it under another name; its own imports resolve to the repo root
    spec = importlib.util.spec_from_file_location("socal_models_reference", path)
    # Module object
    module = importlib.util.module_from_spec(spec)
    # Run it
    spec.loader.exec_module(module)
    # Done
    return module


def blank(x, present, b, m, slots):
    """Remove meter m's readings from the given history slots of window b: every feature and flag to zero in x, and the same slots marked absent in present, the plain record the hand calculation reads."""
    # No reading and no flag in the input
    x[b, slots, m, :] = 0.0
    # The same in the plain record
    present[b, slots, m] = False


def usable_by_hand(present, max_gap):
    """History-only usability by a plain loop: present is (H,) bool for one meter in one window. Usable means no run of max_gap or more absent slots."""
    # Length of the current run of absent slots
    run = 0
    # Slot by slot
    for p in present:
        # Extend or reset the run
        run = 0 if p else run + 1
        # A run of the gap length makes the history unusable
        if run >= max_gap:
            # Unusable
            return False
    # Usable
    return True


def eligible_by_hand(present, y_valid, gids, max_gap):
    """
    Eligible meters and meters left out for want of a neighbor, by plain
    loops and the hand-written neighbor lists, with no project code.
    present (B, H, M) bool, y_valid (B, M, C, 2) bool, gids (B,) edge set
    per window. Returns two bool arrays (B, M).
    """
    # Sizes
    B, M = present.shape[0], present.shape[2]
    # Outputs
    eligible, lonely = np.zeros((B, M), dtype=bool), np.zeros((B, M), dtype=bool)
    # Each window
    for b in range(B):
        # Usable history of each meter
        usable = [usable_by_hand(present[b, :, j], max_gap) for j in range(M)]
        # Each meter
        for m in range(M):
            # Permitted, with some valid target
            wanted = METERS[m] in PERMITTED and bool(y_valid[b, m].any())
            # A neighbor in this window's edge set with usable history
            has = any(usable[j] for j in NEIGHBORS[int(gids[b])][m])
            # Eligible
            eligible[b, m] = wanted and has
            # Left out for want of a neighbor
            lonely[b, m] = wanted and not has
    # Done
    return eligible, lonely


def case_reference(t, args):
    """R: nothing changed for existing runs."""
    # Heading
    print(f"\nR  Existing runs are unchanged (reference commit {args.reference_commit})")
    # The reference commit's model code
    ref = load_reference(args.reference_commit)
    # Not available
    if ref is None:
        # Not run
        t.skip("R: the reference commit's code could not be read from git")
        # Done
        return
    # Sizes
    B, H, M, C, F = args.batch, args.history, len(METERS), len(CHANNELS), sd.n_features()
    # All-valid inputs
    x = gd.draw_inputs((B, H, M, C * F), C)
    # Edge set per example: both sets
    gid = torch.arange(B) % 2
    # The ordinary and the gated model on the time-indexed graph
    for arm, name in (("current", "an ordinary model"), ("gate", "a sender-gated model")):
        # Reference and present model, not masked
        old, new = build(arm, args, module=ref, masked=False), build(arm, args, masked=False)
        # The present model takes the reference weights; strict loading fails on any missing or extra key
        new.load_state_dict(old.state_dict(), strict=True)
        # Forecasts
        with torch.no_grad():
            # Largest difference
            d = float((old(x, graph_ids=gid) - new(x, graph_ids=gid)).abs().max())
        # Judge
        t.check(d == 0.0 and list(old.state_dict()) == list(new.state_dict()), f"R {name} on the time-indexed graph: same saved keys ({len(new.state_dict())}), forecasts identical with the reference weights (largest difference {d:.1e})")
    # The masked model on a fixed graph, legacy policy
    old, new = build("current", args, module=ref, fixed=True), build("current", args, policy=mp.LEGACY, fixed=True)
    # Same weights, strict
    new.load_state_dict(old.state_dict(), strict=True)
    # A mask that hides meter b
    mask = torch.zeros(B, M, dtype=torch.bool)
    # Hide
    mask[:, 1] = True
    # Target validity with some entries off
    yv = torch.rand(B, M, C, 2, generator=torch.Generator().manual_seed(args.seed)) > 0.3
    # Forecasts
    with torch.no_grad():
        # The masked forward
        d1 = float((old(x, mask) - new(x, mask)).abs().max())
        # The scoring path
        d2 = float((ref.masked_forecast(old, x, 2)[0] - sm.masked_forecast(new, x, 2)[0]).abs().max())
    # The same random state for both draws
    torch.manual_seed(args.seed)
    # Reference draw
    m_old = ref.draw_masks(yv, old.maskable)
    # The same random state again
    torch.manual_seed(args.seed)
    # Present draw
    m_new = sm.draw_masks(yv, new.maskable)
    # Judge
    t.check(d1 == 0.0 and d2 == 0.0 and torch.equal(m_old, m_new) and list(old.state_dict()) == list(new.state_dict()), f"R a masked model on a fixed graph, legacy policy: same saved keys, masked forward identical ({d1:.1e}), scoring path identical ({d2:.1e}), same hidden meters drawn")
    # Evaluation sums on one batch, reference against present
    y = torch.randn(B, M, C, 3, generator=torch.Generator().manual_seed(args.seed))
    # Sums of both
    s_old, s_new = ref.new_sums(), sm.new_sums()
    # No gradients
    with torch.no_grad():
        # Reference
        ref.run_batches(old, [(x, y, yv, torch.zeros(B, dtype=torch.long))], True, 0.01, s_old, mask_chunk=2)
        # Present
        sm.run_batches(new, [(x, y, yv, torch.zeros(B, dtype=torch.long))], True, 0.01, s_new, mask_chunk=2)
    # Judge the keys both have
    t.check(all(s_old[k] == s_new[k] for k in s_old) and s_new["masked_no_neighbor"] == 0, f"R masked evaluation under the legacy policy: identical error sums and counts ({int(s_new['mag_n'])} magnitude targets), nothing counted as left out")
    # The window dataset by default: a piece with gaps
    rng = np.random.default_rng(args.seed)
    # Validity with gaps
    mv = rng.random((80, M, C)) > 0.25
    # Features
    X = rng.standard_normal((80, M, C, F)).astype(np.float32)
    # Both datasets, default settings
    d_old, d_new = ref.GraphWindowDataset(X, mv, mv, H, 1, args.max_gap, 1), sm.GraphWindowDataset(X, mv, mv, H, 1, args.max_gap, 1)
    # Every window of both
    got_old, got_new = d_old.get_batch(np.arange(len(d_old))), d_new.get_batch(np.arange(len(d_new)))
    # Judge
    t.check(len(d_old) == len(d_new) and len(d_new) > 0 and all(np.array_equal(a, b) for a, b in zip(got_old, got_new)), f"R the window dataset by default hands out the same inputs, targets, validity and slots as before ({len(d_new)} windows)")


def case_eligibility(t, args):
    """E1 to E9 and E12: eligibility against answers worked by hand, and what is scored."""
    # Heading
    print("\nE  Eligibility, against answers worked by hand")
    # Sizes
    H, M, C, F = args.history, len(METERS), len(CHANNELS), sd.n_features()
    # One model per arm
    models = {arm: build(arm, args) for arm in ARMS}
    # The reference model for the hand cases
    model = models["current"]

    def scene(B, period):
        """All-valid inputs, all-present record, all-valid targets and edge set ids for B windows of one period."""
        # Inputs, record, targets, ids
        return gd.draw_inputs((B, H, M, C * F), C), np.ones((B, H, M), dtype=bool), torch.ones(B, M, C, 2, dtype=torch.bool), torch.full((B,), period, dtype=torch.long)

    def judge(label, x, y_valid, gid, want_eligible, want_lonely):
        """Compare the model's eligibility for window 0 with the answer worked by hand, as lists of meter names."""
        # From the model
        el, lo = model.mask_eligible(x, y_valid, gid)
        # As names
        got_e, got_l = [METERS[i] for i in range(M) if el[0, i]], [METERS[i] for i in range(M) if lo[0, i]]
        # Judge
        t.check(got_e == want_eligible and got_l == want_lonely, f"{label}: eligible {got_e}, left out for want of a neighbor {got_l}; by hand {want_eligible} and {want_lonely}")

    # E1: everything present, period 0
    x, present, yv, gid = scene(1, 0)
    # a, b, c have a neighbor with data; d is not permitted
    judge("E1 every reading present, edge set 0", x, yv, gid, ["a", "b", "c"], [])
    # E2: everything present, period 1, where c has no edge
    x, present, yv, gid = scene(1, 1)
    # c is left out and counted
    judge("E2 edge set 1, where c has no neighbor", x, yv, gid, ["a", "b"], ["c"])
    # E3: a's own history removed entirely, period 0
    x, present, yv, gid = scene(1, 0)
    # Remove it
    blank(x, present, 0, 0, slice(None))
    # a stays eligible through b; b stays eligible through c
    judge("E3 meter a has no usable history of its own, a valid target and neighbor b with data", x, yv, gid, ["a", "b", "c"], [])
    # E4: b's history removed, period 1, where b is a's only neighbor
    x, present, yv, gid = scene(1, 1)
    # Remove it
    blank(x, present, 0, 1, slice(None))
    # a is left out; b is eligible through a; c has no neighbor
    judge("E4 edge set 1, a's only neighbor b has no usable history", x, yv, gid, ["b"], ["a", "c"])
    # E5: b has usable history and no valid target, period 1
    x, present, yv, gid = scene(1, 1)
    # No target for b
    yv[0, 1] = False
    # a is eligible, since b can still send; b has nothing to score
    judge("E5 edge set 1, neighbor b has usable history and no valid future target", x, yv, gid, ["a"], ["c"])
    # A gap shorter than the rule does not make a history unusable; a gap of the rule's length does
    x, present, yv, gid = scene(1, 1)
    # A run one slot shorter than the gap length in b's history
    blank(x, present, 0, 1, slice(2, 2 + args.max_gap - 1))
    # a still eligible
    judge(f"E4 a run of {args.max_gap - 1} absent slots in b's history (shorter than the rule)", x, yv, gid, ["a", "b"], ["c"])
    # One more slot
    blank(x, present, 0, 1, slice(2, 2 + args.max_gap))
    # Now a is left out
    judge(f"E4 a run of {args.max_gap} absent slots in b's history (the rule's length)", x, yv, gid, ["b"], ["a", "c"])
    # E6: a has one valid magnitude target and no valid angle target, period 0
    x, present, yv, gid = scene(1, 0)
    # Only magnitude on the first channel
    yv[0, 0] = False
    # That one entry
    yv[0, 0, 0, 0] = True
    # Eligible
    judge("E6 meter a has one valid magnitude target and no valid angle target", x, yv, gid, ["a", "b", "c"], [])
    # What would be scored for a: the validity of eligible meters, entry by entry
    scored = (yv & model.mask_eligible(x, yv, gid)[0][:, :, None, None])[0, 0]
    # Judge: exactly that entry
    t.check(scored.sum().item() == 1 and bool(scored[0, 0]), f"E6 exactly that one entry of meter a is scored: {int(scored.sum())} of {scored.numel()} entries, the magnitude of the first channel")
    # E7: no valid target at all
    x, present, yv, gid = scene(args.batch, 0)
    # Nothing valid
    yv[:] = False
    # Eligibility
    el, lo = model.mask_eligible(x, yv, gid)
    # The draw
    mask = sm.draw_masks(yv, model.maskable, el)
    # Judge
    t.check(not bool(el.any()) and not bool(lo.any()) and not bool(mask.any()), "E7 windows with no valid target: no meter eligible, none counted as left out, nothing hidden")
    # A training pass over such a batch must take no step
    trainee = build("current", args)
    # Its weights before
    before = [p.clone() for p in trainee.parameters()]
    # Sums and an optimizer
    sums, optimizer = sm.new_sums(), torch.optim.Adam(trainee.parameters(), lr=0.1)
    # The pass
    sm.run_batches(trainee, [(x, torch.zeros(args.batch, M, C, 3), yv, torch.stack([torch.zeros(args.batch, dtype=torch.long), gid], dim=1))], True, 0.01, sums, optimizer)
    # Judge
    t.check(all(torch.equal(a, b) for a, b in zip(before, trainee.parameters())) and sums["skipped_windows"] == args.batch and sums["mag_n"] == 0, f"E7 a batch with no eligible window takes no training step: weights unchanged, {sums['skipped_windows']} windows counted as skipped")
    # E8 and E9: random patterns over mixed periods
    g = torch.Generator().manual_seed(args.seed)
    # Windows
    B = args.patterns
    # Inputs, record, targets
    x, present, yv, _ = scene(B, 0)
    # Random edge set per window
    gid = torch.randint(0, 2, (B,), generator=g)
    # Random target validity
    yv = torch.rand(B, M, C, 2, generator=g) > 0.4
    # Random gaps: a run of random length at a random place for some meters
    rng = np.random.default_rng(args.seed)
    # Each window and meter
    for b in range(B):
        # Each meter
        for m in range(M):
            # About half get a gap
            if rng.random() < 0.5:
                # Length from 1 slot to twice the rule's length
                n = int(rng.integers(1, 2 * args.max_gap + 1))
                # Start
                s = int(rng.integers(0, H - n + 1))
                # Remove
                blank(x, present, b, m, slice(s, s + n))
    # By hand
    want_e, want_l = eligible_by_hand(present, yv.numpy(), gid.numpy(), args.max_gap)
    # From each arm
    got = {arm: models[arm].mask_eligible(x, yv, gid) for arm in ARMS}
    # E9: the reference arm against the plain calculation
    t.check(np.array_equal(got["current"][0].numpy(), want_e) and np.array_equal(got["current"][1].numpy(), want_l), f"E9 {B} random windows over both periods: the model's eligibility equals the plain calculation ({int(want_e.sum())} eligible meter-windows, {int(want_l.sum())} left out for want of a neighbor, {int((~want_e & ~want_l).sum())} neither)")
    # E8: every arm against the reference arm
    same = all(torch.equal(got[arm][0], got["current"][0]) and torch.equal(got[arm][1], got["current"][1]) for arm in ARMS)
    # The hidden meters drawn under one random state
    draws = []
    # Each arm
    for arm in ARMS:
        # The same random state
        torch.manual_seed(args.seed)
        # The draw
        draws.append(sm.draw_masks(yv, models[arm].maskable, got[arm][0]))
    # Judge
    t.check(same and all(torch.equal(d, draws[0]) for d in draws) and int(draws[0].sum()) > 0, f"E8 the four arms ({', '.join(ARMS)}) agree on every eligible and every left-out meter-window, and draw the same {int(draws[0].sum())} hidden meters under one random state")
    # Every drawn meter is eligible, one per window at most
    t.check(bool((draws[0] & ~got["current"][0]).sum() == 0) and int(draws[0].sum(dim=1).max()) == 1, "E8 every hidden meter drawn is eligible, at most one per window")
    # E12: evaluation scores exactly the eligible valid targets
    y = torch.randn(B, M, C, 3, generator=g)
    # Batch extra: target slot and period (period equals edge set in the made-up network)
    extra = torch.stack([torch.zeros(B, dtype=torch.long), gid], dim=1)
    # By hand: valid entries of eligible meters
    want_mag, want_ang = int((yv[..., 0].numpy() & want_e[:, :, None]).sum()), int((yv[..., 1].numpy() & want_e[:, :, None]).sum())
    # Each arm
    counts = {}
    # Loop
    for arm in ARMS:
        # Sums
        sums = sm.new_sums()
        # Evaluation pass
        with torch.no_grad():
            # Run
            sm.run_batches(models[arm], [(x, y, yv, extra)], True, 0.01, sums, mask_chunk=2)
        # Keep the counts
        counts[arm] = (int(sums["mag_n"]), int(sums["ang_n"]), int(sums["masked_no_neighbor"]))
    # Judge
    t.check(all(c == (want_mag, want_ang, int(want_l.sum())) for c in counts.values()), f"E12 evaluation scores exactly the eligible valid targets in every arm: {want_mag} magnitude and {want_ang} angle targets, {int(want_l.sum())} meter-windows counted as left out; arms {counts}")


def case_dataset(t, args):
    """E10 and E11: the dataset side."""
    # Heading
    print("\nE  Eligibility on the dataset side")
    # Sizes
    H, M, C, F = args.history, len(METERS), len(CHANNELS), sd.n_features()
    # A piece of 400 slots inside period 0, with gaps
    rng = np.random.default_rng(args.seed)
    # Slots
    T = 400
    # Magnitude validity: readings missing in runs, so that some histories fail the gap rule
    mv = np.ones((T, M, C), dtype=bool)
    # Runs of missing readings per meter
    for m in range(M):
        # Several runs
        for _ in range(12):
            # Length up to twice the rule's
            n = int(rng.integers(1, 2 * args.max_gap + 1))
            # Start
            s = int(rng.integers(0, T - n))
            # Remove every channel
            mv[s:s + n, m, :] = False
    # A stretch where every meter is absent, so that some windows are dropped by the pipeline
    mv[200:200 + H + 2, :, :] = False
    # A stretch where meter a is absent for a whole history and present again right after: a valid target with no usable history of its own
    mv[300:300 + H, 0, :] = False
    # Present at the slot after it
    mv[300 + H, 0, :] = True
    # A window the pipeline drops although it holds an eligible target. History slots 100 to 100 + H - 1, target slot 100 + H.
    # Meter a: no reading in the history, a reading at the target. Meter b: readings in the history, none at the target.
    # Meters c and d: no reading at all. No meter has both a usable history and a valid target, so the window is dropped.
    # Meter a still has a valid target and a neighbor (b) with usable history.
    mv[100:100 + H + 1, :, :] = False
    # Meter a at the target slot
    mv[100 + H, 0, :] = True
    # Meter b over the history
    mv[100:100 + H, 1, :] = True
    # Angle validity: a subset of the magnitude validity
    av = mv & (rng.random((T, M, C)) > 0.2)
    # Features, with the two validity flags where the model code reads them
    X = rng.standard_normal((T, M, C, F)).astype(np.float32)
    # Magnitude flag
    X[..., 3] = mv
    # Angle flag
    X[..., 4] = av
    # Slot times: ten-second slots from time 0
    times = 10 * np.arange(T)
    # The topology changes at 2,003 seconds, between two slots, so the piece holds windows of both periods and histories that cross the change
    boundary = 2003
    # The timeline
    body = fixture(boundary=boundary)[1]
    # Validated
    tl = sm.load_timeline(body)
    # The dataset by default, and with unfiltered targets
    d_def = sm.GraphWindowDataset(X, mv, av, H, 1, args.max_gap, 1, t=times, timeline=tl)
    # Unfiltered
    d_raw = sm.GraphWindowDataset(X, mv, av, H, 1, args.max_gap, 1, t=times, timeline=tl, raw_targets=True)
    # Every window of both
    idx = np.arange(len(d_def))
    # Batches
    x_d, y_d, v_d, e_d = d_def.get_batch(idx)
    # Unfiltered
    x_r, y_r, v_r, e_r = d_raw.get_batch(idx)
    # The filter rebuilt from the unfiltered validity and the inputs
    rebuilt = mp.own_usable_valid(torch.from_numpy(x_r), torch.from_numpy(v_r), C, args.max_gap).numpy()
    # Judge: same windows, same inputs, the rebuilt filter equal to the default, and the two validities really differ
    t.check(len(d_raw) == len(d_def) and np.array_equal(x_r, x_d) and np.array_equal(e_r, e_d) and np.array_equal(rebuilt, v_d) and int((v_r & ~v_d).sum()) > 0,
            f"E10 the usual per-meter filter rebuilt from unfiltered validity equals the dataset's default on {len(d_def)} windows ({int(v_d.sum())} valid entries); the unfiltered validity holds {int((v_r & ~v_d).sum())} more entries, the targets of meters whose own history fails the gap rule")
    # E11: counts from the model path on the kept windows
    model = build("current", args, boundary=boundary)
    # Eligibility of every kept window
    el, lo = model.mask_eligible(torch.from_numpy(x_r), torch.from_numpy(v_r), model.graph_ids(torch.from_numpy(e_r[:, 1])))
    # The edge set of every window start, worked out here: the period holding the window's last history slot, then that period's edge set
    window_graph = np.array([body["periods"][int(p)]["graph"] for p in smg.period_index_at(body["periods"], times[np.arange(T - H) + H - 1])], dtype=np.int64)
    # The NumPy count for every window start, with both edge sets' neighbor matrices and the permitted meters
    per = mp.eligibility_counts_np(mv, av, np.stack([mp.meter_adjacency(g, [0, 1, 2, 3]) for g in body["graphs"]]), window_graph, np.array([m in PERMITTED for m in METERS]), H, 1, args.max_gap, d_def.starts, chunk=64)
    # Totals over all windows
    counts = mp.sum_counts(per)
    # From the model path
    got = (int(el.sum()), int((torch.from_numpy(v_r[..., 0]) & el[:, :, None]).sum()), int((torch.from_numpy(v_r[..., 1]) & el[:, :, None]).sum()), int(lo.sum()))
    # From the count
    want = (counts["kept_eligible_pairs"], counts["kept_eligible_mag"], counts["kept_eligible_ang"], counts["kept_no_neighbor_pairs"])
    # Windows of each edge set among the kept ones, to show both periods are exercised
    both = (int((window_graph[d_def.starts] == 0).sum()), int((window_graph[d_def.starts] == 1).sum()))
    # Judge
    t.check(got == want and counts["kept_windows"] == len(d_def) and min(both) > 0 and sum(counts["kept_eligible_mag_by_channel"]) == want[1], f"E11 the NumPy count on the kept windows equals the model path across a change of topology ({both[0]} windows on edge set 0, {both[1]} on edge set 1, each judged by its last history slot): {want[0]} eligible meter-windows, {want[1]} magnitude and {want[2]} angle targets, {want[3]} left out for want of a neighbor, over {counts['kept_windows']} of {counts['windows']} windows")
    # Eligible targets inside windows the pipeline drops are counted, and reported, not added
    t.check(counts["windows"] > counts["kept_windows"] and counts["dropped_eligible_pairs"] > 0, f"E11 windows the pipeline drops are counted separately: {counts['windows'] - counts['kept_windows']} dropped windows hold {counts['dropped_eligible_pairs']} eligible meter-windows ({counts['dropped_eligible_mag']} magnitude targets); they are reported and not added to training or scoring")


def case_leakage(t, args):
    """L, G and M: leakage with eligibility held fixed, sender gating with a hidden meter, mixed periods."""
    # Heading
    print("\nL  Leakage with eligibility held fixed, sender gating with a hidden meter, mixed periods")
    # Sizes
    B, H, M, C, F = args.batch, args.history, len(METERS), len(CHANNELS), sd.n_features()
    # Generator
    g = torch.Generator().manual_seed(args.seed)
    # Mixed periods
    gid = torch.arange(B) % 2
    # All-valid inputs and targets, so every permitted meter with a neighbor is eligible and stays so
    x, yv = gd.draw_inputs((B, H, M, C * F), C), torch.ones(B, M, C, 2, dtype=torch.bool)
    # Each arm
    for arm in ARMS:
        # Model
        model = build(arm, args)
        # Eligibility before
        el0 = model.mask_eligible(x, yv, gid)[0]
        # Largest change of a hidden meter's forecast over the three replacements, and whether eligibility moved
        worst, moved = 0.0, False
        # Each permitted meter
        for m in range(len(PERMITTED)):
            # Hide it in every window
            mask = torch.zeros(B, M, dtype=torch.bool)
            # Set
            mask[:, m] = True
            # Forecast before
            with torch.no_grad():
                # Hidden meter's forecast
                base = model(x, mask, graph_ids=gid)[:, m]
            # The three replacements of its own history: zeros with flags off, another window's history, random values with random flags
            for other in (torch.zeros(B, H, C * F), x[torch.roll(torch.arange(B), 1), :, m, :], torch.randn(B, H, C * F, generator=g)):
                # Replace
                x2 = x.clone()
                # Its history
                x2[:, :, m, :] = other
                # Forecast after
                with torch.no_grad():
                    # Largest change
                    worst = max(worst, float((model(x2, mask, graph_ids=gid)[:, m] - base).abs().max()))
                # Eligibility after: held fixed for the hidden meter, whose own history is not part of its own rule
                moved = moved or not torch.equal(model.mask_eligible(x2, yv, gid)[0][:, m], el0[:, m])
        # Judge
        t.check(worst <= args.tolerance and not moved, f"L arm {arm}: a hidden meter's forecast does not change when its own history is replaced by zeros, another window's history or random values, flags included (largest change {worst:.1e}); its eligibility is unchanged")
        # The scoring path: every permitted meter hidden in turn
        with torch.no_grad():
            # Before
            f0 = sm.masked_forecast(model, x, 2, gid)[0]
            # One pass per meter, for case M
            single = torch.zeros_like(f0)
            # Each permitted meter
            for m in range(len(PERMITTED)):
                # Each example alone with its own edge set
                for b in range(B):
                    # Mask for one example
                    one = torch.zeros(1, M, dtype=torch.bool)
                    # Hide
                    one[0, m] = True
                    # Forecast
                    single[b, m] = model(x[b:b + 1], one, graph_ids=gid[b:b + 1])[0, m]
        # M: batched equals one pass per example and meter
        t.check(float((f0[:, :len(PERMITTED)] - single[:, :len(PERMITTED)]).abs().max()) <= 1e-5, f"M arm {arm}: the batched masked forecast over mixed periods equals one pass per example and meter (largest difference {float((f0[:, :len(PERMITTED)] - single[:, :len(PERMITTED)]).abs().max()):.1e})")
        # Controls on meter a, whose neighbor is b in both edge sets
        mask = torch.zeros(B, M, dtype=torch.bool)
        # Hide a
        mask[:, 0] = True
        # Change b's history
        x2 = x.clone()
        # New values, flags on
        x2[:, :, 1, :] = gd.redraw(x[:, :, 1, :], C)
        # Change of a's forecast
        with torch.no_grad():
            # Largest change
            d = float((model(x2, mask, graph_ids=gid)[:, 0] - model(x, mask, graph_ids=gid)[:, 0]).abs().max())
        # No edges: no other meter may reach a
        if arm == "none":
            # Judge
            t.check(d <= args.tolerance, f"L arm none, control: with no edges, a neighbor's history does not change the hidden meter's forecast (largest change {d:.1e})")
        # Edges: the neighbor must reach a
        else:
            # Judge
            t.check(d > args.tolerance, f"L arm {arm}, control: a neighbor's history changes the hidden meter's forecast (largest change {d:.1e}), so the test above can tell a leak from a dead path")
    # Added features: a model with three more features per channel
    model = build("current", args, n_features=F + 3)
    # Inputs with the added features
    xf = torch.rand(B, H, M, C, F + 3, generator=g)
    # Validity flags on
    xf[..., 3:5] = 1.0
    # Other values in the added features of every meter
    xf2 = xf.clone()
    # Replace
    xf2[..., F:] = torch.rand(B, H, M, C, 3, generator=g)
    # Flat
    xa, xb = xf.reshape(B, H, M, -1), xf2.reshape(B, H, M, -1)
    # Hide b
    mask = torch.zeros(B, M, dtype=torch.bool)
    # Set
    mask[:, 1] = True
    # Forecasts
    with torch.no_grad():
        # Masked forward and scoring path
        d1, d2 = float((model(xa, mask, graph_ids=gid) - model(xb, mask, graph_ids=gid)).abs().max()), float((sm.masked_forecast(model, xa, 2, gid)[0] - sm.masked_forecast(model, xb, 2, gid)[0]).abs().max())
        # The unmasked forward must change, or the test could not tell
        d3 = float((model(xa, graph_ids=gid) - model(xb, graph_ids=gid)).abs().max())
    # Eligibility must not read the added features either
    same = torch.equal(model.mask_eligible(xa, yv, gid)[0], model.mask_eligible(xb, yv, gid)[0])
    # Judge
    t.check(d1 == 0.0 and d2 == 0.0 and d3 > 0.0 and same, f"L added features of every meter changed: masked forward {d1:.1e} and scoring path {d2:.1e} (both must be 0), unmasked forward {d3:.1e} (must be above 0), eligibility unchanged")
    # G: sender gating with a hidden meter. Hide b, the middle meter of edge set 0.
    gated, plain = build("gate", args), build("current", args)
    # All examples on edge set 0
    gid0 = torch.zeros(B, dtype=torch.long)
    # Hide b
    mask = torch.zeros(B, M, dtype=torch.bool)
    # Set
    mask[:, 1] = True
    # The same input with b's history removed by hand and nothing hidden
    x_absent = x.clone()
    # Remove
    x_absent[:, :, 1, :] = 0.0
    # Forecasts of a and c, the meters b would send to
    with torch.no_grad():
        # Gated: b hidden against b absent
        d_gate = float((gated(x, mask, graph_ids=gid0)[:, [0, 2]] - gated(x_absent, graph_ids=gid0)[:, [0, 2]]).abs().max())
        # Ungated: the same comparison differs, since a hidden b still sends its state with the mask token
        d_plain = float((plain(x, mask, graph_ids=gid0)[:, [0, 2]] - plain(x_absent, graph_ids=gid0)[:, [0, 2]]).abs().max())
        # Gated: b still receives; change a's history and read b's forecast
        x2 = x.clone()
        # New values for a
        x2[:, :, 0, :] = gd.redraw(x[:, :, 0, :], C)
        # Change of b's forecast
        d_recv = float((gated(x2, mask, graph_ids=gid0)[:, 1] - gated(x, mask, graph_ids=gid0)[:, 1]).abs().max())
    # Judge
    t.check(d_gate <= args.tolerance and d_plain > args.tolerance and d_recv > args.tolerance, f"G with sender gating a hidden meter sends nothing: its neighbors' forecasts equal those with the meter absent (largest difference {d_gate:.1e}; without gating {d_plain:.1e}); it still receives: a neighbor's history changes its forecast ({d_recv:.1e})")


def case_score(t, args):
    """S: the residual tables that scoring writes, end to end, on made-up feature files."""
    # Heading
    print("\nS  Residual tables written by score_split, on made-up feature files")
    # Sizes
    H, M, C = args.history, len(METERS), len(CHANNELS)
    # Slot times: ten-second slots at the start of January 2025 in the feature files' convention
    T, t0 = 240, 1735718400
    # Times
    times = t0 + 10 * np.arange(T)
    # The topology changes between slots 120 and 121
    boundary = int(times[120]) + 3
    # Every reading present to begin with
    present = np.ones((T, M, C), dtype=bool)
    # Period 0, target slot 72: meter a has no reading in its history (slots 60 to 71) and a reading at the target.
    # Its neighbor b has data, so a is eligible although its own history is unusable.
    present[60:72, 0, :] = False
    # Period 1 (a-b only), target slot 162: over the history and the target, b, c and d have no reading and a has all of them.
    # The pipeline keeps the window, since a has a usable history and a valid target, and persistence can score a.
    # Nobody is eligible: a's only neighbor has no data, and no other meter has a target.
    present[150:163, 1:, :] = False
    # Angles valid wherever a reading is
    angle_ok = present.copy()
    # Random generator
    rng = np.random.default_rng(args.seed)
    # A temporary folder for the feature files and the tables
    with tempfile.TemporaryDirectory() as tmp:
        # Feature folder
        fdir = os.path.join(tmp, "features")
        # Make it
        os.makedirs(fdir)
        # The month file, in the layout the data side reads
        np.savez_compressed(os.path.join(fdir, "2025-01.npz"), t=times, rms=(100 + rng.standard_normal((T, M, C))).astype(np.float16), ang=rng.uniform(-180, 180, (T, M, C)).astype(np.float32), freq=np.full((T, M), 60.0, np.float32), mask=present, angle_ok=angle_ok)
        # The manifest
        with open(os.path.join(fdir, "manifest.json"), "w") as f:
            # Meter and channel order
            json.dump({"version": "1", "meters": list(METERS), "channels": list(CHANNELS), "months": ["2025-01"], "arrays": {}}, f)
        # Manifest as the data side reads it
        manifest = sd.load_manifest(fdir)
        # One validation block over all the slots
        splits = {"validation": [{"start": "s", "end": "e", "start_utc": pd.Timestamp(int(times[0]), unit="s", tz="UTC").isoformat(), "end_utc_exclusive": pd.Timestamp(int(times[-1]) + 10, unit="s", tz="UTC").isoformat()}]}
        # Unit scale
        scale = np.ones((M, C), dtype=np.float32)
        # Options the scoring reads
        options = {"kind": "gcn", "history": H, "horizon": 1, "max_gap": args.max_gap, "batch_size": 32, "mask_chunk": 2}
        # Validity as the data side builds it
        month = sd.load_month(fdir, "2025-01", manifest=manifest)
        # Magnitude and angle validity
        mv, av = month["mag_valid"], month["ang_valid"]

        def table(model, name, timeline, module=sm):
            """Run score_split for one model and read the table it wrote."""
            # Output folder
            out = os.path.join(tmp, name)
            # Write the residual tables
            module.score_split(model, fdir, splits, "validation", scale, manifest, torch.device("cpu"), options, out, lambda line: None, timeline=timeline)
            # Read the one table
            return pd.read_parquet(os.path.join(out, "validation_s_e.parquet"))

        def flags(frame, name):
            """A table's validity flags of one kind as an array (T, M, C) over all slot times, False where the table has no row."""
            # All slot times as the table indexes them
            index = pd.to_datetime(times, unit="s", utc=True)
            # One column per meter and channel
            return np.stack([np.stack([frame[f"{m}|{ch}|{name}"].reindex(index).fillna(False).to_numpy(dtype=bool) for ch in CHANNELS], axis=1) for m in METERS], axis=1)

        # The timeline with that boundary, as the present code reads it
        tl = sm.load_timeline(fixture(boundary=boundary)[1])
        # The masked model under the available_neighbor policy, and an ordinary model
        masked_new, ordinary = build("current", args, boundary=boundary), build("current", args, masked=False, boundary=boundary)
        # Their tables
        tab_masked, tab_ordinary = table(masked_new, "masked_new", tl), table(ordinary, "ordinary", tl)
        # S1: the persistence side of the masked table equals the ordinary run's, entry by entry
        pers = [c for c in tab_ordinary.columns if "|pers_" in c]
        # Both tables on the union of their rows
        index = tab_ordinary.index.union(tab_masked.index)
        # Validity columns and residual columns
        valid_cols, res_cols = [c for c in pers if c.endswith("_valid")], [c for c in pers if c.endswith("_res")]
        # Validity: False where a table has no row
        same_valid = np.array_equal(tab_ordinary[valid_cols].reindex(index).fillna(False).to_numpy(dtype=bool), tab_masked[valid_cols].reindex(index).fillna(False).to_numpy(dtype=bool))
        # Residuals: equal where present, missing in both otherwise
        same_res = np.array_equal(tab_ordinary[res_cols].reindex(index).to_numpy(dtype=np.float64), tab_masked[res_cols].reindex(index).to_numpy(dtype=np.float64), equal_nan=True)
        # Judge
        t.check(same_valid and same_res and int(tab_ordinary[valid_cols].to_numpy().sum()) > 0, f"S1 the persistence residuals and validity flags in the masked table equal the ordinary run's, entry by entry ({int(tab_ordinary[valid_cols].to_numpy().sum())} valid persistence entries, {len(tab_ordinary)} ordinary rows, {len(tab_masked)} masked rows)")
        # S2: the window with valid persistence and no eligible hidden meter keeps its row
        stamp = pd.Timestamp(int(times[162]), unit="s", tz="UTC")
        # Present in the masked table
        kept = stamp in tab_masked.index
        # In that row: no masked residual is valid, and persistence is valid for meter a
        none_masked = kept and not any(bool(tab_masked.loc[stamp, f"{m}|{ch}|{k}"]) for m in METERS for ch in CHANNELS for k in ("mag_valid", "ang_valid"))
        # Persistence for a
        pers_a = kept and bool(tab_masked.loc[stamp, "a|L1|pers_mag_valid"]) and np.isfinite(tab_masked.loc[stamp, "a|L1|pers_mag_res"])
        # Judge
        t.check(kept and none_masked and pers_a, f"S2 a kept window with valid persistence and no eligible hidden meter keeps its row: row present {kept}, no masked residual valid {none_masked}, persistence valid for meter a {pers_a}")
        # S3: the masked validity written equals eligibility worked by hand, for every slot
        want_mag, want_ang = np.zeros((T, M, C), dtype=bool), np.zeros((T, M, C), dtype=bool)
        # Each window the pipeline keeps
        for s in sd.retained_starts(mv, H, 1, args.max_gap, 1)[0]:
            # Target slot, and the edge set of the period holding the last history slot
            target, g = s + H, int(times[s + H - 1] >= boundary)
            # Eligible meters by the plain calculation, from presence and unfiltered target validity
            el, _ = eligible_by_hand(mv[s:s + H].any(axis=2)[None], np.stack([mv[target], av[target]], axis=-1)[None], [g], args.max_gap)
            # The targets of eligible meters
            want_mag[target], want_ang[target] = mv[target] & el[0][:, None], av[target] & el[0][:, None]
        # From the table
        got_mag, got_ang = flags(tab_masked, "mag_valid"), flags(tab_masked, "ang_valid")
        # Judge
        t.check(np.array_equal(got_mag, want_mag) and np.array_equal(got_ang, want_ang) and int(want_mag.sum()) > 0, f"S3 the masked validity written equals eligibility worked by hand at every slot: {int(want_mag.sum())} magnitude and {int(want_ang.sum())} angle targets")
        # Residuals are finite exactly where valid
        finite = np.stack([np.stack([np.isfinite(tab_masked[f"{m}|{ch}|mag_res"].reindex(pd.to_datetime(times, unit="s", utc=True)).to_numpy(dtype=np.float64)) for ch in CHANNELS], axis=1) for m in METERS], axis=1)
        # Judge
        t.check(np.array_equal(finite, want_mag), "S3 a masked magnitude residual is written exactly where its validity flag is set, and nowhere else")
        # S4: the hidden meter with a gap in its own history is scored by the masked model and not by the ordinary one
        stamp = pd.Timestamp(int(times[72]), unit="s", tz="UTC")
        # Masked: valid. Ordinary: not valid at that slot.
        t.check(bool(got_mag[72, 0].all()) and not flags(tab_ordinary, "mag_valid")[72, 0].any() and not flags(tab_ordinary, "pers_mag_valid")[72, 0].any(), f"S4 meter a at {stamp} has no usable history of its own: the masked table scores it, the ordinary table and persistence do not")
        # S5: existing runs write the same tables as the reference commit
        ref = load_reference(args.reference_commit)
        # Not available
        if ref is None:
            # Not run
            t.skip("S5: the reference commit's code could not be read from git")
            # Done
            return
        # The ordinary model of the reference commit, with the same weights
        ordinary_ref = build("current", args, module=ref, masked=False, boundary=boundary)
        # Same weights
        ordinary_ref.load_state_dict(ordinary.state_dict(), strict=True)
        # Its table
        tab_ref = table(ordinary_ref, "ordinary_ref", ref.load_timeline(fixture(boundary=boundary)[1]), module=ref)
        # Judge
        t.check(tab_ref.equals(tab_ordinary), f"S5 an ordinary run on the time-indexed graph writes the same table as the reference commit: {len(tab_ordinary)} rows, {len(tab_ordinary.columns)} columns")
        # The masked model on a fixed graph under the legacy policy
        legacy, legacy_ref = build("current", args, policy=mp.LEGACY, fixed=True), build("current", args, module=ref, fixed=True)
        # Same weights
        legacy.load_state_dict(legacy_ref.state_dict(), strict=True)
        # Their tables, no timeline
        tab_legacy, tab_legacy_ref = table(legacy, "legacy", None), table(legacy_ref, "legacy_ref", None, module=ref)
        # Judge
        t.check(tab_legacy_ref.equals(tab_legacy), f"S5 a masked run on a fixed graph under the legacy policy writes the same table as the reference commit: {len(tab_legacy)} rows")


def case_refusals(t, args):
    """X and C: refusals and checkpoint rules."""
    # Heading
    print("\nX  Refusals and checkpoint rules")
    # The made-up network
    manifest, body, fixed = fixture()
    # Timeline
    tl = sm.load_timeline(body)

    def refuses(call):
        """Whether a call raises ValueError, the model code's refusal."""
        # Try
        try:
            # Run
            call()
        # Refused
        except ValueError:
            # Yes
            return True
        # It ran
        return False

    # Legacy policy on a time-indexed graph
    r1 = refuses(lambda: sm.make_model("gcn", manifest, None, timeline=tl, hidden=args.hidden, masked=True, maskable_meters=list(PERMITTED)))
    # New policy without the gap length
    r2 = refuses(lambda: sm.make_model("gcn", manifest, None, timeline=tl, hidden=args.hidden, masked=True, maskable_meters=list(PERMITTED), mask_policy=mp.AVAILABLE_NEIGHBOR))
    # Unknown policy
    r3 = refuses(lambda: sm.make_model("gcn", manifest, None, timeline=tl, hidden=args.hidden, masked=True, maskable_meters=list(PERMITTED), mask_policy="anything", gate_max_gap=args.max_gap))
    # Judge
    t.check(r1 and r2 and r3, f"X refused: masked mode on a time-indexed graph under the legacy policy ({r1}), the available_neighbor policy without the gap length ({r2}), an unknown policy name ({r3})")
    # A model on several edge sets
    model = build("current", args)
    # Inputs
    x = gd.draw_inputs((2, args.history, len(METERS), len(CHANNELS) * sd.n_features()), len(CHANNELS))
    # The scoring path without edge set ids; eligibility asked of a legacy model
    r4, r5 = refuses(lambda: sm.masked_forecast(model, x, 2)), refuses(lambda: build("current", args, policy=mp.LEGACY, fixed=True).mask_eligible(x, torch.ones(2, 4, 2, 2, dtype=torch.bool)))
    # Judge
    t.check(r4 and r5, f"X refused: a masked forecast on several edge sets without their ids ({r4}), eligibility asked of a legacy model ({r5})")
    # The new policy on a fixed graph is allowed
    fixed_new = build("current", args, fixed=True)
    # Its eligibility on all-valid inputs: a, b, c
    el = fixed_new.mask_eligible(x, torch.ones(2, 4, 2, 2, dtype=torch.bool))[0]
    # Judge
    t.check(el.tolist() == [[True, True, True, False]] * 2, "X the available_neighbor policy also runs on a fixed graph: a, b and c eligible, d not")
    # C: saved keys under the two policies, on the fixed graph
    legacy = build("current", args, policy=mp.LEGACY, fixed=True)
    # Same keys, and no neighbor matrix among them
    t.check(list(legacy.state_dict()) == list(fixed_new.state_dict()) and not any("mask_adjacency" in k for k in fixed_new.state_dict()), f"C the saved keys are the same under either policy ({len(legacy.state_dict())} keys); the neighbor matrices are not saved")
    # Options as a run saves them
    options = {k: None for k in sm.EXPERIMENT_KEYS}
    # The ones that matter here
    options.update({"kind": "gcn", "masked": True, "maskable_meters": list(PERMITTED), "graph_selection": "fixed", "physics": "none", "n_features": sd.n_features(), "features": list(sd.FEATURES), "angdiff": "none", "sender_gate": False, "mask_policy": mp.AVAILABLE_NEIGHBOR})
    # A temporary folder
    with tempfile.TemporaryDirectory() as tmp:
        # A checkpoint of the new policy
        new_path = os.path.join(tmp, "new.pt")
        # Save
        sm.save_checkpoint(new_path, fixed_new, torch.optim.Adam(fixed_new.parameters()), options, 1, 1.0, 0, np.random.default_rng(0))
        # The same options load
        sm.load_checkpoint(new_path, fixed_new, dict(options))
        # The other policy is refused, by name
        try:
            # Attempt
            sm.load_checkpoint(new_path, legacy, dict(options, mask_policy=mp.LEGACY))
            # Loaded, wrong
            named = False
        # Refused
        except ValueError as err:
            # The option must be named
            named = "mask_policy" in str(err)
        # A checkpoint from before the option existed: no such key
        old_options = {k: v for k, v in options.items() if k != "mask_policy"}
        # Path
        old_path = os.path.join(tmp, "old.pt")
        # Save
        sm.save_checkpoint(old_path, legacy, torch.optim.Adam(legacy.parameters()), old_options, 1, 1.0, 0, np.random.default_rng(0))
        # It loads for a legacy run, with the key given and with the key absent
        sm.load_checkpoint(old_path, legacy, dict(old_options, mask_policy=mp.LEGACY))
        # Key absent on both sides
        sm.load_checkpoint(old_path, legacy, dict(old_options))
        # It is refused for a run under the new policy
        old_refused = refuses(lambda: sm.load_checkpoint(old_path, fixed_new, dict(options)))
    # Judge
    t.check(named and old_refused, f"C a checkpoint of the new policy loads with the same options and is refused, by name, for the legacy policy ({named}); a checkpoint without the option loads as legacy and is refused for the new policy ({old_refused})")


def case_real_graph(t, args):
    """T: the real graph file."""
    # Heading
    print(f"\nT  The real graph file {args.graph}")
    # Absent
    if not os.path.exists(os.path.join(ROOT, args.graph)) and not os.path.exists(args.graph):
        # Not run
        t.skip(f"T: {args.graph} is not present")
        # Done
        return
    # Read it
    body = sm.load_graph(args.graph if os.path.exists(args.graph) else os.path.join(ROOT, args.graph))
    # Timeline
    tl = sm.load_timeline(body)
    # Meter names in node order, and each meter's node
    names = [n["meter"] for n in body["nodes"]]
    # Nodes
    nodes = [int(body["meters"][m]["node"]) for m in names]
    # Neighbor matrices
    A = mp.adjacency_stack(tl["graphs"], nodes).numpy()
    # Each edge set: the matrix holds exactly the file's edges, both directions
    exact = True
    # Loop
    for k, graph in enumerate(tl["graphs"]):
        # The file's edges as a matrix, built here
        want = np.zeros_like(A[k])
        # Each edge
        for e in graph["edges"]:
            # Both directions, by node, which is the meter's position in this file
            want[nodes.index(int(e["source"])), nodes.index(int(e["target"]))] = want[nodes.index(int(e["target"])), nodes.index(int(e["source"]))] = True
        # Compare
        exact = exact and np.array_equal(A[k], want)
    # Meters with no neighbor in any edge set
    never = [names[i] for i in range(len(names)) if not A[:, i, :].any()]
    # Judge
    t.check(exact and A.shape == (len(tl["graphs"]), len(names), len(names)), f"T the neighbor matrices hold exactly the file's edges in all {len(tl['graphs'])} edge sets ({[int(a.sum() // 2) for a in A]} pairs); meters with no neighbor in any edge set: {never}; {len(names) - len(never)} meters can be permitted")


def main():
    """Run every case, print the tally, return the exit code."""
    # Options
    args = parse_args()
    # Seed
    torch.manual_seed(args.seed)
    # Tally
    t = Tally()
    # What runs
    print(f"masked forecasting on the time-indexed graph: made-up network of {len(METERS)} meters and 2 edge sets, history {args.history}, gap rule {args.max_gap}, tolerance {args.tolerance}; torch {torch.__version__}")
    # Existing runs
    case_reference(t, args)
    # Eligibility
    case_eligibility(t, args)
    # Dataset side
    case_dataset(t, args)
    # Leakage, gating, mixed periods
    case_leakage(t, args)
    # The residual tables scoring writes
    case_score(t, args)
    # Refusals and checkpoints
    case_refusals(t, args)
    # The real file
    case_real_graph(t, args)
    # Tally
    print(f"\n{t.passed} of {t.passed + t.failed} cases PASS, {t.failed} FAIL, {t.skipped} not run")
    # Failures and cases that could not run are not a pass
    return 0 if t.failed == 0 and t.skipped == 0 else 1


# Run only when executed directly
if __name__ == "__main__":
    # Exit with the code
    sys.exit(main())
