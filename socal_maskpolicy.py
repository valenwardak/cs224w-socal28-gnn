# =============================================================================
# File:        socal_maskpolicy.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Helper functions that decide which meters may be hidden in
#              masked forecasting, window by window.
#
#              Why it exists: the course staff's milestone feedback asked
#              whether the model uses the graph. Masked forecasting is the
#              direct test: one meter's own history is hidden and the model
#              has to forecast that meter anyway. For the test to say
#              something about the graph, a hidden meter must have at least
#              one neighbor that actually has data in that window. If no
#              neighbor has data, the model can only fall back on what it
#              has learned about that meter in general (its embedding and
#              the shared weights). That is the same as having no graph, so
#              such an example only dilutes the comparison. The diagnostics
#              showed that neighbors without data are common, so the rule
#              matters.
#
#              Hiding a meter's history does not make its neighbors the
#              only source of information. The model still has a learned
#              profile of every meter. That is why the masked model is
#              always compared with a matched model that has no edges and
#              is scored on exactly the same hidden targets.
#
#              The two policies:
#                "legacy"              the rule of the first masked trial,
#                                      kept so that its checkpoints still
#                                      load and behave as they did: a meter
#                                      may be hidden when the caller
#                                      permits it and it has a valid
#                                      target in the window. Fixed graph
#                                      only. This module is not used.
#                "available_neighbor"  the rule the reviewers agreed on. A
#                                      target (window, meter, channel,
#                                      magnitude or angle) is eligible
#                                      when all of these hold:
#                                        1. the caller permits the meter
#                                        2. the target itself is valid for
#                                           that channel and mode
#                                        3. the meter has at least one
#                                           neighbor in the edge set in
#                                           force for the window, and at
#                                           least one such neighbor has
#                                           usable history in the window
#
#              Details of the agreed rule, each with its reason:
#                - Usable history is the pipeline's history-only gap rule:
#                  no run of max_gap or more consecutive absent slots in
#                  the history. It does not look at the neighbor's future
#                  target, since a neighbor can send a message whether or
#                  not its next reading exists. It is the same function
#                  sender gating uses (socal_sendergate.history_available),
#                  so gating and eligibility cannot disagree.
#                - The hidden meter's own history does not need to be
#                  usable. Its history is blanked out before the model sees
#                  it, so its usability makes no difference to the input.
#                - Eligibility is computed before anything is hidden, from
#                  the original inputs.
#                - Neighbors come from the graph file's own edges for the
#                  window's period, never from the edges the model passes
#                  messages over. The no-edge control therefore hides and
#                  scores exactly the same targets as the wired models.
#                - A permitted meter with a valid target and no available
#                  neighbor is not eligible. It is counted separately, so
#                  the report can say how many there were.
#
#              The window dataset normally removes a meter's target when
#              the meter's own history fails the gap rule. For this policy
#              the dataset hands out the target validity unfiltered
#              (raw_targets), and own_usable_valid() rebuilds the filtered
#              version from the inputs where it is still needed: for the
#              persistence baseline, whose residual tables must not change.
#
# Used by:     socal_models.py (GraphForecaster, run_batches, score_split,
#              make_model), main_gnn.py, main_maskpopulation.py (the report
#              of eligible targets on the real data) and
#              checks/main_maskedtimeline.py.
#
# Usage:       Imported, not run. For example, for one batch:
#                eligible, lonely = eligible_meters(x, y_valid, permitted, adjacency, graph_ids, n_channels, max_gap)
#
# Functions:
#   check_policy(policy)                    refuse a policy name that is not known
#   meter_adjacency(graph, meter_node)      (M, M) bool: which meters are neighbors under one graph's own edges
#   adjacency_stack(graphs, meter_node)     (G, M, M) bool tensor, one matrix per edge set
#   neighbor_available(x, adjacency, graph_ids, n_channels, max_gap)
#                                           (B, M) bool: the meter has a neighbor with usable history in the window
#   eligible_meters(x, y_valid, permitted, adjacency, graph_ids, n_channels, max_gap)
#                                           (B, M) bool eligible meters, and (B, M) bool meters left out for want of a neighbor
#   own_usable_valid(x, y_valid, n_channels, max_gap)
#                                           the dataset's filtered target validity, rebuilt from unfiltered validity and the inputs
#   eligibility_counts_np(mag_valid, ang_valid, adjacency, window_graph, permitted, history, horizon, max_gap, kept_starts, chunk)
#                                           counts for every window of a piece, in NumPy, under each window's own edge set
#   sum_counts(per_window, select)          totals of those counts, split by windows the pipeline keeps and drops
# =============================================================================

# Third party: numerics, for the NumPy functions
import numpy as np
# Third party: tensors
import torch

# Project: the history-only gap rule, shared with sender gating
import socal_sendergate as sg

# The policy of the first masked trial, kept for its checkpoints
LEGACY = "legacy"
# The agreed policy: a valid target and at least one neighbor with usable history
AVAILABLE_NEIGHBOR = "available_neighbor"
# Every policy name that is accepted
POLICIES = (LEGACY, AVAILABLE_NEIGHBOR)


def check_policy(policy):
    """Return the policy name, or raise ValueError when it is not one of POLICIES."""
    # Unknown name
    if policy not in POLICIES:
        # Refuse
        raise ValueError(f"mask policy must be one of {POLICIES}, got {policy!r}")
    # Known
    return policy


def meter_adjacency(graph, meter_node):
    """
    Which meters are neighbors under one graph's own edges. graph is one
    graph in graph.json layout (nodes, edges), with the file's edges, not
    the edges a control run passes messages over. meter_node gives each
    meter's node, in manifest order. Meter j is a neighbor of meter i when
    their nodes are joined by an edge, or when they sit on the same node
    (their states are averaged there, so each reaches the other). A meter
    is never its own neighbor. Returns a NumPy bool array (M, M),
    symmetric.
    """
    # Each meter's node as plain numbers
    nodes = [int(n) for n in meter_node]
    # Number of meters
    M = len(nodes)
    # Nodes joined to each node
    joined = {}
    # Each edge of the file
    for e in graph["edges"]:
        # One direction
        joined.setdefault(int(e["source"]), set()).add(int(e["target"]))
        # The other direction
        joined.setdefault(int(e["target"]), set()).add(int(e["source"]))
    # Empty matrix
    A = np.zeros((M, M), dtype=bool)
    # Each meter
    for i in range(M):
        # Nodes that reach meter i: its own node and the nodes joined to it
        reach = joined.get(nodes[i], set()) | {nodes[i]}
        # Each other meter
        for j in range(M):
            # A neighbor when it sits on one of those nodes
            A[i, j] = (j != i) and (nodes[j] in reach)
    # Done
    return A


def adjacency_stack(graphs, meter_node):
    """The neighbor matrix of every edge set, stacked: a bool tensor (G, M, M). graphs is the list of graphs in graph.json layout, one per edge set."""
    # One matrix per edge set
    return torch.as_tensor(np.stack([meter_adjacency(g, meter_node) for g in graphs]), dtype=torch.bool)


def neighbor_available(x, adjacency, graph_ids, n_channels, max_gap):
    """
    Whether each meter has at least one neighbor with usable history in
    each window. x is the model input (B, H, M, C * F) before anything is
    hidden. adjacency is (G, M, M) bool. graph_ids (B,) names each
    window's edge set, or is None for a fixed graph (edge set 0). Returns
    (has (B, M) bool, available (B, M) bool) with available the
    history-only usability of every meter.
    """
    # History-only usability of every meter in every window
    available = sg.history_available(x, n_channels, max_gap)
    # The neighbor matrix of each window's edge set: (B, M, M)
    A = adjacency[graph_ids] if graph_ids is not None else adjacency[0].unsqueeze(0).expand(x.shape[0], -1, -1)
    # A neighbor counts when it is adjacent and available
    has = (A.to(available.device) & available[:, None, :]).any(dim=-1)
    # Done
    return has, available


def eligible_meters(x, y_valid, permitted, adjacency, graph_ids, n_channels, max_gap):
    """
    The meters that may be hidden in each window under the
    available_neighbor policy. x (B, H, M, C * F) is the input before
    anything is hidden. y_valid (B, M, C, 2) is the unfiltered target
    validity. permitted (M,) bool is the caller's list. Returns
      eligible (B, M) bool  permitted, some valid target, and a neighbor
                            with usable history
      lonely   (B, M) bool  permitted with some valid target and no such
                            neighbor: left out, and counted by the caller
    Which channels and modes of an eligible meter are scored is decided by
    y_valid itself, entry by entry.
    """
    # A neighbor with usable history, per window and meter
    has, _ = neighbor_available(x, adjacency, graph_ids, n_channels, max_gap)
    # Permitted meters with at least one valid target
    wanted = y_valid.any(dim=-1).any(dim=-1) & permitted[None, :]
    # Eligible, and left out for want of a neighbor
    return wanted & has, wanted & ~has


def own_usable_valid(x, y_valid, n_channels, max_gap):
    """
    The target validity the window dataset hands out by default, rebuilt
    from the unfiltered validity y_valid (B, M, C, 2) and the inputs x: a
    meter's targets count only when its own history passes the gap rule
    and it has a valid magnitude on some channel at the target slot. This
    is the per-meter rule of socal_modeldata.retained_starts. It keeps the
    persistence baseline's residuals the same as in an ordinary run.
    Returns a bool tensor (B, M, C, 2).
    """
    # Own history passes the gap rule
    history_ok = sg.history_available(x, n_channels, max_gap)
    # A valid magnitude on some channel at the target slot
    target_ok = y_valid[..., 0].any(dim=-1)
    # Both, applied to every channel and mode of the meter
    return y_valid & (history_ok & target_ok)[:, :, None, None]


def eligibility_counts_np(mag_valid, ang_valid, adjacency, window_graph, permitted, history, horizon, max_gap, kept_starts, chunk=4096):
    """
    Count eligible masked targets for every window of one piece, without
    torch, so the population can be reported before any training. A piece
    is one block-month of feature data. mag_valid and ang_valid are
    (T, M, C) bool. adjacency is (G, M, M) bool, one neighbor matrix per
    edge set. window_graph (W,) gives the edge set of each window start,
    chosen by the window's last history slot, as the model chooses it. A
    window whose history crosses a change of topology is therefore kept
    whole and judged under the edge set of its last history slot.
    permitted is (M,) bool. kept_starts are the window starts the
    pipeline keeps (socal_modeldata.retained_starts).

    The pipeline drops a window when no meter (fewer than min_meters) has
    both a usable history and a valid target. A hidden meter does not
    need a usable history of its own, so a dropped window can still hold
    an eligible target. Those are counted here and reported. They are not
    added to training or scoring, so the evaluated windows stay the
    pipeline's.

    Returns a dictionary of arrays with one entry per window start:
      kept               (W,) bool   the pipeline keeps the window
      eligible_pairs     (W,) int    eligible meters in the window
      eligible_mag       (W, C) int  eligible magnitude targets per channel
      eligible_ang       (W, C) int  eligible angle targets per channel
      no_neighbor_pairs  (W,) int    permitted meters with a valid target
                                     left out for want of an available neighbor
    Use sum_counts() to total them over all or some of the windows.
    """
    # Sizes
    T, M, C = mag_valid.shape
    # Number of window starts
    n = max(T - (history + horizon) + 1, 0)
    # The edge set of every window start must be given
    if len(window_graph) != n:
        # Refuse
        raise ValueError(f"window_graph has {len(window_graph)} entries for {n} window starts")
    # Which starts the pipeline keeps
    kept = np.zeros(n, dtype=bool)
    # Mark them
    kept[np.asarray(kept_starts, dtype=np.int64)] = True
    # A meter is present in a slot when any channel has a valid magnitude
    present = mag_valid.any(axis=2)
    # Outputs, one entry per window start
    out = {"kept": kept, "eligible_pairs": np.zeros(n, dtype=np.int64), "eligible_mag": np.zeros((n, C), dtype=np.int64), "eligible_ang": np.zeros((n, C), dtype=np.int64), "no_neighbor_pairs": np.zeros(n, dtype=np.int64)}
    # Edge set of every window as an array
    window_graph = np.asarray(window_graph, dtype=np.int64)
    # Windows in chunks, to bound memory
    for a in range(0, n, chunk):
        # Starts of this chunk
        s = np.arange(a, min(a + chunk, n))
        # History slots of each window: (W, history)
        rows = s[:, None] + np.arange(history)[None, :]
        # History-only usability: (W, M)
        available = sg.available_from_presence_np(present[rows], max_gap)
        # A neighbor that is adjacent under the window's own edge set, and available: (W, M)
        has = (adjacency[window_graph[s]] & available[:, None, :]).any(axis=-1)
        # Target slots
        target = s + history + horizon - 1
        # Valid targets of permitted meters
        mv, av = mag_valid[target] & permitted[None, :, None], ang_valid[target] & permitted[None, :, None]
        # Permitted meters with some valid target
        wanted = (mv | av).any(axis=2)
        # Eligible meters
        el = wanted & has
        # Eligible meters per window
        out["eligible_pairs"][s] = el.sum(axis=1)
        # Eligible magnitude targets per window and channel
        out["eligible_mag"][s] = (mv & el[:, :, None]).sum(axis=1)
        # Eligible angle targets per window and channel
        out["eligible_ang"][s] = (av & el[:, :, None]).sum(axis=1)
        # Meters left out for want of an available neighbor
        out["no_neighbor_pairs"][s] = (wanted & ~has).sum(axis=1)
    # Done
    return out


def sum_counts(per_window, select=None):
    """
    Totals of eligibility_counts_np over all windows, or over the windows
    flagged in select (W,) bool, split by kept and dropped windows.
    Returns a dictionary: windows, kept_windows, and for "kept" and
    "dropped" the eligible_pairs, eligible_mag, eligible_ang (all
    channels together), no_neighbor_pairs, and the per-channel lists
    eligible_mag_by_channel and eligible_ang_by_channel.
    """
    # Windows to total: all of them unless a selection is given
    select = np.ones(len(per_window["kept"]), dtype=bool) if select is None else np.asarray(select, dtype=bool)
    # Window counts
    out = {"windows": int(select.sum()), "kept_windows": int((per_window["kept"] & select).sum())}
    # Kept and dropped windows
    for name, rows in (("kept", per_window["kept"] & select), ("dropped", ~per_window["kept"] & select)):
        # Eligible meter-windows
        out[f"{name}_eligible_pairs"] = int(per_window["eligible_pairs"][rows].sum())
        # Eligible magnitude targets
        out[f"{name}_eligible_mag"] = int(per_window["eligible_mag"][rows].sum())
        # Eligible angle targets
        out[f"{name}_eligible_ang"] = int(per_window["eligible_ang"][rows].sum())
        # Left out for want of a neighbor
        out[f"{name}_no_neighbor_pairs"] = int(per_window["no_neighbor_pairs"][rows].sum())
        # Magnitude targets per channel
        out[f"{name}_eligible_mag_by_channel"] = per_window["eligible_mag"][rows].sum(axis=0).tolist()
        # Angle targets per channel
        out[f"{name}_eligible_ang_by_channel"] = per_window["eligible_ang"][rows].sum(axis=0).tolist()
    # Done
    return out
