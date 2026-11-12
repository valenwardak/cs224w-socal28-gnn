# =============================================================================
# File:        socal_sendergate.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Helper functions for sender gating, the first alternative to
#              the current message passing.
#
#              Why it exists: the course staff's milestone feedback asked
#              whether nodes without readings hurt the graph model. The
#              diagnostics showed that meters without usable history stay
#              connected and keep sending messages to their neighbors. At
#              least one such sender is present in 97 to 100 percent of the
#              retained windows of the main 2024 training block and of the
#              evaluated later blocks, and in 57.5 percent of those of the
#              January 2024 design block. Sender gating stops that: a meter
#              without usable history sends no message.
#
#              The rule, per window:
#                - A meter has usable history when its history holds no run
#                  of max_gap or more consecutive absent slots. A slot is
#                  absent when none of the meter's channels has a valid
#                  magnitude. This is the pipeline's own gap rule
#                  (socal_modeldata.window_starts) with the target condition
#                  switched off. It reads the history only. It never reads
#                  the target slot, and it never reads angle validity.
#                - The existing edge weight of a message from sender j to
#                  receiver i is multiplied by 1 when j has usable history
#                  and by 0 when it has not. A weight that is not 1 keeps
#                  its value for an available sender. On the current meter
#                  graph every stored weight is 1.0.
#                - Every node keeps its own self-loop with weight 1.
#                - Every node still receives. A meter without usable
#                  history, or a meter hidden on purpose in the masked
#                  mode, keeps the messages from its available neighbors.
#                - The weights are gated BEFORE the layer normalizes them.
#                  The GCN layer computes each node's degree from the
#                  weights it is given, so a silenced sender no longer
#                  counts in any receiver's degree:
#                      d_i = 1 + sum of w_ij * a_j over the neighbors j of i
#                      message j -> i is scaled by w_ij * a_j / sqrt(d_i * d_j)
#                  with w_ij the stored edge weight, and a_j = 1 for usable
#                  history and 0 otherwise.
#                  Multiplying already normalized messages by zero would be
#                  a different operator and is not what this module does.
#
#              In the masked mode the model zeroes a hidden meter's whole
#              history, validity flags included, before anything reads it.
#              The rule then finds no usable history, so a hidden meter is
#              silenced as a sender with no extra code.
#
#              A node without any meter (possible in the bus graph, not in
#              the meter graph) has no history to judge. It keeps sending,
#              as it does today.
#
#              The stored graph file and the edge sets are not changed by
#              sender gating. It changes only the weight a message carries
#              in a given window.
#
# Used by:     socal_models.py (GraphForecaster.forward, when the model is
#              built with sender_gate=True) and checks/main_sendergate.py.
#
# Usage:       Imported, not run. For example, inside a forward pass:
#                available = history_available(x, n_channels, max_gap)
#                senders = node_senders(available, meter_node, node_count)
#                edge_weight = gate_edge_weights(edge_index, edge_weight, senders)
#
# Functions:
#   presence_from_inputs(x, n_channels)   (B, H, M) bool: the meter has a valid magnitude on any channel in the slot
#   available_from_presence(present, max_gap)
#                                         (B, M) bool: no run of max_gap absent slots in the history
#   history_available(x, n_channels, max_gap)
#                                         the two steps above in one call, from the model input
#   node_senders(available, meter_node, node_count)
#                                         (B, n_nodes) bool: the node may send in this window
#   gate_edge_weights(edge_index, edge_weight, senders)
#                                         edge weights with zero for every message from a silenced sender
#   available_from_presence_np(present, max_gap)
#                                         the same rule in NumPy, for tests without torch
# =============================================================================

# Third party: numerics, for the NumPy copy of the rule
import numpy as np
# Third party: tensors
import torch

# Position of the magnitude validity flag among a channel's input features.
# socal_modeldata.build_features writes it there for every channel.
MAG_VALID_FEATURE = 3
# A validity flag counts as set above this value, the same cutoff the model's persistence base uses
FLAG_CUTOFF = 0.5


def presence_from_inputs(x, n_channels):
    """
    Whether each meter has a reading in each history slot, from the model
    input x of shape (B, H, M, C * F). A meter is present in a slot when
    at least one of its channels has its magnitude validity flag set.
    This is the pipeline's "any valid magnitude in the slot" rule.
    Returns a bool tensor (B, H, M).
    """
    # Sizes of the batch, the history and the meters
    B, H, M = x.shape[0], x.shape[1], x.shape[2]
    # Split the last dimension into channels and features
    xf = x.reshape(B, H, M, n_channels, x.shape[-1] // n_channels)
    # Present when any channel's magnitude flag is set
    return (xf[..., MAG_VALID_FEATURE] > FLAG_CUTOFF).any(dim=-1)


def available_from_presence(present, max_gap):
    """
    Whether each meter has usable history, from present of shape (B, H, M).
    Usable means no run of max_gap or more consecutive absent slots inside
    the H history slots. Returns a bool tensor (B, M).
    """
    # The gap length must be a positive whole number
    if int(max_gap) < 1:
        # Refuse
        raise ValueError(f"max_gap must be at least 1, got {max_gap}")
    # Sizes
    B, H, M = present.shape
    # A history shorter than the gap can never hold such a run, as in the pipeline
    if int(max_gap) > H:
        # Every meter is available
        return torch.ones(B, M, dtype=torch.bool, device=present.device)
    # Absent slots as 1.0, present slots as 0.0
    absent = (~present).to(torch.float32)
    # Every stretch of max_gap consecutive slots along the history: (B, H - max_gap + 1, M, max_gap)
    stretches = absent.unfold(1, int(max_gap), 1)
    # A stretch is a forbidden gap when all of its slots are absent
    gap = stretches.sum(dim=-1) > (int(max_gap) - 0.5)
    # Usable history: no forbidden gap anywhere in the history
    return ~gap.any(dim=1)


def history_available(x, n_channels, max_gap):
    """
    Whether each meter has usable history in each window, from the model
    input x of shape (B, H, M, C * F). Returns a bool tensor (B, M).
    """
    # Presence per slot, then the gap rule
    return available_from_presence(presence_from_inputs(x, n_channels), max_gap)


def node_senders(available, meter_node, node_count):
    """
    Whether each node may send messages in each window. available is
    (B, M) bool, meter_node (M,) gives each meter's node, node_count
    (n_nodes,) is the number of meters on each node. A node may send when
    at least one of its meters has usable history. A node without any
    meter keeps sending, since it has no history to judge.
    Returns a bool tensor (B, n_nodes).
    """
    # Sizes
    B, n_nodes = available.shape[0], node_count.shape[0]
    # Count the available meters on each node
    hits = torch.zeros(B, n_nodes, dtype=torch.float32, device=available.device).index_add(1, meter_node, available.to(torch.float32))
    # Send when a meter on the node is available, or when the node holds no meter
    return (hits > 0) | (node_count == 0)[None, :]


def gate_edge_weights(edge_index, edge_weight, senders):
    """
    Edge weights after gating. edge_index is (2, E) over the stacked batch,
    where example b's nodes sit at offset b * n_nodes, row 0 is the sender
    and row 1 the receiver. senders is (B, n_nodes) bool. A message keeps
    its weight when its sender may send and gets weight 0 when it may not.
    Self-loops are not in edge_index. The layer adds them with weight 1.
    Returns a tensor (E,).
    """
    # One flag per stacked node, in the same order as the stacked node features
    flat = senders.reshape(-1).to(edge_weight.dtype)
    # Multiply each message's weight by its sender's flag
    return edge_weight * flat[edge_index[0]]


def available_from_presence_np(present, max_gap):
    """
    The same gap rule as available_from_presence, in NumPy, so the rule can
    be tested where torch is not installed. present is a bool array
    (B, H, M). Returns a bool array (B, M).
    """
    # The gap length must be a positive whole number
    if int(max_gap) < 1:
        # Refuse
        raise ValueError(f"max_gap must be at least 1, got {max_gap}")
    # Sizes
    B, H, M = present.shape
    # A history shorter than the gap can never hold such a run
    if int(max_gap) > H:
        # Every meter is available
        return np.ones((B, M), dtype=bool)
    # Absent slots as 1, present slots as 0
    absent = (~present).astype(np.int64)
    # Running total of absent slots along the history, with a leading zero
    total = np.concatenate([np.zeros((B, 1, M), dtype=np.int64), np.cumsum(absent, axis=1)], axis=1)
    # Absent slots inside every stretch of max_gap consecutive slots
    inside = total[:, int(max_gap):, :] - total[:, :H - int(max_gap) + 1, :]
    # Usable history: no stretch is absent throughout
    return ~(inside == int(max_gap)).any(axis=1)
