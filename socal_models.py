# =============================================================================
# File:        socal_models.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-22
#
# Purpose:     The learned models and everything around them: the shared
#              meter-conditioned GRU, the three GNNs built on top of it
#              (GCN, GAT, GraphSAGE), the persistence baseline, the loss,
#              the training loop with checkpoints, and the scoring that
#              writes a residual per slot, meter, and channel. The data
#              side lives in socal_modeldata.py. Imported by the thin main
#              scripts main_gru.py, main_gcn.py, main_gat.py, main_sage.py.
#
# How the models relate:
#              GRU baseline   each meter's own history -> GRU -> forecast.
#                             One GRU shared by all meters, with a learned
#                             meter embedding, so it is a shared,
#                             meter-conditioned GRU rather than 22 separate
#                             networks. No graph.
#              GNNs           the same GRU encodes every meter's history,
#                             the states are placed on the grid graph, one
#                             message-passing layer mixes each node with its
#                             wired neighbors, and the forecast head reads
#                             the meter's own state and its node's mixed
#                             state. Same inputs, loss, and training loop as
#                             the GRU, so the gap between them is the graph.
#              Persistence    forecast = last valid reading, magnitude and
#                             angle each from the last slot where that part
#                             was valid. No parameters. The GRU has to beat it.
#
# Predict the change (--predict-delta, default on):
#              with it, every learned model outputs a correction added to the
#              persistence forecast: forecast = last valid reading + head(state).
#              "No change" then costs nothing to learn, and the model only has
#              to capture what persistence misses. Where the history holds no
#              valid value to start from, the head output stands alone. The
#              loss, residuals, and scoring are unchanged.
#
# Batches:     assembled in bulk with array indexing (get_batch on the
#              datasets, iter_batches below), not one example at a time,
#              which was the cost in the first smoke test.
#
# Graph interface (the GNNs):
#              graph.json has 43 nodes and 22 meters. Each meter's GRU state
#              goes to its node; a node with two meters (19 and 21 share the
#              A-side main bus) takes their mean; a node with no meter takes
#              one learned "unmetered" vector. Message passing runs over all
#              43 nodes. Only metered nodes produce forecasts. Edges are
#              undirected, stored in both directions. GCN uses the scalar
#              admittance_abs as the edge weight, GAT takes it as a one-
#              number edge feature, GraphSAGE ignores weights (mean of
#              neighbors). --edges none|random|true and --edge-weights
#              binary|admittance give the topology ablation. Two limits of
#              that ablation, stated so nobody over-reads it: with "none",
#              meters that share a node (19 and 21) are still averaged
#              before message passing, so it is a no-inter-node-edges
#              control, not a fully independent-meter one. And with one
#              message-passing layer, information crosses at most one edge,
#              so a meter two hops away through an unmetered node is not
#              reached; --gnn-layers adds hops.
#
# Loss:        Squared error on the scaled magnitude where the target
#              magnitude is valid, plus squared error on cos and sin where
#              the target angle is valid. Each term is averaged over its own
#              valid count, then combined with a fixed angle weight. No NaN
#              ever enters the arithmetic: inputs are zero where invalid.
#
# Residuals written by the scoring, per slot, meter, channel:
#              mag_res   |forecast - actual| in scaled units (divide by scale.json
#                        to go back to volts or amps is the reverse: multiply)
#              ang_res   angle between the forecast (cos, sin) vector and the
#                        actual one, in degrees, 0 to 180, computed from the
#                        directions of both vectors; a forecast vector of
#                        exactly zero length has no direction and scores 90
#              mag_valid, ang_valid   whether each residual exists
#              The same four for the persistence baseline, prefixed "pers_",
#              where pers_mag_valid and pers_ang_valid also require that a
#              valid magnitude or angle existed in the history to copy.
#              No combined score here: that is defined once in the scoring
#              script so every model is scored the same way.
#
# Validation:  block 0 of "validation" (January 2024) is for early stopping,
#              block 1 (February 2025) is reserved for conformal calibration
#              and is never used to choose a model.
#
# Checkpoints: model weights, optimizer state, the options used, the epoch,
#              the best validation loss, the early-stopping counter, the
#              NumPy shuffling state, and torch's CPU and CUDA random states,
#              written every epoch, so a Colab disconnect loses at most one
#              epoch and a resumed run continues the same sequence. On
#              resume the saved options are compared with the current ones
#              and a mismatch raises.
#
# Losses reported: pooled over valid readings, not averaged over batches:
#              sum of squared errors divided by the number of valid targets,
#              for magnitude and angle separately, then combined with the
#              angle weight. An empty validation set raises.
#
# Functions and classes:
#   Datasets
#     GraphWindowDataset              synchronized windows over all meters, for the GNNs
#     iter_batches(dataset, batch_size, shuffle, rng, device)
#                                     yield tensor batches from either dataset kind
#   Graph
#     load_graph(path)                graph.json
#     meter_node_index(graph, manifest)
#                                     (M,) node index per meter
#     build_edges(graph, mode, weights, seed)
#                                     edge_index (2, E) and edge_weight (E,) tensors
#   Models
#     MeterGRU                        shared meter-conditioned GRU with a forecast head
#     GraphForecaster                 MeterGRU encoder + one GNN layer + head
#     persistence_forecast(x, C)      last valid reading as the forecast, with validity
#   Loss and metrics
#     masked_loss(y_hat, y, y_valid, angle_weight)
#                                     the training loss
#     residuals(y_hat, y, y_valid)    mag_res and ang_res arrays
#   Training
#     make_model(kind, ...)           build a model by name
#     run_batches(...)                one pass over a batch iterator, accumulates error sums
#     pooled(sums)                    pooled losses from the sums
#     make_dataset(...)               the right dataset kind for one piece
#     evaluate(...)                   pooled loss on a split, plus per meter
#     train(...)                      epochs over the train split, early stopping, checkpoints
#     save_checkpoint / load_checkpoint / save_atomic
#   Scoring
#     score_split(...)                residual tables per block, written to disk
# =============================================================================

# Standard library: JSON for graph and summaries
import json
# Standard library: paths
import os
# Standard library: timing
import time
# Third party: numerics
import numpy as np
# Third party: tables for the residual output
import pandas as pd
# Third party: deep learning
import torch
# Third party: layers
import torch.nn as nn
# Project: data side
import socal_modeldata as sd

# Feature count per channel, from the data module
F = sd.F


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

class GraphWindowDataset:
    """
    Synchronized windows over all meters, for the GNNs. One example is
    one window start, with every meter's history at once:
      x        (history, M, C * F)   every meter's history
      y        (M, C, 3)             every meter's target
      y_valid  (M, C, 2)             target validity, zeroed for meters whose
                                     history fails the gap rule in this window
    A window is kept when at least min_meters meters pass the gap rule and
    have a valid target. The per-meter rule is the same one WindowDataset
    uses, so the GRU and the GNNs train on the same target slots.
    """

    def __init__(self, X, mag_valid, ang_valid, history, horizon, max_gap, min_meters=1):
        # Keep the arrays
        self.X, self.history, self.horizon = X, history, horizon
        # Sizes
        T, M, C, _ = X.shape
        # Targets: scaled magnitude, cos, sin
        self.y = X[..., 0:3]
        # Target validity
        self.y_valid = np.stack([mag_valid, ang_valid], axis=-1)
        # Candidate starts
        span = history + horizon
        # Number of starts
        n_starts = max(T - span + 1, 0)
        # Usable flag per start and meter
        usable = np.zeros((n_starts, M), dtype=bool)
        # Loop over meters
        for mi in range(M):
            # Any reading in the slot
            present_any = mag_valid[:, mi, :].any(axis=1)
            # Usable starts for this meter
            starts = sd.window_starts(present_any, history, horizon, max_gap, present_any)
            # Mark
            usable[starts, mi] = True
        # Keep windows with enough meters
        self.starts = np.nonzero(usable.sum(axis=1) >= min_meters)[0]
        # Usable flags for the kept windows
        self.usable = usable[self.starts]

    def __len__(self):
        """Number of windows."""
        # Count
        return len(self.starts)

    def __getitem__(self, i):
        """One window as numpy arrays."""
        # Start
        s = self.starts[i]
        # Target slot
        target = s + self.history + self.horizon - 1
        # History, channels and features flattened per meter
        x = self.X[s:s + self.history].reshape(self.history, self.X.shape[1], -1)
        # Validity, zeroed for meters that fail the gap rule in this window
        y_valid = self.y_valid[target] & self.usable[i][:, None, None]
        # Return
        return x, self.y[target], y_valid, int(target)

    def get_batch(self, idx):
        """
        A whole batch at once with array indexing:
          x        (B, history, M, C * F)
          y        (B, M, C, 3)
          y_valid  (B, M, C, 2)
          target   (B,) int64 target slot index
        """
        # Starts
        s = self.starts[idx]
        # Slot index of every history step: (B, history)
        rows = s[:, None] + np.arange(self.history)[None, :]
        # Gather: (B, history, M, C, F) then flatten channels and features
        x = self.X[rows].reshape(len(idx), self.history, self.X.shape[1], -1)
        # Target slots
        target = s + self.history + self.horizon - 1
        # Validity with the per-meter gap rule applied
        y_valid = self.y_valid[target] & self.usable[idx][:, :, None, None]
        # Return
        return x, self.y[target], y_valid, target.astype(np.int64)


def iter_batches(dataset, batch_size, shuffle, rng, device):
    """
    Yield tensor batches (x, y, y_valid, extra) from either dataset kind,
    assembled in bulk with get_batch. extra is the meter index for the
    per-meter dataset and the target slot for the synchronized one.
    """
    # Example order
    order = np.arange(len(dataset))
    # Shuffle if asked
    if shuffle:
        # In place
        rng.shuffle(order)
    # Loop over batches
    for start in range(0, len(order), batch_size):
        # Indices
        idx = order[start:start + batch_size]
        # Arrays
        x, y, y_valid, extra = dataset.get_batch(idx)
        # Tensors on the device
        yield (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), torch.from_numpy(y_valid).to(device), torch.from_numpy(extra).to(device))


# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------

def load_graph(path):
    """Read graph.json."""
    # Fail with the path tried
    if not os.path.exists(path):
        # Raise
        raise FileNotFoundError(f"graph file not found: {path}")
    # Read
    with open(path) as f:
        # Parse
        return json.load(f)


def meter_node_index(graph, manifest):
    """Node index per meter in manifest order, shape (M,). Raises if a meter has no node."""
    # Lookup
    out = []
    # Loop over meters in order
    for meter in manifest["meters"]:
        # Node
        node = graph["meters"].get(meter, {}).get("node")
        # Must exist
        if node is None:
            # Raise
            raise ValueError(f"meter {meter} has no node in graph.json")
        # Keep
        out.append(int(node))
    # Tensor
    return torch.tensor(out, dtype=torch.long)


def build_edges(graph, mode="true", weights="admittance", seed=0):
    """
    Edge tensors for the GNN layers, both directions of every edge.
      mode    "true"   the wiring from graph.json
              "random" the same number of edges between random node pairs
              "none"   no edges, message passing sees only self-loops
      weights "admittance" admittance_abs scaled to a 0 to 1 range by its
                           maximum, "binary" all ones
    Returns (edge_index LongTensor (2, E), edge_weight FloatTensor (E,)).
    """
    # Node count
    n = len(graph["nodes"])
    # Source, target, weight lists
    src, dst, w = [], [], []
    # Loop over edges
    for e in graph["edges"]:
        # Ends
        src.append(e["source"]); dst.append(e["target"]); w.append(float(e["admittance_abs"]))
    # Random rewiring keeps the edge count
    if mode == "random":
        # Generator
        rng = np.random.default_rng(seed)
        # New ends, no self-loops
        pairs = set()
        # Draw until enough distinct pairs
        while len(pairs) < len(src):
            # Pair
            a, b = int(rng.integers(n)), int(rng.integers(n))
            # Keep distinct ends, unordered
            if a != b:
                # Add
                pairs.add((min(a, b), max(a, b)))
        # Replace
        src, dst = [p[0] for p in pairs], [p[1] for p in pairs]
    # No edges at all
    elif mode == "none":
        # Empty
        src, dst, w = [], [], []
    # Weights
    if weights == "binary" or not w:
        # All ones
        w = [1.0] * len(src)
    # Admittance scaled by its maximum
    else:
        # Maximum
        top = max(w)
        # Scale
        w = [v / top for v in w]
    # Both directions
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long).reshape(2, -1)
    # Weights for both directions
    edge_weight = torch.tensor(w + w, dtype=torch.float32)
    # Return
    return edge_index, edge_weight


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class MeterGRU(nn.Module):
    """
    Shared, meter-conditioned GRU forecaster.

    Input per step: the meter's C * F features plus a learned embedding of
    the meter's identity. A GRU reads the history, and a head turns the
    final state into the forecast: C magnitudes and C (cos, sin) pairs.
    encode() returns the final state for the GNNs to build on.
    """

    def __init__(self, n_meters, n_channels, hidden=64, embed=8, layers=1, dropout=0.0, predict_delta=True):
        # Base class
        super().__init__()
        # Sizes
        self.C, self.hidden = n_channels, hidden
        # Whether the head outputs a correction to the persistence forecast
        self.predict_delta = predict_delta
        # Meter identity vectors
        self.meter_embed = nn.Embedding(n_meters, embed)
        # Input projection
        self.input = nn.Linear(n_channels * F + embed, hidden)
        # Recurrent core, batch first
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        # Magnitude head
        self.head_mag = nn.Linear(hidden, n_channels)
        # Angle head, cos and sin per channel
        self.head_ang = nn.Linear(hidden, 2 * n_channels)

    def encode(self, x, meter):
        """
        x (B, H, C * F), meter (B,) -> final state (B, hidden).
        """
        # Embedding repeated along the history
        emb = self.meter_embed(meter)[:, None, :].expand(-1, x.shape[1], -1)
        # Project the concatenation
        h = torch.relu(self.input(torch.cat([x, emb], dim=-1)))
        # Run the GRU
        _, state = self.gru(h)
        # Last layer's state
        return state[-1]

    def head(self, state):
        """state (B, hidden) -> forecast (B, C, 3): magnitude, cos, sin."""
        # Magnitude
        mag = self.head_mag(state)
        # Angle pair
        ang = self.head_ang(state).reshape(-1, self.C, 2)
        # Stack
        return torch.cat([mag[..., None], ang], dim=-1)

    def forward(self, x, meter):
        """Forecast from one meter's history: persistence plus the head's correction, or the head alone."""
        # Head output
        out = self.head(self.encode(x, meter))
        # Plain forecast
        if not self.predict_delta:
            # Done
            return out
        # Persistence base and where it exists
        base, ok = persistence_forecast(x, self.C)
        # Zero base where nothing valid was in the history: magnitude from ok[..., 0], angle pair from ok[..., 1]
        keep = torch.cat([ok[..., 0:1], ok[..., 1:2], ok[..., 1:2]], dim=-1)
        # Add
        return out + torch.where(keep, base, torch.zeros_like(base))


class GraphForecaster(nn.Module):
    """
    MeterGRU encoder for every meter, one message-passing layer over the
    grid graph, and a head that reads each meter's own state together with
    its node's mixed state.

    kind: "gcn", "gat", or "sage". Needs torch_geometric. gnn_layers is
    the number of message-passing hops; one hop does not reach a meter
    that sits two edges away behind an unmetered node.
    """

    def __init__(self, kind, n_meters, n_channels, n_nodes, meter_node, edge_index, edge_weight, hidden=64, embed=8, layers=1, dropout=0.0, heads=4, gnn_layers=1, predict_delta=True):
        # Base class
        super().__init__()
        # Whether the head outputs a correction to the persistence forecast
        self.predict_delta = predict_delta
        # Layers from PyTorch Geometric, imported here so the GRU runs without it
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv
        # Sizes
        self.C, self.hidden, self.n_nodes, self.kind = n_channels, hidden, n_nodes, kind
        # Shared encoder
        self.encoder = MeterGRU(n_meters, n_channels, hidden, embed, layers, dropout)
        # Meter to node, fixed
        self.register_buffer("meter_node", meter_node)
        # Edges, fixed
        self.register_buffer("edge_index", edge_index)
        # Weights, fixed
        self.register_buffer("edge_weight", edge_weight)
        # Learned vector for nodes without a meter
        self.unmetered = nn.Parameter(torch.zeros(hidden))
        # Message passing layers, gnn_layers of them, each one hop
        convs = []
        # Build
        for _ in range(gnn_layers):
            # By kind
            if kind == "gcn":
                # Scalar edge weights, self-loops added by the layer
                convs.append(GCNConv(hidden, hidden, add_self_loops=True))
            elif kind == "gat":
                # Attention with the weight as a one-number edge feature
                convs.append(GATConv(hidden, hidden, heads=heads, concat=False, edge_dim=1, add_self_loops=True))
            elif kind == "sage":
                # Mean of neighbors, no weights
                convs.append(SAGEConv(hidden, hidden))
            else:
                # Unknown
                raise ValueError(f"unknown GNN kind: {kind}")
        # Register
        self.convs = nn.ModuleList(convs)
        # Head reads own state and node state
        self.head_mag = nn.Linear(2 * hidden, n_channels)
        # Angle head
        self.head_ang = nn.Linear(2 * hidden, 2 * n_channels)
        # Count of meters per node, for averaging shared nodes
        counts = torch.zeros(n_nodes).index_add_(0, meter_node, torch.ones(len(meter_node)))
        # Keep
        self.register_buffer("node_count", counts)

    def forward(self, x):
        """
        x (B, H, M, C * F) -> forecast (B, M, C, 3).
        """
        # Sizes
        B, H, M, _ = x.shape
        # Every meter through the shared encoder: fold meters into the batch
        flat = x.permute(0, 2, 1, 3).reshape(B * M, H, -1)
        # Meter ids matching the fold
        meter = torch.arange(M, device=x.device).repeat(B)
        # States (B * M, hidden) -> (B, M, hidden)
        own = self.encoder.encode(flat, meter).reshape(B, M, self.hidden)
        # Node features start as the unmetered vector everywhere
        nodes = self.unmetered.expand(B, self.n_nodes, self.hidden).clone()
        # Sum meter states into their nodes
        nodes = nodes.index_add(1, self.meter_node, own) - self.unmetered * (self.node_count > 0).float()[None, :, None]
        # Average nodes with more than one meter
        nodes = nodes / self.node_count.clamp(min=1)[None, :, None]
        # Message passing per example: PyG works on one graph at a time, so batch by stacking graphs
        offsets = torch.arange(B, device=x.device) * self.n_nodes
        # Edge index for the stacked graphs
        ei = torch.cat([self.edge_index + o for o in offsets], dim=1)
        # Weights repeated
        ew = self.edge_weight.repeat(B)
        # Stacked node features
        mixed = nodes.reshape(B * self.n_nodes, self.hidden)
        # Each layer is one hop, with a residual connection
        for conv in self.convs:
            # By kind
            if self.kind == "gcn":
                # Weighted
                out = conv(mixed, ei, edge_weight=ew)
            elif self.kind == "gat":
                # Weight as a feature
                out = conv(mixed, ei, edge_attr=ew[:, None])
            else:
                # Unweighted
                out = conv(mixed, ei)
            # Residual connection and nonlinearity
            mixed = torch.relu(out) + mixed
        # Back to (B, n_nodes, hidden)
        mixed = mixed.reshape(B, self.n_nodes, self.hidden)
        # Each meter reads its node
        node_state = mixed[:, self.meter_node, :]
        # Head input: own state and node state
        h = torch.cat([own, node_state], dim=-1)
        # Magnitude
        mag = self.head_mag(h)
        # Angle pair
        ang = self.head_ang(h).reshape(B, M, self.C, 2)
        # Head output
        out = torch.cat([mag[..., None], ang], dim=-1)
        # Plain forecast
        if not self.predict_delta:
            # Done
            return out
        # Persistence base per meter, from the same histories
        base, ok = persistence_forecast(x, self.C)
        # Zero base where nothing valid was in the history
        keep = torch.cat([ok[..., 0:1], ok[..., 1:2], ok[..., 1:2]], dim=-1)
        # Add
        return out + torch.where(keep, base, torch.zeros_like(base))


def persistence_forecast(x, n_channels):
    """
    The persistence baseline. Per channel, the magnitude is copied from
    the last history slot with a valid magnitude and the (cos, sin) pair
    from the last slot with a valid angle, found separately. x is
    (B, H, C * F) or (B, H, M, C * F). Returns (forecast, valid) with
    forecast (..., C, 3) and valid (..., C, 2) bool, False where the
    history held nothing to copy.
    """
    # Unflatten channels and features
    xf = x.reshape(x.shape[:-1] + (n_channels, F))
    # History length
    H = xf.shape[1]
    # Positions along the history, shaped to broadcast against (B, H, ..., C)
    pos = torch.arange(H, device=x.device).view(*([1, H] + [1] * (xf.dim() - 3)))
    # Magnitude validity flag per slot (input feature 3 is mag_valid: present and finite)
    mag_ok = xf[..., 3] > 0.5
    # Angle validity flag per slot
    ang_ok = xf[..., 4] > 0.5
    # Last valid position per part, -1 where none
    last_mag = torch.where(mag_ok, pos, torch.full_like(pos, -1)).max(dim=1).values
    # Angle
    last_ang = torch.where(ang_ok, pos, torch.full_like(pos, -1)).max(dim=1).values
    # Validity of the copy
    valid = torch.stack([last_mag >= 0, last_ang >= 0], dim=-1)
    # Gather magnitude from its slot, index 0 where none (value unused, flagged invalid)
    im = last_mag.clamp(min=0)[:, None, ..., None].expand(-1, 1, *last_mag.shape[1:], 1)
    # Magnitude
    mag = torch.gather(xf[..., 0:1], 1, im).squeeze(1)
    # Gather cos and sin from their slot
    ia = last_ang.clamp(min=0)[:, None, ..., None].expand(-1, 1, *last_ang.shape[1:], 2)
    # Angle pair
    ang = torch.gather(xf[..., 1:3], 1, ia).squeeze(1)
    # Forecast
    return torch.cat([mag, ang], dim=-1), valid


# -----------------------------------------------------------------------------
# Loss and metrics
# -----------------------------------------------------------------------------

def masked_loss(y_hat, y, y_valid, angle_weight=1.0):
    """
    Squared error on magnitude over valid magnitudes, plus angle_weight
    times squared error on (cos, sin) over valid angles. Each term is a
    mean over its own valid count. Shapes (..., C, 3) and (..., C, 2).
    Returns (total, mag_term, ang_term) as tensors.
    """
    # Validity as floats
    mv = y_valid[..., 0].float()
    # Angle validity
    av = y_valid[..., 1].float()
    # Magnitude squared error
    mag_se = (y_hat[..., 0] - y[..., 0]) ** 2
    # Angle squared error summed over cos and sin
    ang_se = ((y_hat[..., 1:3] - y[..., 1:3]) ** 2).sum(dim=-1)
    # Means over valid counts, guarded against zero
    mag_term = (mag_se * mv).sum() / mv.sum().clamp(min=1.0)
    # Angle term
    ang_term = (ang_se * av).sum() / av.sum().clamp(min=1.0)
    # Total
    return mag_term + angle_weight * ang_term, mag_term, ang_term


def residuals(y_hat, y, y_valid):
    """
    Residual arrays from a forecast:
      mag_res  |forecast - actual| scaled magnitude, NaN where invalid
      ang_res  angle between the directions of the forecast and actual
               (cos, sin) vectors in degrees, NaN where invalid; a forecast
               vector of exactly zero length has no direction and scores 90
    Inputs are tensors of shape (..., C, 3) and (..., C, 2); outputs numpy.
    """
    # Magnitude residual
    mag = (y_hat[..., 0] - y[..., 0]).abs()
    # Forecast vector length
    norm = y_hat[..., 1:3].norm(dim=-1)
    # Cosine of the angle between the two directions; the actual vector is unit length where valid.
    # Dividing by the length keeps the direction for any nonzero vector; a zero vector has no
    # direction, gives cos 0, and so scores 90 degrees, which is the documented policy
    cos_err = ((y_hat[..., 1:3] * y[..., 1:3]).sum(dim=-1) / norm.clamp(min=1e-12)).clamp(-1.0, 1.0)
    # Exactly zero forecast vectors: cos 0
    cos_err = torch.where(norm > 0, cos_err, torch.zeros_like(cos_err))
    # Degrees
    ang = torch.rad2deg(torch.acos(cos_err))
    # NaN where invalid
    mag = torch.where(y_valid[..., 0], mag, torch.full_like(mag, float("nan")))
    # NaN where invalid
    ang = torch.where(y_valid[..., 1], ang, torch.full_like(ang, float("nan")))
    # Numpy
    return mag.cpu().numpy(), ang.cpu().numpy()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def make_model(kind, manifest, graph=None, edges="true", edge_weights="admittance", hidden=64, embed=8, layers=1, dropout=0.0, heads=4, seed=0, gnn_layers=1, predict_delta=True):
    """Build a model by name: "gru", "gcn", "gat", or "sage"."""
    # Sizes
    M, C = len(manifest["meters"]), len(manifest["channels"])
    # Plain GRU
    if kind == "gru":
        # Build
        return MeterGRU(M, C, hidden, embed, layers, dropout, predict_delta)
    # GNNs need the graph
    if graph is None:
        # Raise
        raise ValueError("a graph is required for a GNN")
    # Meter placement
    meter_node = meter_node_index(graph, manifest)
    # Edges
    edge_index, edge_weight = build_edges(graph, edges, edge_weights, seed)
    # Build
    return GraphForecaster(kind, M, C, len(graph["nodes"]), meter_node, edge_index, edge_weight, hidden, embed, layers, dropout, heads, gnn_layers, predict_delta)


def new_sums():
    """Empty accumulator for pooled losses: squared-error sums and valid counts."""
    # Dictionary
    return {"mag_se": 0.0, "mag_n": 0.0, "ang_se": 0.0, "ang_n": 0.0, "per_meter_se": {}, "per_meter_n": {}}


def pooled(sums, angle_weight):
    """Pooled losses from the sums: each term is its error sum over its valid count."""
    # Magnitude
    mag = sums["mag_se"] / max(sums["mag_n"], 1.0)
    # Angle
    ang = sums["ang_se"] / max(sums["ang_n"], 1.0)
    # Total
    return {"loss": mag + angle_weight * ang, "mag": mag, "ang": ang, "mag_n": sums["mag_n"], "ang_n": sums["ang_n"]}


def run_batches(model, batches, graph_mode, angle_weight, sums, optimizer=None):
    """
    One pass over an iterable of tensor batches already on the device.
    With an optimizer, trains; without, evaluates. Adds squared-error sums
    and valid counts into `sums` (see new_sums), so the caller can pool
    over any number of pieces without batch-size bias.
    """
    # Mode
    model.train(optimizer is not None)
    # Loop
    for x, y, y_valid, extra in batches:
        # Forward
        y_hat = model(x) if graph_mode else model(x, extra)
        # Loss for the gradient step, batch-level means are fine for optimization
        loss, _, _ = masked_loss(y_hat, y, y_valid, angle_weight)
        # Train step
        if optimizer is not None:
            # Clear
            optimizer.zero_grad()
            # Backward
            loss.backward()
            # Clip, keeps the GRU stable
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Step
            optimizer.step()
        # Bookkeeping without gradient
        with torch.no_grad():
            # Validity as floats
            mv = y_valid[..., 0].float()
            # Angle validity
            av = y_valid[..., 1].float()
            # Squared errors
            mag_se = ((y_hat[..., 0] - y[..., 0]) ** 2) * mv
            # Angle
            ang_se = ((y_hat[..., 1:3] - y[..., 1:3]) ** 2).sum(dim=-1) * av
            # Pooled sums
            sums["mag_se"] += float(mag_se.sum()); sums["mag_n"] += float(mv.sum())
            # Angle
            sums["ang_se"] += float(ang_se.sum()); sums["ang_n"] += float(av.sum())
            # Per meter, magnitude only
            if graph_mode:
                # Every meter in every example
                per_se = mag_se.sum(dim=(0, 2)).cpu().numpy()
                # Counts
                per_n = mv.sum(dim=(0, 2)).cpu().numpy()
                # Meter ids are positions
                ids = range(len(per_se))
            else:
                # One meter per example
                per_se = mag_se.sum(dim=1).cpu().numpy()
                # Counts
                per_n = mv.sum(dim=1).cpu().numpy()
                # Meter ids from the batch
                ids = extra.cpu().numpy()
            # Add
            for mi, se, n in zip(ids, per_se, per_n):
                # Error
                sums["per_meter_se"][int(mi)] = sums["per_meter_se"].get(int(mi), 0.0) + float(se)
                # Count
                sums["per_meter_n"][int(mi)] = sums["per_meter_n"].get(int(mi), 0.0) + float(n)


def make_dataset(piece, graph_mode, history, horizon, max_gap, min_meters=1):
    """The right dataset kind over one block-month piece, or None when it has no window."""
    # Unpack
    _, _, _, X, mag_valid, ang_valid = piece
    # Dataset
    if graph_mode:
        # Synchronized
        ds = GraphWindowDataset(X, mag_valid, ang_valid, history, horizon, max_gap, min_meters)
    else:
        # Per meter
        ds = sd.WindowDataset(X, mag_valid, ang_valid, history, horizon, max_gap)
    # Empty
    return ds if len(ds) else None


def evaluate(model, features_dir, splits, split_name, scale, manifest, device, graph_mode, history, horizon, max_gap, batch_size, angle_weight, blocks=None):
    """
    Pooled loss over a split, or over the listed block indices of it:
    error sums divided by valid counts, so batch sizes do not bias it.
    Raises if the split has no valid magnitude target at all. Returns a
    dictionary with loss, mag, ang, the counts, and per_meter_mag.
    """
    # Accumulator
    sums = new_sums()
    # Blocks to use
    chosen = splits[split_name] if blocks is None else [splits[split_name][i] for i in blocks]
    # Subset split
    subset = {split_name: chosen}
    # Generator, unused since evaluation does not shuffle
    rng = np.random.default_rng(0)
    # No gradient
    with torch.no_grad():
        # Loop over pieces
        for piece in sd.iter_blocks(features_dir, subset, split_name, scale, manifest=manifest):
            # Dataset
            ds = make_dataset(piece, graph_mode, history, horizon, max_gap)
            # Skip empty
            if ds is None:
                # Next
                continue
            # Pass, in order
            run_batches(model, iter_batches(ds, batch_size, False, rng, device), graph_mode, angle_weight, sums)
    # Nothing to evaluate on is an error, never a perfect score
    if sums["mag_n"] == 0:
        # Raise
        raise RuntimeError(f"no valid targets in split {split_name} blocks {blocks}: cannot evaluate")
    # Pooled
    out = pooled(sums, angle_weight)
    # Per-meter magnitude loss
    out["per_meter_mag"] = {manifest["meters"][mi]: sums["per_meter_se"][mi] / max(sums["per_meter_n"][mi], 1.0) for mi in sorted(sums["per_meter_se"])}
    # Return
    return out


# Options that define the experiment; a resumed run must match them
EXPERIMENT_KEYS = ("kind", "history", "horizon", "max_gap", "angle_weight", "hidden", "embed", "layers", "dropout", "gnn_layers", "heads", "edges", "edge_weights", "min_meters", "lr", "weight_decay", "batch_size", "seed", "train_blocks", "splits_file", "predict_delta")


def save_atomic(body, path):
    """torch.save to a temporary file, then rename, so a crash never leaves a half file."""
    # Temporary
    tmp = path + ".tmp"
    # Save
    torch.save(body, tmp)
    # Replace
    os.replace(tmp, path)


def rng_states():
    """Every random state that shapes a training run: NumPy is passed separately, this is torch's."""
    # CPU state, and CUDA states when a GPU is in use
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_states(states):
    """Put torch's random states back."""
    # CPU: torch.load with map_location may have moved the state tensor to the GPU, and set_rng_state wants it on the CPU
    if states and states.get("torch_cpu") is not None:
        # Restore from a CPU tensor
        torch.set_rng_state(states["torch_cpu"].cpu())
    # CUDA, only when the saved states match the devices present
    if states and states.get("torch_cuda") is not None and torch.cuda.is_available() and len(states["torch_cuda"]) == torch.cuda.device_count():
        # Restore, each state as a CPU tensor
        torch.cuda.set_rng_state_all([state.cpu() for state in states["torch_cuda"]])


def save_checkpoint(path, model, optimizer, options, epoch, best, stale, rng):
    """Write a full checkpoint: weights, optimizer, options, progress, the shuffling state, and torch's random states."""
    # Body
    body = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "options": options, "epoch": epoch, "best": best, "stale": stale, "rng": rng.bit_generator.state, "torch_rng": rng_states()}
    # Write
    save_atomic(body, path)


def load_checkpoint(path, model, options, optimizer=None, device="cpu", rng=None):
    """
    Load a checkpoint. Refuses to resume when any experiment-defining
    option differs from the saved one. Restores the optimizer and the
    shuffling state when given. Returns (epoch, best, stale).
    """
    # Load
    body = torch.load(path, map_location=device, weights_only=False)
    # Saved options
    saved = body.get("options", {})
    # Compare the keys that define the experiment
    diffs = {k: (saved.get(k), options.get(k)) for k in EXPERIMENT_KEYS if saved.get(k) != options.get(k)}
    # Any difference is a different experiment
    if diffs:
        # Raise with the list
        raise ValueError(f"checkpoint {path} was made with different options: {diffs}. Use a new checkpoint path.")
    # Weights
    model.load_state_dict(body["model"])
    # Optimizer
    if optimizer is not None and "optimizer" in body:
        # Restore
        optimizer.load_state_dict(body["optimizer"])
    # Shuffling state
    if rng is not None and "rng" in body:
        # Restore
        rng.bit_generator.state = body["rng"]
    # Torch's random states, for dropout; the batch shuffle uses the numpy generator above
    restore_rng_states(body.get("torch_rng"))
    # Return
    return body.get("epoch", 0), body.get("best", float("inf")), body.get("stale", 0)


def train(model, features_dir, splits, scale, manifest, device, options, checkpoint_path, log):
    """
    Train for options["epochs"] epochs over the train split, one block-
    month piece at a time in a shuffled order, with early stopping on
    validation block 0 and a checkpoint every epoch. Resumes from the
    checkpoint if it exists and its options match. On return the model
    holds the best weights, not the last epoch's. `log` is a function
    that takes a string. Returns the best validation loss.
    """
    # Graph models take synchronized batches
    graph_mode = options["kind"] != "gru"
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    # Generator for the piece order
    rng = np.random.default_rng(options["seed"])
    # Start
    start_epoch, best, stale = 0, float("inf"), 0
    # Resume
    if os.path.exists(checkpoint_path):
        # Load, options checked
        start_epoch, best, stale = load_checkpoint(checkpoint_path, model, options, optimizer, device, rng)
        # Say so
        log(f"resumed from {checkpoint_path} at epoch {start_epoch}, best {best:.5f}, {stale} stale")
    # Best-weights path
    best_path = checkpoint_path.replace(".pt", "_best.pt")
    # Loop over epochs
    for epoch in range(start_epoch, options["epochs"]):
        # Clock
        t0 = time.time()
        # All train pieces as (block index, month), one month file each
        order = [(bi, month) for bi, block in enumerate(splits["train"]) for month in sd.months_in_block(block)]
        # Shuffle
        rng.shuffle(order)
        # Accumulator for the epoch's training loss
        sums = new_sums()
        # Loop over pieces
        for bi, month in order:
            # Load exactly this block-month
            piece = sd.load_piece(features_dir, splits["train"][bi], month, scale, manifest=manifest)
            # Skip empty
            if piece is None:
                # Next
                continue
            # Dataset, the piece tuple needs the two label slots the function ignores
            ds = make_dataset((None, None) + piece, graph_mode, options["history"], options["horizon"], options["max_gap"], options.get("min_meters", 1))
            # Skip empty
            if ds is None:
                # Next
                continue
            # Train pass over shuffled batches
            run_batches(model, iter_batches(ds, options["batch_size"], True, rng, device), graph_mode, options["angle_weight"], sums, optimizer)
        # An epoch that saw no valid training target is an error, not a free pass
        if sums["mag_n"] == 0:
            # Raise
            raise RuntimeError("no valid training targets in this epoch: check the train blocks, the gap rule, and the feature files")
        # Pooled training loss
        tr = pooled(sums, options["angle_weight"])
        # Validation on block 0 only, raises if empty
        val = evaluate(model, features_dir, splits, "validation", scale, manifest, device, graph_mode, options["history"], options["horizon"], options["max_gap"], options["batch_size"], options["angle_weight"], blocks=[0])
        # Report
        log(f"epoch {epoch + 1}: train {tr['loss']:.5f}  val {val['loss']:.5f} (mag {val['mag']:.5f} ang {val['ang']:.5f}, {int(val['mag_n'])} targets)  {time.time() - t0:.0f} s")
        # Per-meter report, worst three
        worst = sorted(val["per_meter_mag"].items(), key=lambda kv: -kv[1])[:3]
        # Print
        log("  worst meters by magnitude loss: " + ", ".join(f"{m} {v:.4f}" for m, v in worst))
        # Improvement
        if val["loss"] < best - options["min_delta"]:
            # Record
            best, stale = val["loss"], 0
            # Best weights kept separately, atomically
            save_atomic(model.state_dict(), best_path)
        # No improvement
        else:
            # Count
            stale += 1
        # Checkpoint every epoch
        save_checkpoint(checkpoint_path, model, optimizer, options, epoch + 1, best, stale, rng)
        # Early stop
        if stale >= options["patience"]:
            # Say so
            log(f"early stop after {stale} epochs without improvement")
            # Done
            break
    # Leave the best weights in the model, not the last epoch's
    if os.path.exists(best_path):
        # Load
        model.load_state_dict(torch.load(best_path, map_location=device))
        # Say so
        log(f"restored best weights from {best_path} (val {best:.5f})")
    # Return
    return best


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

def score_split(model, features_dir, splits, split_name, scale, manifest, device, options, out_dir, log):
    """
    Write one residual table per block of a split:
      <out_dir>/<label>.parquet  index t_utc, columns "<meter>|<channel>|<name>"
        for name in mag_res, ang_res, mag_valid, ang_valid, and the same with
        a pers_ prefix for the persistence baseline
      <out_dir>/<label>.json     summary: slots scored, median and 99th
        percentile of mag_res and ang_res, options used
    Every slot that is the target of a usable window gets a row; other
    slots are absent, never written as zero.
    """
    # Graph models take synchronized batches
    graph_mode = options["kind"] != "gru"
    # Names
    meters, channels = manifest["meters"], manifest["channels"]
    # Sizes
    M, C = len(meters), len(channels)
    # Output folder
    os.makedirs(out_dir, exist_ok=True)
    # Eval mode
    model.eval()
    # Collect per block label
    tables = {}
    # No gradient
    with torch.no_grad():
        # Loop over pieces
        for label, month, t, X, mag_valid, ang_valid in sd.iter_blocks(features_dir, splits, split_name, scale, manifest=manifest):
            # Dataset, every meter kept so residuals exist wherever a meter is usable
            ds = make_dataset((label, month, t, X, mag_valid, ang_valid), graph_mode, options["history"], options["horizon"], options["max_gap"], min_meters=1)
            # Nothing
            if ds is None:
                # Next
                continue
            # Batches in order
            loader = iter_batches(ds, options["batch_size"], False, np.random.default_rng(0), device)
            # Result arrays over the piece, NaN where no residual
            res = {k: np.full((len(t), M, C), np.nan, dtype=np.float32) for k in ("mag_res", "ang_res", "pers_mag_res", "pers_ang_res")}
            # Validity arrays, model and persistence
            val = {k: np.zeros((len(t), M, C), dtype=bool) for k in ("mag_valid", "ang_valid", "pers_mag_valid", "pers_ang_valid")}
            # Position in the dataset
            pos = 0
            # Loop over batches
            for x, y, y_valid, extra in loader:
                # Forecast
                y_hat = model(x) if graph_mode else model(x, extra)
                # Persistence forecast and whether the history held anything to copy
                y_pers, p_ok = persistence_forecast(x, C)
                # Persistence validity: a target to compare with and a value to copy
                p_valid = y_valid & p_ok
                # Residuals
                mr, ar = residuals(y_hat, y, y_valid)
                # Persistence residuals under its own validity
                pmr, par = residuals(y_pers, y, p_valid)
                # Place into the piece arrays
                B = len(x)
                # Graph: extra is the target slot, every meter at once
                if graph_mode:
                    # Slots
                    slots = extra.cpu().numpy()
                    # Write
                    res["mag_res"][slots] = mr; res["ang_res"][slots] = ar
                    # Persistence
                    res["pers_mag_res"][slots] = pmr; res["pers_ang_res"][slots] = par
                    # Validity
                    val["mag_valid"][slots] = y_valid[..., 0].cpu().numpy(); val["ang_valid"][slots] = y_valid[..., 1].cpu().numpy()
                    # Persistence validity
                    val["pers_mag_valid"][slots] = p_valid[..., 0].cpu().numpy(); val["pers_ang_valid"][slots] = p_valid[..., 1].cpu().numpy()
                # Meter: extra is the meter index, the slot comes from the dataset pairs
                else:
                    # Pairs for this batch
                    pairs = ds.pairs[pos:pos + B]
                    # Target slots
                    slots = pairs[:, 1] + options["history"] + options["horizon"] - 1
                    # Meters
                    mis = pairs[:, 0]
                    # Write
                    res["mag_res"][slots, mis] = mr; res["ang_res"][slots, mis] = ar
                    # Persistence
                    res["pers_mag_res"][slots, mis] = pmr; res["pers_ang_res"][slots, mis] = par
                    # Validity
                    val["mag_valid"][slots, mis] = y_valid[..., 0].cpu().numpy(); val["ang_valid"][slots, mis] = y_valid[..., 1].cpu().numpy()
                    # Persistence validity
                    val["pers_mag_valid"][slots, mis] = p_valid[..., 0].cpu().numpy(); val["pers_ang_valid"][slots, mis] = p_valid[..., 1].cpu().numpy()
                # Advance
                pos += B
            # Rows that got any residual
            scored = np.isfinite(res["mag_res"]).any(axis=(1, 2))
            # Columns
            cols = {}
            # Loop over meters and channels
            for mi, meter in enumerate(meters):
                # Channels
                for ci, ch in enumerate(channels):
                    # Residuals
                    for k in res:
                        # Column
                        cols[f"{meter}|{ch}|{k}"] = res[k][scored, mi, ci]
                    # Validity
                    for k in val:
                        # Column
                        cols[f"{meter}|{ch}|{k}"] = val[k][scored, mi, ci]
            # Table
            table = pd.DataFrame(cols, index=pd.to_datetime(t[scored], unit="s", utc=True))
            # Index name
            table.index.name = "t_utc"
            # Append to the block
            tables.setdefault(label, []).append(table)
            # Progress
            log(f"  scored {label} {month}: {int(scored.sum())} slots")
    # Write per block
    for label, parts in tables.items():
        # Stack months
        table = pd.concat(parts).sort_index()
        # Path
        path = os.path.join(out_dir, f"{label}.parquet")
        # Save
        table.to_parquet(path)
        # Summary over all mag_res and ang_res columns
        mag_cols = [c for c in table.columns if c.endswith("|mag_res")]
        # Angle columns
        ang_cols = [c for c in table.columns if c.endswith("|ang_res")]
        # Values
        mag_vals = table[mag_cols].to_numpy().ravel(); mag_vals = mag_vals[np.isfinite(mag_vals)]
        # Angles
        ang_vals = table[ang_cols].to_numpy().ravel(); ang_vals = ang_vals[np.isfinite(ang_vals)]
        # Persistence
        pm_cols = [c for c in table.columns if c.endswith("|pers_mag_res")]
        # Values
        pm_vals = table[pm_cols].to_numpy().ravel(); pm_vals = pm_vals[np.isfinite(pm_vals)]
        # Summary
        summary = {
            # Label
            "label": label, "slots": int(len(table)), "model": options["kind"],
            # Magnitude residual in scaled units
            "mag_res_median": float(np.median(mag_vals)) if len(mag_vals) else None,
            # 99th percentile
            "mag_res_p99": float(np.percentile(mag_vals, 99)) if len(mag_vals) else None,
            # Angle residual in degrees
            "ang_res_median": float(np.median(ang_vals)) if len(ang_vals) else None,
            # 99th percentile
            "ang_res_p99": float(np.percentile(ang_vals, 99)) if len(ang_vals) else None,
            # Persistence for comparison
            "pers_mag_res_median": float(np.median(pm_vals)) if len(pm_vals) else None,
            # Options
            "options": options,
        }
        # Write
        with open(os.path.join(out_dir, f"{label}.json"), "w") as f:
            # Pretty JSON
            json.dump(summary, f, indent=2, default=str)
        # Report
        log(f"wrote {path}: {len(table)} slots, mag_res median {summary['mag_res_median']}, persistence {summary['pers_mag_res_median']}")
