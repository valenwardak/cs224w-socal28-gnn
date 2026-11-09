# =============================================================================
# File:        main_physicsinputcheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-09
#
# Purpose:     Check the two safeguards around the physics inputs that
#              need torch to test, before any training run uses them:
#              1. A masked forecast cannot depend on the physics inputs.
#                 The physics residual was computed from every meter's
#                 readings, the hidden meter's included, so in masked mode
#                 the model must zero those features in every masked
#                 forward (training, evaluation and scoring). Test: build
#                 a small masked graph model with nine features per
#                 channel, forecast a random batch with a mask, change the
#                 four physics features to other random values, forecast
#                 again, and require the two forecasts to be identical;
#                 then require that an unmasked forward of the same model
#                 does change, so the test can tell the difference.
#              2. A checkpoint made with other physics divisors or another
#                 feature order is refused. Test: save a checkpoint with
#                 one set of options, then try to load it with the
#                 physics_scale changed, with the features reordered, with
#                 physics turned off, and with a different feature count;
#                 every attempt must raise naming the option.
#              Runs on random tensors, no data files needed. Needs torch
#              and torch_geometric, so it runs on Colab or a machine with
#              them, not where only numpy is installed.
#
# Usage (from the repo root, where torch is installed):
#   python checks/main_physicsinputcheck.py
#
# Functions:
#   parse_args()          command-line options with defaults
#   tiny_graph(M)         a small graph dictionary for make_model
#   check_masked(args)    safeguard 1
#   check_checkpoint(args)
#                         safeguard 2
#   main()                run both and report
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Standard library: import path for the repo root
import sys
# Standard library: temporary folder for the checkpoint
import tempfile
# Third party: numerics
import numpy as np
# Third party: torch
import torch

# Repo root on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data side, for the feature constants
import socal_modeldata as sd
# Models
import socal_models as sm


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Check the physics-input safeguards that need torch: masked invariance and checkpoint refusal")
    # Meters
    parser.add_argument("--meters", type=int, default=4, help="Meters in the synthetic graph")
    # Channels
    parser.add_argument("--channels", type=int, default=6, help="Channels per meter")
    # History
    parser.add_argument("--history", type=int, default=12, help="History slots per window")
    # Batch
    parser.add_argument("--batch", type=int, default=5, help="Windows per batch")
    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # Parse
    return parser.parse_args()


def tiny_graph(M):
    """A small graph in the layout make_model reads (graph.json): M nodes in a chain with unit admittance, meter i on node i."""
    # Nodes, one per meter
    nodes = [{"name": f"node_{i}"} for i in range(M)]
    # Chain edges by node index
    edges = [{"source": i, "target": i + 1, "admittance_abs": 1.0} for i in range(M - 1)]
    # Meter placement
    meters = {f"egauge_{i}": {"node": i} for i in range(M)}
    # Return
    return {"nodes": nodes, "edges": edges, "meters": meters}


def check_masked(args):
    """Safeguard 1: a masked forecast is identical whatever the physics features hold; an unmasked one is not."""
    # Sizes
    M, C, H, B = args.meters, args.channels, args.history, args.batch
    # Manifest
    manifest = {"meters": [f"egauge_{i}" for i in range(M)], "channels": [f"ch{i}" for i in range(C)]}
    # Nine features per channel
    nf = sd.F + len(sd.PHYSICS_FEATURES)
    # Masked model, every meter maskable, fixed graph
    model = sm.make_model("gcn", manifest, tiny_graph(M), edges="true", hidden=16, embed=4, seed=args.seed, masked=True, maskable_meters=manifest["meters"], n_features=nf)
    # Evaluation mode, no dropout
    model.eval()
    # Random inputs (B, H, M, C, nf) with valid flags set
    g = torch.Generator().manual_seed(args.seed)
    # Values
    xf = torch.rand(B, H, M, C, nf, generator=g)
    # Flags to one so persistence has something to copy
    xf[..., 3] = 1.0
    # Angle flag
    xf[..., 4] = 1.0
    # Flat layout the model takes
    x1 = xf.reshape(B, H, M, C * nf)
    # Second input: the same standard features, other physics features
    xf2 = xf.clone()
    # New physics values and flags
    xf2[..., sd.F:] = torch.rand(B, H, M, C, len(sd.PHYSICS_FEATURES), generator=g)
    # Flat
    x2 = xf2.reshape(B, H, M, C * nf)
    # Mask: hide meter 0 in every window
    mask = torch.zeros(B, M, dtype=torch.bool)
    # Hide
    mask[:, 0] = True
    # Masked forecasts
    with torch.no_grad():
        # First
        y1 = model(x1, mask)
        # Second
        y2 = model(x2, mask)
        # Unmasked forecasts, for the contrast
        u1 = model(x1)
        # Second
        u2 = model(x2)
    # Masked: identical
    d_masked = float((y1 - y2).abs().max())
    # Unmasked: different
    d_open = float((u1 - u2).abs().max())
    # Say
    print(f"masked forward: largest difference from changing the physics features {d_masked:.3e} (must be 0); unmasked forward: {d_open:.3e} (must be > 0)")
    # Verdict
    if d_masked != 0.0:
        # Fail
        raise AssertionError("a masked forecast changed when only the physics features changed")
    # The contrast must exist
    if d_open == 0.0:
        # Fail
        raise AssertionError("the unmasked forecast did not change with the physics features, so the test cannot tell anything")
    # Also the masked_forecast path used by scoring
    with torch.no_grad():
        # Every maskable meter in turn
        f1, _ = sm.masked_forecast(model, x1, 2)
        # Again with other physics values
        f2, _ = sm.masked_forecast(model, x2, 2)
    # Identical
    d_score = float((f1 - f2).abs().max())
    # Say
    print(f"masked_forecast (scoring path): largest difference {d_score:.3e} (must be 0)")
    # Verdict
    if d_score != 0.0:
        # Fail
        raise AssertionError("the scoring path's masked forecast changed with the physics features")


def check_checkpoint(args):
    """Safeguard 2: a checkpoint is refused when the physics divisors, the feature order, the physics identity or the feature count differ."""
    # Sizes
    M, C = args.meters, args.channels
    # Manifest
    manifest = {"meters": [f"egauge_{i}" for i in range(M)], "channels": [f"ch{i}" for i in range(C)]}
    # Nine features
    nf = sd.F + len(sd.PHYSICS_FEATURES)
    # A plain GRU is enough for the option check
    model = sm.make_model("gru", manifest, hidden=8, embed=2, seed=args.seed, n_features=nf)
    # Options as main_gnn.py records them for a physics run
    options = {k: None for k in sm.EXPERIMENT_KEYS}
    # Fill the ones that matter here
    options.update({"kind": "gru", "physics": "physics_manifest.json:abc", "n_features": nf, "features": list(sd.FEATURES) + list(sd.PHYSICS_FEATURES), "physics_scale": [0.005, 0.003]})
    # Temporary checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        # Path
        path = os.path.join(tmp, "checkpoint.pt")
        # Save
        sm.save_checkpoint(path, model, torch.optim.Adam(model.parameters()), options, 1, 1.0, 0, np.random.default_rng(0))
        # The same options load
        sm.load_checkpoint(path, model, dict(options))
        # Say
        print("checkpoint with identical options: loads")
        # Each change must be refused
        changes = {"physics_scale": [0.005, 0.0031], "features": list(sd.FEATURES) + ["phys_b", "phys_b_valid", "phys_a", "phys_a_valid"], "physics": "none", "n_features": sd.F}
        # Loop
        for key, value in changes.items():
            # Changed options
            other = dict(options)
            # Change
            other[key] = value
            # Must raise naming the key
            try:
                # Attempt
                sm.load_checkpoint(path, model, other)
                # No error is a failure
                raise AssertionError(f"checkpoint loaded although {key} differed")
            # Expected
            except ValueError as e:
                # The key must be named
                if key not in str(e):
                    # Fail
                    raise AssertionError(f"refusal for {key} does not name it: {e}")
                # Say
                print(f"changed {key}: refused as required")


def main():
    """Run both checks."""
    # Options
    args = parse_args()
    # Seed
    torch.manual_seed(args.seed)
    # Safeguard 1
    check_masked(args)
    # Safeguard 2
    check_checkpoint(args)
    # Done
    print("both safeguards hold")


# Entry point
if __name__ == "__main__":
    # Run
    main()
