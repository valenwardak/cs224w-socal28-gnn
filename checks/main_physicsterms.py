# =============================================================================
# File:        main_physicsterms.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-08
#
# Purpose:     Break the physics baseline's residual into its terms over one
#              window, so we can see which measurements the readings and
#              the normal wiring disagree on. main_physics.py reports one
#              number per slot, the sum of the normalized residuals of
#              every equation term. This check rebuilds the same reference
#              model and measurement plan with socal_physics (unchanged),
#              solves the same state estimate slot by slot, and keeps every
#              term: one per measured bus-phase voltage, one per phase of
#              every measured injection (the clamp currents behind it,
#              with their signs), and one per phase of every zero-injection
#              node. At every solved slot the sum of the terms is checked
#              against socal_physics.objective_value and a mismatch stops
#              the run. For each term over the window it prints: what the
#              term is, how often its readings were available, its median
#              and 99th percentile contribution in percent of nominal, its
#              share of the window's total residual, the median measured
#              and fitted magnitudes in volts or amps, and the median
#              magnitude of the complex difference between measured and
#              fitted (so a disagreement in angle counts, not only in
#              size). Terms are ranked by share. A large share names a
#              mismatch between a reading and the wiring model; it does
#              not by itself say the meter is at fault, and a small one
#              does not by itself confirm a clamp's sign, since the fitted
#              state can absorb part of a mismatch.
#
#              An hourly line of the total residual per term count is
#              printed first so the reader can see where in the window the
#              mismatch sits. With --compare-solvers the cvxpy solver
#              (Caltech's objective) is run on every --step-th slot as
#              well and the two are compared: per-slot total difference
#              and whether they rank the same terms highest, both on the
#              slots that both solvers solved. Nothing is written.
#
#              The window is given in UTC with --start and --end, or as a
#              fixed-PST calendar day with --day (08:00 UTC to 08:00 UTC
#              the next day, as the block labels).
#
# Usage (from the repo root):
#   python checks/main_physicsterms.py --day 2025-06-12
#   python checks/main_physicsterms.py --day 2025-08-10 --compare-solvers --step 60
#   python checks/main_physicsterms.py --start 2025-07-29T08:00:00 --end 2025-07-29T20:00:00 --top 12
#
# Functions:
#   parse_args()                           command-line options with defaults
#   window_from_args(args)                 (label, start_utc, end_utc) from --day or --start/--end
#   term_labels(model, plan)               a name for every voltage, injection and zero-injection term
#   solve_window(model, plan, table, solver, weight, step, freq_tolerance)
#                                          per-slot state estimates and per-term residuals
#   term_table(res, labels, weight)        per-term summary over the window
#   print_hourly(res)                      total residual per hour beside the term counts
#   print_terms(table, top)                the ranked per-term lines
#   restrict(res, keep)                    a result limited to chosen slots
#   compare_solvers(res_a, res_b, name_a, name_b, v_labels, i_labels, i_kind)
#                                          per-slot totals and top-term agreement of two solvers on their shared solved slots
#   main()                                 build, solve, print
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: paths
import os
# Standard library: repo root on the import path
import sys
# Third party: numerics
import numpy as np
# Third party: tables and timestamps
import pandas as pd

# This file lives in checks/; the modules are one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The physics baseline's helpers, used as they are
import socal_physics as sp


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Per-term breakdown of the physics baseline's residual over one window")
    # Same inputs as main_physics.py
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet")
    # Topology
    parser.add_argument("--topology-dir", default="topology", help="Caltech topology folder copied into the repo")
    # Caltech code
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech utils folder copied into the repo")
    # Config
    parser.add_argument("--config", default="physics_config.json", help="Reference model and injection map")
    # Window as a fixed-PST day
    parser.add_argument("--day", default=None, help="Fixed-PST calendar day, like 2025-08-10 (08:00 UTC to 08:00 UTC next day)")
    # Or as UTC bounds
    parser.add_argument("--start", default=None, help="Window start, UTC, like 2025-07-29T08:00:00 (used when --day is not given)")
    # End
    parser.add_argument("--end", default=None, help="Window end, UTC, exclusive")
    # Solver
    parser.add_argument("--solver", default="numpy", choices=["numpy", "cvxpy"], help="Solver for the main pass")
    # Solver comparison
    parser.add_argument("--compare-solvers", action="store_true", help="Also run the other solver on the same slots and compare")
    # Step
    parser.add_argument("--step", type=int, default=1, help="Solve every step-th slot, 1 means all (use a larger step with --compare-solvers)")
    # Current weight
    parser.add_argument("--current-weight", type=float, default=None, help="Weight on current residuals, default from config")
    # Frequency tolerance
    parser.add_argument("--freq-tolerance", type=float, default=0.01, help="Hertz; as in main_physics.py")
    # How many terms to print in the ranked list
    parser.add_argument("--top", type=int, default=10, help="Terms printed in the ranked list (all terms are counted)")
    # Parse
    args = parser.parse_args()
    # Window check
    if args.day is None and not (args.start and args.end):
        # Refuse
        parser.error("give --day, or both --start and --end")
    # Step check
    if args.step < 1:
        # Refuse
        parser.error("--step must be at least 1")
    # Done
    return args


def window_from_args(args):
    """Return (label, start_utc, end_utc) for the window, from --day or from --start and --end."""
    # Day given
    if args.day:
        # Start at 08:00 UTC, fixed PST midnight
        start = pd.Timestamp(args.day) + pd.Timedelta(hours=8)
        # End one day later
        end = start + pd.Timedelta(days=1)
        # Texts
        return f"day_{args.day}", start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S")
    # Bounds given
    return f"window_{args.start}_{args.end}".replace(":", ""), args.start, args.end


def term_labels(model, plan):
    """
    A readable name for every term in the order assemble() builds them:
    voltage terms first, then the three phases of every measured
    injection, then the three phases of every zero-injection node.
    Returns (voltage_labels, current_labels, current_kind) where
    current_kind is "injection" or "zero" per current row.
    """
    # Node names
    nodes = model["node_list"]
    # Phases
    phases = sp.PHASES
    # Voltage terms
    v_labels = [f"V {v['meter']}|{v['channel']} node {nodes[v['node_idx']][:30]} phase {phases[v['phase_idx']]}" for v in plan["voltages"]]
    # Current terms
    i_labels, i_kind = [], []
    # Measured injections
    for inj in plan["injections"]:
        # Each phase
        for p in phases:
            # The clamps behind the row
            parts = " ".join(f"{'+' if t['sign'] > 0 else '-'}{t['meter']}|{t['channels'][p]}" for t in inj["terms"])
            # Label
            i_labels.append(f"I inj node {nodes[inj['node_idx']][:30]} phase {p}: {parts}")
            # Kind
            i_kind.append("injection")
    # Zero-injection nodes
    for node_idx in plan["zeros"]:
        # Each phase
        for p in phases:
            # Label
            i_labels.append(f"I zero node {nodes[node_idx][:30]} phase {p}")
            # Kind
            i_kind.append("zero")
    # Done
    return v_labels, i_labels, i_kind


def solve_window(model, plan, table, solver, weight, step, freq_tolerance):
    """
    Solve every step-th slot and keep the per-term residuals. Returns a
    dict with t (slot times), status, total (objective per slot), count
    (terms per slot), V_res and I_res (normalized residual per term and
    slot, NaN where the term was unavailable), V_meas_abs, V_fit_abs,
    I_meas_abs, I_fit_abs (magnitudes in volts and amps), V_ok, I_ok.
    """
    # Admittance matrix
    Y = model["Y"].astype(np.complex128)
    # State size
    n_state = 3 * model["n_nodes"]
    # Measurement arrays
    m = sp.assemble(model, plan, table, freq_tolerance)
    # Slot positions
    positions = np.arange(len(table))[::step]
    # Times
    t = table.index[positions]
    # Term counts
    MV, MI = m["V_meas"].shape[0], m["I_meas"].shape[0]
    # Outputs
    out = {"t": t, "status": [], "total": np.full(len(positions), np.nan), "count": np.zeros(len(positions), dtype=int),
           "V_res": np.full((MV, len(positions)), np.nan), "I_res": np.full((MI, len(positions)), np.nan),
           "V_meas_abs": np.full((MV, len(positions)), np.nan), "V_fit_abs": np.full((MV, len(positions)), np.nan),
           "I_meas_abs": np.full((MI, len(positions)), np.nan), "I_fit_abs": np.full((MI, len(positions)), np.nan),
           "V_diff": np.full((MV, len(positions)), np.nan), "I_diff": np.full((MI, len(positions)), np.nan),
           "V_ok": m["V_ok"][:, positions], "I_ok": m["I_ok"][:, positions]}
    # Solver function
    solve = sp.solve_slot_cvxpy if solver == "cvxpy" else sp.solve_slot_numpy
    # Each slot
    for k, pos in enumerate(positions):
        # Availability
        v_ok, i_ok = m["V_ok"][:, pos], m["I_ok"][:, pos]
        # Counts
        n_v, n_i = int(v_ok.sum()), int(i_ok.sum())
        # No data at all, as run_block decides it
        if n_v == 0 and (n_i == 0 or np.all(m["I_meas"][i_ok, pos] == 0)):
            # Status
            out["status"].append("no_data")
            # Next
            continue
        # Too few
        if n_v == 0:
            # Status
            out["status"].append("too_few")
            # Next
            continue
        # Solve, the same call as run_block
        V_hat = solve(Y, m["V_meas"][v_ok, pos], m["V_idx"][v_ok], m["V_nom"][v_ok], m["I_meas"][i_ok, pos], m["I_rows"][i_ok], m["I_nom"][i_ok], weight, n_state)
        # Failed
        if V_hat is None:
            # Status
            out["status"].append("solver_failed")
            # Next
            continue
        # Status
        out["status"].append("ok")
        # Fitted voltages at the measured rows
        V_fit = V_hat[m["V_idx"]]
        # Fitted injections at the equation rows
        I_fit = Y[m["I_rows"]] @ V_hat
        # Voltage term residuals, normalized, the objective's own formula
        v_res = np.abs(m["V_meas"][:, pos] - V_fit) / m["V_nom"]
        # Current term residuals, normalized and weighted, the objective's own formula
        i_res = weight * np.abs(m["I_meas"][:, pos] - I_fit) / m["I_nom"]
        # Keep only available terms
        out["V_res"][v_ok, k] = v_res[v_ok]
        # Same
        out["I_res"][i_ok, k] = i_res[i_ok]
        # Magnitudes
        out["V_meas_abs"][v_ok, k] = np.abs(m["V_meas"][v_ok, pos])
        # Fitted
        out["V_fit_abs"][v_ok, k] = np.abs(V_fit[v_ok])
        # Currents
        out["I_meas_abs"][i_ok, k] = np.abs(m["I_meas"][i_ok, pos])
        # Fitted
        out["I_fit_abs"][i_ok, k] = np.abs(I_fit[i_ok])
        # Full mismatch, magnitude of the complex difference, so an angle disagreement counts
        out["V_diff"][v_ok, k] = np.abs(m["V_meas"][v_ok, pos] - V_fit[v_ok])
        # Same for currents, in amps
        out["I_diff"][i_ok, k] = np.abs(m["I_meas"][i_ok, pos] - I_fit[i_ok])
        # Total and count from the term formulas
        out["total"][k] = float(v_res[v_ok].sum() + i_res[i_ok].sum())
        # Count
        out["count"][k] = n_v + n_i
        # The same total and count from the baseline's own objective_value; the two must agree
        value, count = sp.objective_value(Y, V_hat, m["V_meas"][v_ok, pos], m["V_idx"][v_ok], m["V_nom"][v_ok], m["I_meas"][i_ok, pos], m["I_rows"][i_ok], m["I_nom"][i_ok], weight)
        # Mismatch stops the check
        if count != out["count"][k] or not np.isclose(value, out["total"][k], rtol=1e-9, atol=1e-12):
            # Refuse
            raise ValueError(f"slot {t[k]}: term sum {out['total'][k]:.9g} over {out['count'][k]} terms does not reproduce objective_value {value:.9g} over {count} terms")
    # Status as array
    out["status"] = np.array(out["status"] + ["no_data"] * (len(positions) - len(out["status"])))
    # Done
    return out


def term_table(res, v_labels, i_labels, i_kind):
    """
    Per-term summary over the solved slots: availability, median and 99th
    percentile contribution in percent, share of the total residual, and
    median measured, fitted and absolute-difference magnitudes.
    """
    # Solved slots
    ok = res["status"] == "ok"
    # Total residual over the window, solved slots
    grand = float(np.nansum(res["total"][ok]))
    # Rows
    rows = []
    # Voltage terms
    for j, lab in enumerate(v_labels):
        # Residuals on solved slots
        r = res["V_res"][j, ok]
        # Available
        have = np.isfinite(r)
        # Row
        rows.append({"term": lab, "kind": "voltage", "unit": "V", "available": float(have.mean()) if ok.any() else 0.0,
                     "median_pct": 100 * float(np.median(r[have])) if have.any() else float("nan"),
                     "p99_pct": 100 * float(np.percentile(r[have], 99)) if have.any() else float("nan"),
                     "share": float(np.nansum(r)) / grand if grand > 0 else float("nan"),
                     "meas": float(np.nanmedian(res["V_meas_abs"][j, ok])) if have.any() else float("nan"),
                     "fit": float(np.nanmedian(res["V_fit_abs"][j, ok])) if have.any() else float("nan"),
                     "diff": float(np.nanmedian(res["V_diff"][j, ok])) if have.any() else float("nan")})
    # Current terms
    for j, lab in enumerate(i_labels):
        # Residuals on solved slots
        r = res["I_res"][j, ok]
        # Available
        have = np.isfinite(r)
        # Row
        rows.append({"term": lab, "kind": i_kind[j], "unit": "A", "available": float(have.mean()) if ok.any() else 0.0,
                     "median_pct": 100 * float(np.median(r[have])) if have.any() else float("nan"),
                     "p99_pct": 100 * float(np.percentile(r[have], 99)) if have.any() else float("nan"),
                     "share": float(np.nansum(r)) / grand if grand > 0 else float("nan"),
                     "meas": float(np.nanmedian(res["I_meas_abs"][j, ok])) if have.any() else float("nan"),
                     "fit": float(np.nanmedian(res["I_fit_abs"][j, ok])) if have.any() else float("nan"),
                     "diff": float(np.nanmedian(res["I_diff"][j, ok])) if have.any() else float("nan")})
    # Table ranked by share
    return pd.DataFrame(rows).sort_values("share", ascending=False).reset_index(drop=True), grand


def print_hourly(res):
    """Print the total residual per term (percent) by hour, with the solved count and the median term count."""
    # Solved
    ok = res["status"] == "ok"
    # Per-term residual in percent
    pct = 100.0 * res["total"] / np.where(res["count"] > 0, res["count"], 1)
    # Hour key in fixed PST
    hour = (pd.DatetimeIndex(res["t"]) - pd.Timedelta(hours=8)).tz_localize(None).strftime("%Y-%m-%d %H")
    # Header
    print(f"\n{'hour (fixed PST)':<18} {'slots':>6} {'ok':>5} {'median %':>9} {'p99 %':>8} {'max %':>8} {'terms':>6}")
    # Each hour
    for h in sorted(set(hour)):
        # Slots of the hour
        sel = hour == h
        # Solved
        s = sel & ok
        # Formatter
        f = lambda v: "   n/a" if not np.isfinite(v) else f"{v:.3f}"
        # Line
        print(f"{h:<18} {int(sel.sum()):>6} {int(s.sum()):>5} {f(np.median(pct[s]) if s.any() else float('nan')):>9} {f(np.percentile(pct[s], 99) if s.any() else float('nan')):>8} {f(pct[s].max() if s.any() else float('nan')):>8} {f(float(np.median(res['count'][s])) if s.any() else float('nan')):>6}")


def print_terms(table, grand, n_ok, top):
    """Print the ranked per-term lines and the shares by kind."""
    # Header
    print(f"\nterms ranked by share of the window's total residual ({n_ok} solved slots); contribution in percent of nominal per slot; meas and fit are median magnitudes in V or A; diff is the median magnitude of the complex difference, so angle disagreements count")
    # Column names
    print(f"{'rank':>4} {'share':>7} {'avail':>6} {'median %':>9} {'p99 %':>8} {'meas':>9} {'fit':>9} {'diff':>8}  term")
    # Each row
    for i, r in table.head(top).iterrows():
        # Formatter
        f = lambda v, d=3: "    n/a" if not np.isfinite(v) else f"{v:.{d}f}"
        # Line
        print(f"{i + 1:>4} {f(r['share'], 3):>7} {f(r['available'], 2):>6} {f(r['median_pct']):>9} {f(r['p99_pct']):>8} {f(r['meas'], 1):>9} {f(r['fit'], 1):>9} {f(r['diff'], 1):>8}  {r['term']}")
    # Shares by kind
    by_kind = table.groupby("kind")["share"].sum()
    # Line
    print("\nshare of the total residual by kind: " + ", ".join(f"{k} {v:.3f}" for k, v in by_kind.items()) + f"; {len(table)} terms in all")


def restrict(res, keep):
    """A copy of a solve_window result limited to the slots where keep is True, so two solvers are compared on the same sample."""
    # Copy with every per-slot array cut
    return {k: (v[..., keep] if isinstance(v, np.ndarray) and v.shape[-1] == len(keep) else v) for k, v in res.items()}


def compare_solvers(res_a, res_b, name_a, name_b, v_labels, i_labels, i_kind):
    """Compare two solvers on the slots both solved: per-slot per-term totals and whether they rank the same terms highest on that same sample."""
    # Both solved
    both = (res_a["status"] == "ok") & (res_b["status"] == "ok")
    # Nothing shared
    if not both.any():
        # Say and stop
        print(f"\nsolver comparison {name_a} vs {name_b}: no shared solved slots ({name_a} solved {int((res_a['status'] == 'ok').sum())}, {name_b} solved {int((res_b['status'] == 'ok').sum())})")
        # Done
        return
    # Per-term totals in percent
    pa = 100.0 * res_a["total"] / np.where(res_a["count"] > 0, res_a["count"], 1)
    # Second
    pb = 100.0 * res_b["total"] / np.where(res_b["count"] > 0, res_b["count"], 1)
    # Difference on the shared slots
    d = (pb - pa)[both]
    # Correlation needs at least two slots and variation
    corr = float(np.corrcoef(pa[both], pb[both])[0, 1]) if both.sum() > 1 and np.std(pa[both]) > 0 and np.std(pb[both]) > 0 else float("nan")
    # Say
    print(f"\nsolver comparison {name_a} vs {name_b}: both solved on {int(both.sum())} of {len(both)} slots; {name_a} median {np.median(pa[both]):.3f} %, {name_b} median {np.median(pb[both]):.3f} %; {name_b} minus {name_a}: median {np.median(d):+.4f}, 5th to 95th {np.percentile(d, 5):+.4f} to {np.percentile(d, 95):+.4f}, max abs {np.abs(d).max():.4f} points; correlation {corr:.4f}")
    # Rankings on the same jointly solved slots
    ta, _ = term_table(restrict(res_a, both), v_labels, i_labels, i_kind)
    # Second
    tb, _ = term_table(restrict(res_b, both), v_labels, i_labels, i_kind)
    # Top five names
    top_a, top_b = list(ta["term"].head(5)), list(tb["term"].head(5))
    # Overlap
    print(f"  on those slots, top-5 terms shared by both solvers: {len(set(top_a) & set(top_b))} of 5; {name_a} top 3: {top_a[:3]}; {name_b} top 3: {top_b[:3]}")


def main():
    """Build the reference model, solve the window, and print the hourly and per-term tables."""
    # Options
    args = parse_args()
    # Config
    config = sp.load_config(args.config)
    # Weight
    weight = args.current_weight if args.current_weight is not None else float(config.get("current_weight", 0.1))
    # Caltech modules
    ct = sp.setup_caltech(args.caltech_dir, args.topology_dir)
    # Reference model
    model = sp.build_reference_model(ct, args.topology_dir, config)
    # Measurement plan
    plan = sp.measurement_plan(model, config)
    # Window
    label, start_utc, end_utc = window_from_args(args)
    # Say
    print(f"window {label}: {start_utc} to {end_utc} UTC; solver {args.solver}, step {args.step}, current weight {weight}")
    # Data
    table = sp.load_block(args.processed_dir, plan["meters"], start_utc, end_utc)
    # Empty
    if table.empty:
        # Stop
        print("no processed data in this window")
        # Exit
        return
    # Labels
    v_labels, i_labels, i_kind = term_labels(model, plan)
    # Solve
    res = solve_window(model, plan, table, args.solver, weight, args.step, args.freq_tolerance)
    # Status counts
    vals, cnts = np.unique(res["status"], return_counts=True)
    # Say
    print("status: " + ", ".join(f"{v} {c}" for v, c in zip(vals, cnts)))
    # Hourly
    print_hourly(res)
    # Terms
    table_t, grand = term_table(res, v_labels, i_labels, i_kind)
    # Print
    print_terms(table_t, grand, int((res["status"] == "ok").sum()), args.top)
    # Solver comparison
    if args.compare_solvers:
        # The other solver
        other = "cvxpy" if args.solver == "numpy" else "numpy"
        # Solve again
        res_b = solve_window(model, plan, table, other, weight, args.step, args.freq_tolerance)
        # Compare
        compare_solvers(res, res_b, args.solver, other, v_labels, i_labels, i_kind)


# Entry point
if __name__ == "__main__":
    # Run
    main()
