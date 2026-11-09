# =============================================================================
# File:        main_signsearch.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-09
#
# Purpose:     Decide the direction of current through the measured
#              injections of a physics configuration from training data.
#              Each clamp needs a sign: +1 when it reads current flowing
#              into its node, -1 when out of it. The B-side signs were set
#              by analogy with the A side. This script tries every
#              combination of signs on screened training weeks and reports
#              which signs the least-squares fit clearly prefers.
#
#              Rules, as agreed in the design note:
#              - All combinations are solved on identical slots. A sign
#                only negates measured values, so availability, the
#                offset filter, the rank check and the matrix are the
#                same for every combination.
#              - Before comparing, usable counts and current magnitudes
#                are printed per clamp; a clamp whose readings are usable
#                on fewer than the support fraction of solved slots gets
#                "insufficient support" instead of a verdict.
#              - Two acceptance rules, kept separate. Overall: the best
#                combination is the same in every week and beats the
#                runner-up by the margin factor in every week. Per clamp:
#                every near-best combination of every week agrees on its
#                sign.
#              - Repeatability: a second build of the model, arrays and
#                statuses must match the first exactly in which readings
#                are present, which rows exist, which slots are rejected
#                or solvable, or the run stops; then the all-plus
#                combination is solved again and the residual difference
#                is the repeat noise. The overall rule trusts the
#                best-to-runner-up gap, and each clamp its own gap to the
#                best combination with the opposite sign, only when that
#                gap exceeds the repeat noise by the guard factor.
#              - Exclusion: the rank of the plan with each clamp removed,
#                with the unresolved clamps removed together, and on the
#                actual retained slots, is reported so a reduced
#                configuration can be judged before it is adopted.
#              The config file is never modified; the result is printed
#              and saved as JSON.
#
# Usage (from the repo root):
#   python main_signsearch.py --config physics_config_b.json \
#       --weeks 2025-01-13T08:00:00/2025-01-20T08:00:00 2025-01-20T08:00:00/2025-01-27T08:00:00
#   Options: python main_signsearch.py --help
#
# Output:
#   data/processed/physics_b_signs/<label>.json   everything printed, as data
#
# Functions:
#   parse_args()                 command-line options with defaults
#   build_inputs(ct, args, config, start_utc, end_utc)
#                                fresh model, all-plus plan, table, arrays and offset verdicts
#   search_week(ct, args, config, start_utc, end_utc, combos)
#                                statuses, residual medians per combination, support, repeat check
#   print_week(res, combos, plan_elements)
#                                print one week's tables
#   main()                       run the weeks, apply the rules, print and save
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: JSON output
import json
# Standard library: paths
import os
# Third party: numerics
import numpy as np

# The physics baseline's helpers
import socal_physics as sp


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Sign search for the measured current injections of a physics configuration")
    # Processed data
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet")
    # Topology
    parser.add_argument("--topology-dir", default="topology", help="Caltech topology folder copied into the repo")
    # Caltech code
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech utils folder copied into the repo")
    # Config under test
    parser.add_argument("--config", default="physics_config_b.json", help="Configuration whose measured injections get their signs tested")
    # Weeks
    parser.add_argument("--weeks", nargs="+", default=["2025-01-13T08:00:00/2025-01-20T08:00:00", "2025-01-20T08:00:00/2025-01-27T08:00:00"], help="Windows as start/end in UTC, each solved separately")
    # Sampling
    parser.add_argument("--step", type=int, default=6, help="Solve every step-th slot (6 = one per minute)")
    # Baseline settings, same meaning as in main_physics.py
    parser.add_argument("--current-weight", type=float, default=None, help="Weight on current terms; default from the config")
    # Frequency tolerance
    parser.add_argument("--freq-tolerance", type=float, default=0.01, help="Hz; readings off the meter's voltage frequency by more are dropped")
    # Offset filter
    parser.add_argument("--offset-threshold", type=float, default=0.5, help="Seconds; whole slot dropped when any planned meter captured this far or farther from the slot time")
    # Rank tolerance
    parser.add_argument("--rank-rel-tol", type=float, default=1e-9, help="Singular values below this fraction of the largest count as zero")
    # Margin factor
    parser.add_argument("--margin", type=float, default=1.10, help="A combination is near-best when its median is within this factor of the best")
    # Support fraction
    parser.add_argument("--support", type=float, default=0.80, help="Minimum fraction of solved slots on which a clamp's three readings must be usable")
    # Repeatability guard
    parser.add_argument("--repeat-guard", type=float, default=10.0, help="The best-to-runner-up gap must exceed this many times the repeat difference")
    # Output
    parser.add_argument("--out-dir", default=os.path.join("data", "processed", "physics_b_signs"), help="Folder for the JSON result")
    # Label
    parser.add_argument("--label", default="signsearch", help="File name of the JSON result, without extension")
    # Parse
    args = parser.parse_args()
    # Windows must be start/end
    for w in args.weeks:
        # Check
        if w.count("/") != 1:
            # Refuse
            parser.error(f"week {w} must be start/end")
    # Step must be positive
    if args.step < 1:
        # Refuse
        parser.error("--step must be 1 or more")
    # Done
    return args


def build_inputs(ct, args, config, start_utc, end_utc):
    """
    Build everything a week needs from scratch: the reference model, the
    plan with every sign set to +1, the table, the measurement arrays and
    the offset verdicts. Called twice per week so the repeat check sees a
    fresh build, not a copy.
    """
    # Reference model
    model = sp.build_reference_model(ct, args.topology_dir, config)
    # Plan from the config
    plan = sp.measurement_plan(model, config)
    # Same plan with all signs +1; the search applies the signs itself
    plan_plus = sp.plan_with_signs(plan, [1.0] * len(plan["injections"]))
    # Data
    table = sp.load_block(args.processed_dir, plan_plus["meters"], start_utc, end_utc)
    # Arrays
    m = sp.assemble(model, plan_plus, table, args.freq_tolerance)
    # Offset verdicts
    rejected = sp.offset_rejected(table, plan_plus["meters"], args.offset_threshold)
    # Return
    return model, plan, plan_plus, table, m, rejected


def search_week(ct, args, config, start_utc, end_utc, combos):
    """
    Run the search on one week. Returns a dictionary with the sampled
    positions, statuses, per-combination medians and means, per-clamp
    support, the repeat check and the exclusion ranks for single clamps.
    """
    # Weight from the option or the config
    weight = args.current_weight if args.current_weight is not None else float(config.get("current_weight", 0.1))
    # First build
    model, plan, plan_plus, table, m, rejected = build_inputs(ct, args, config, start_utc, end_utc)
    # Nothing to do
    if table.empty:
        # Raise
        raise ValueError(f"no processed data between {start_utc} and {end_utc}")
    # Sampled positions
    positions = np.arange(0, len(table), args.step)
    # Statuses, sign-independent
    statuses = sp.slot_statuses(model, plan_plus, m, rejected, positions, weight, args.rank_rel_tol)
    # Solvable mask
    ok = np.array([s == "ok" for s in statuses])
    # Row multipliers per combination: injection rows get the sign, zero rows stay +1
    rows = sp.injection_rows(plan_plus)
    # (MI, C)
    row_signs = np.ones((m["I_meas"].shape[0], len(combos)))
    # Fill
    for c, signs in enumerate(combos):
        # Each injection
        for k, s in enumerate(signs):
            # Its three rows
            row_signs[rows[k], c] = s
    # Residuals for every combination
    res = sp.solve_all_signs(model, m, positions, ok, row_signs, weight)
    # Medians and means over solved slots
    medians = np.nanmedian(res, axis=1) if ok.any() else np.full(len(combos), np.nan)
    # Means
    means = np.nanmean(res, axis=1) if ok.any() else np.full(len(combos), np.nan)
    # Per-clamp support and magnitudes
    support = sp.clamp_support(m, plan_plus, positions, ok)
    # Repeat: fresh build, all-plus only
    model2, _, plan_plus2, table2, m2, rejected2 = build_inputs(ct, args, config, start_utc, end_utc)
    # Same slots required
    if len(table2) != len(table) or not table2.index.equals(table.index):
        # Raise
        raise ValueError("the second build read different slots; cannot compare")
    # Same rows in the same order: state rows and nominals of every measurement
    for key in ("V_idx", "V_nom", "I_rows", "I_nom"):
        # Compare
        if m2[key].shape != m[key].shape or not np.array_equal(m2[key], m[key]):
            # Raise
            raise ValueError(f"the second build has different measurement rows ({key}); cannot compare")
    # Same meters in the plan
    if plan_plus2["meters"] != plan_plus["meters"]:
        # Raise
        raise ValueError("the second build has a different meter list; cannot compare")
    # Availability must match reading by reading
    d_vok = int((m2["V_ok"][:, positions] != m["V_ok"][:, positions]).sum())
    # Currents
    d_iok = int((m2["I_ok"][:, positions] != m["I_ok"][:, positions]).sum())
    # Offset verdicts
    d_rej = int((rejected2[positions] != rejected[positions]).sum())
    # Statuses recomputed from the second build
    statuses2 = sp.slot_statuses(model2, plan_plus2, m2, rejected2, positions, weight, args.rank_rel_tol)
    # Differences
    d_status = int(sum(a != b for a, b in zip(statuses, statuses2)))
    # Any change in what is present, rejected or solvable stops the run
    if d_vok or d_iok or d_rej or d_status:
        # Raise
        raise ValueError(f"the second build differs in availability ({d_vok} voltage, {d_iok} current readings), offset verdicts ({d_rej}) or statuses ({d_status}); cannot issue verdicts")
    # Value differences where both builds have a reading
    d_v = float(np.max(np.abs(m2["V_meas"][:, positions][m["V_ok"][:, positions]] - m["V_meas"][:, positions][m["V_ok"][:, positions]]))) if m["V_ok"][:, positions].any() else 0.0
    # Currents
    d_i = float(np.max(np.abs(m2["I_meas"][:, positions][m["I_ok"][:, positions]] - m["I_meas"][:, positions][m["I_ok"][:, positions]]))) if m["I_ok"][:, positions].any() else 0.0
    # Matrix
    d_y = float(np.max(np.abs(model2["Y"] - model["Y"])))
    # All-plus residuals again on the same slots
    res2 = sp.solve_all_signs(model2, m2, positions, ok, np.ones((m2["I_meas"].shape[0], 1)), weight)[0]
    # Per-slot and median differences
    d_slot = float(np.nanmax(np.abs(res2 - res[0]))) if ok.any() else 0.0
    # Median difference
    d_med = float(abs(np.nanmedian(res2) - medians[0])) if ok.any() else 0.0
    # Rank with each clamp removed, on the retained slots
    excl = [sp.rank_without(model, m, positions, ok, rows[k], weight, args.rank_rel_tol) for k in range(len(rows))]
    # Status counts
    counts = {s: int(statuses.count(s)) for s in sorted(set(statuses))}
    # Return
    return {"start_utc": start_utc, "end_utc": end_utc, "step": int(args.step), "first_slot_utc": str(table.index[0]), "slots_in_window": int(len(table)),
            "sampled": int(len(positions)), "solved": int(ok.sum()), "status_counts": counts,
            "medians": [float(x) for x in medians], "means": [float(x) for x in means], "support": support,
            "repeat": {"availability_changes_V": d_vok, "availability_changes_I": d_iok, "offset_verdict_changes": d_rej, "status_changes": d_status, "max_abs_diff_V": d_v, "max_abs_diff_I": d_i, "max_abs_diff_Y": d_y, "max_abs_diff_residual_per_slot": d_slot, "abs_diff_median": d_med},
            "separation": sp.clamp_separation(combos, medians),
            "rank_without_each": [{"determined": d, "solvable": s} for d, s in excl],
            "_model": model, "_m": m, "_positions": positions, "_ok": ok, "_rows": rows, "_weight": weight, "_elements": [inj["terms"][0]["element"] for inj in plan_plus["injections"]]}


def print_week(res, combos, elements):
    """Print one week's support table, repeat check and ranked combinations."""
    # Header
    print(f"\n== week {res['start_utc']} to {res['end_utc']} UTC: {res['slots_in_window']} slots, every {res['step']}th sampled from {res['first_slot_utc']}, {res['sampled']} sampled, {res['solved']} solved; statuses {res['status_counts']} ==")
    # Support
    print("\n  clamp support on solved sampled slots (usable = present, on the voltage frequency, aligned) and measured current in amps")
    # Column header
    print(f"  {'element':<10} {'meter':<10} {'all 3 usable':>13} {'fraction':>9} {'per phase a/b/c':>18} {'A p10':>8} {'A median':>9} {'A p90':>8} {'rating A':>9} {'median/rating':>14}")
    # Rows
    for s in res["support"]:
        # Per phase
        pp = "/".join(str(s["usable_per_phase"][p]) for p in sp.PHASES)
        # Line
        print(f"  {s['element']:<10} {s['meter']:<10} {s['usable_all_phases']:>13} {s['support_fraction']:>9.3f} {pp:>18} {s['current_a_p10']:>8.2f} {s['current_a_median']:>9.2f} {s['current_a_p90']:>8.2f} {s['rating_a']:>9.1f} {s['median_over_rating']:>14.4f}")
    # Repeat
    r = res["repeat"]
    # Say
    print(f"\n  repeat check (fresh model, arrays and statuses, same slots): availability changes V {r['availability_changes_V']} I {r['availability_changes_I']}, offset verdict changes {r['offset_verdict_changes']}, status changes {r['status_changes']}; max |dV| {r['max_abs_diff_V']:.3e}, max |dI| {r['max_abs_diff_I']:.3e}, max |dY| {r['max_abs_diff_Y']:.3e}, max per-slot residual difference {r['max_abs_diff_residual_per_slot']:.3e}, median difference {r['abs_diff_median']:.3e}")
    # Ranking
    order = np.argsort(res["medians"])
    # Header
    print(f"\n  combinations by median per-term residual (percent); signs in the order {elements}")
    # Rows
    for rank, c in enumerate(order, 1):
        # Signs as text
        txt = " ".join("+" if s > 0 else "-" for s in combos[c])
        # Line
        print(f"  {rank:>3}  {txt}  median {100 * res['medians'][c]:9.4f}  mean {100 * res['means'][c]:9.4f}")
    # Per-clamp separation
    print("\n  per clamp: best median with +1, best with -1 (percent), preferred sign, gap")
    # Rows
    for e, sepr in zip(elements, res["separation"]):
        # Line
        print(f"  {e:<10} +1: {100 * sepr['best_plus']:9.4f}  -1: {100 * sepr['best_minus']:9.4f}  preferred {sepr['preferred']:+.0f}  gap {100 * sepr['gap']:.4e}")
    # Exclusion ranks
    print("\n  determined solved slots with one clamp removed from the plan")
    # Rows
    for e, rk in zip(elements, res["rank_without_each"]):
        # Line
        print(f"  without {e:<10} {rk['determined']} of {rk['solvable']}")


def main():
    """Run the search on every week, apply the two rules, print and save."""
    # Options
    args = parse_args()
    # Config
    config = sp.load_config(args.config)
    # Caltech modules
    ct = sp.setup_caltech(args.caltech_dir, args.topology_dir)
    # Plan once to know the injections
    model0 = sp.build_reference_model(ct, args.topology_dir, config)
    # Plan
    plan0 = sp.measurement_plan(model0, config)
    # Elements in plan order
    elements = [inj["terms"][0]["element"] if len(inj["terms"]) == 1 else None for inj in plan0["injections"]]
    # One clamp per node
    if any(e is None for e in elements):
        # Refuse
        raise ValueError("every measured injection must have exactly one term for the sign search")
    # Current signs in the config, for the record
    current = [inj["terms"][0]["sign"] for inj in plan0["injections"]]
    # Combinations
    combos = sp.sign_combinations(len(elements))
    # Say
    print(f"config {args.config}: {len(elements)} measured injections {elements}; current signs {current}; {len(combos)} combinations; margin {args.margin}, support {args.support}, repeat guard {args.repeat_guard}; offset threshold {args.offset_threshold} s whole slot, rank check on")
    # Each week
    weeks = []
    # Loop
    for w in args.weeks:
        # Bounds
        start_utc, end_utc = w.split("/")
        # Search
        res = search_week(ct, args, config, start_utc, end_utc, combos)
        # Print
        print_week(res, combos, elements)
        # Keep
        weeks.append(res)
    # Rule checks
    print("\n== rules ==")
    # Repeat guard per week: the gap must exceed guard times the repeat median difference
    guard_ok = []
    # Near-best lists
    near_lists = []
    # Overall winners
    winners = []
    # Each week
    for res in weeks:
        # Medians
        med = np.asarray(res["medians"])
        # Order
        order = np.argsort(med)
        # Best and runner-up
        best, second = med[order[0]], med[order[1]]
        # Gap
        gap = second - best
        # Repeat difference
        rep = res["repeat"]["abs_diff_median"]
        # Guard
        g = gap > args.repeat_guard * rep
        # Keep
        guard_ok.append(bool(g))
        # Near-best
        near_lists.append(sp.near_best(med, args.margin))
        # Winner
        winners.append(int(order[0]))
        # Say
        print(f"  week {res['start_utc']}: best {' '.join('+' if s > 0 else '-' for s in combos[order[0]])} median {100 * best:.4f}, runner-up {' '.join('+' if s > 0 else '-' for s in combos[order[1]])} median {100 * second:.4f}, ratio {second / best if best else float('nan'):.4f}, gap {100 * gap:.4e} against repeat difference {100 * rep:.4e} (guard x{args.repeat_guard}: {'passes' if g else 'fails'}), near-best count {len(near_lists[-1])}")
    # Overall rule
    same_winner = len(set(winners)) == 1
    # Separated in every week
    separated = all(np.asarray(res["medians"])[np.argsort(res["medians"])[1]] >= args.margin * np.asarray(res["medians"])[np.argsort(res["medians"])[0]] for res in weeks)
    # Verdict
    overall = "winner separated" if (same_winner and separated and all(guard_ok)) else "no separated winner"
    # Say
    print(f"  overall rule: same best combination in every week: {same_winner}; runner-up at least x{args.margin} in every week: {separated}; repeat guard in every week: {all(guard_ok)} -> {overall}")
    # Support per injection in every week
    support_ok = [all(res["support"][k]["support_fraction"] >= args.support for res in weeks) for k in range(len(elements))]
    # Each clamp's own separation against the repeat noise, in every week
    separation_ok = [all(res["separation"][k]["gap"] > args.repeat_guard * res["repeat"]["abs_diff_median"] for res in weeks) for k in range(len(elements))]
    # Say
    for k, e in enumerate(elements):
        # Per week text
        txt = "; ".join(f"{res['start_utc'][:10]} gap {100 * res['separation'][k]['gap']:.4e} vs repeat {100 * res['repeat']['abs_diff_median']:.4e} preferred {res['separation'][k]['preferred']:+.0f}" for res in weeks)
        # Line
        print(f"  clamp {e:<10} separation guard {'passes' if separation_ok[k] else 'fails'}: {txt}")
    # Per-clamp verdicts
    verdicts = sp.sign_verdicts(combos, near_lists, support_ok, separation_ok)
    # Print
    print("\n  per-clamp verdicts (supported = every near-best combination of every week agrees)")
    # Rows
    for e, cur, v in zip(elements, current, verdicts):
        # Line
        print(f"  {e:<10} config sign {cur:+.0f}  verdict {v['verdict']}" + (f"  sign {v['sign']:+.0f}" + ("  (same as config)" if v["sign"] == cur else "  (differs from config)") if v["sign"] is not None else ""))
    # Unresolved set
    unresolved = [k for k, v in enumerate(verdicts) if v["verdict"] != "supported"]
    # Combined exclusion check on the retained slots, per week
    combined = []
    # Only when something is unresolved
    if unresolved:
        # Rows to drop
        drop = [r for k in unresolved for r in weeks[0]["_rows"][k]]
        # Each week
        for res in weeks:
            # Rank on retained slots
            d, s = sp.rank_without(res["_model"], res["_m"], res["_positions"], res["_ok"], drop, res["_weight"], args.rank_rel_tol)
            # Keep
            combined.append({"week": res["start_utc"], "determined": d, "solvable": s})
            # Say
            print(f"  with all unresolved clamps removed ({[elements[k] for k in unresolved]}), week {res['start_utc']}: {d} of {s} solved slots stay determined")
    # Nothing unresolved
    else:
        # Say
        print("  every clamp supported; no exclusion needed")
    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    # Path
    path = os.path.join(args.out_dir, f"{args.label}.json")
    # Data without the private arrays
    record = {"config": args.config, "elements": elements, "config_signs": current, "combinations": [list(c) for c in combos],
              "settings": {"step": args.step, "margin": args.margin, "support": args.support, "repeat_guard": args.repeat_guard, "offset_threshold": args.offset_threshold, "freq_tolerance": args.freq_tolerance, "rank_rel_tol": args.rank_rel_tol, "current_weight": weeks[0]["_weight"]},
              "weeks": [{k: v for k, v in res.items() if not k.startswith("_")} for res in weeks],
              "near_best": near_lists, "winners": winners, "repeat_guard_ok": guard_ok, "overall": overall, "support_ok": support_ok, "separation_ok": separation_ok, "verdicts": verdicts, "combined_exclusion": combined}
    # Write
    with open(path, "w") as f:
        # Dump
        json.dump(record, f, indent=1)
    # Say
    print(f"\nsaved {path}; the config file was not modified")


# Entry point
if __name__ == "__main__":
    # Run
    main()
