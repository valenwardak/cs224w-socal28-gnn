# =============================================================================
# File:        main_anglecheck.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-08
#
# Purpose:     Compare the measured voltage phasors of the physics model's
#              meters directly, before any fitting, on chosen days. The
#              per-term check (main_physicsterms.py) showed that on the
#              high-residual days the mismatch sits on one meter's voltage
#              terms and that measured and fitted phasors disagree in
#              angle as well as size. This check asks whether unusual
#              relationships already exist between the measured voltages
#              themselves. It reads the same aligned phasors the baseline
#              uses (socal_physics.aligned_phasor: angle shifted to the slot
#              time with the meter's voltage frequency, off-bin readings
#              dropped), takes the channel-to-phase map from the network
#              file through the measurement plan, and reports per day:
#
#                A. Alignment: per meter, readings present and, for the
#                   ones unusable after alignment, why: angle, frequency
#                   or capture offset missing, frequency off the meter's
#                   voltage frequency (the baseline's rule), or magnitude
#                   below a floor (a near-zero phasor has no usable angle;
#                   numpy would give it angle zero). Median capture offset.
#                B. Within-meter phase separations, signed: angle(b) minus
#                   angle(a), angle(c) minus angle(b), angle(a) minus
#                   angle(c), each wrapped to (-180, 180]. For a healthy
#                   positive-sequence meter these sit near -120 degrees. A
#                   swap of two labels shows as +120; a cyclic relabeling
#                   of all three keeps the spacing and is NOT detected here.
#                C. Cross-meter angles: per meter and phase, angle(meter)
#                   minus angle(reference meter, same phase), and the
#                   magnitude ratio. Meters behind transformers are not
#                   expected at zero; the ordinary day is the comparison,
#                   printed side by side with the other days.
#                D. Same-node pairs: meters whose voltages sit on the same
#                   electrical node (egauge_19 and egauge_21), per phase
#                   angle difference and magnitude ratio, expected near
#                   zero and one. This is the strongest direct check.
#                E. With --fitted only: measured against fitted voltage
#                   angle, per slot, for every voltage term, which requires
#                   rerunning the estimator (the per-term check did not
#                   save the complex fitted values). Reported separately.
#
#              All angle summaries are circular: the circular mean
#              direction, the concentration R (1 is all slots agreeing, 0
#              is uniform), the 5th and 95th percentile of the wrapped
#              deviation from the circular mean, and the number of slots
#              compared. Plain medians of angles that wrap at 180 are not
#              used. Readings that are unusable are absent from every
#              summary, never counted as zero. Nothing is written.
#
# Usage (from the repo root):
#   python checks/main_anglecheck.py --days 2025-06-12 2025-04-06 2025-07-29 2025-08-10 2025-08-17
#   python checks/main_anglecheck.py --days 2025-06-12 2025-08-10 --fitted --step 10
#
# Functions:
#   parse_args()                       command-line options with defaults
#   wrap(deg)                          angles wrapped to (-180, 180]
#   circ(deg)                          circular mean, concentration R, 5th and 95th percentile of the deviation, count
#   ctext(stats)                       one text cell from circ()
#   day_bounds(day)                    fixed-PST day to UTC start and end texts
#   meter_phasors(table, plan, freq_tolerance, mag_floor)
#                                      per meter: (3, T) complex phasors in phase order a, b, c, and the reasons readings are unusable
#   section_alignment(ph, counts, T, mag_floor)
#                                      print A
#   section_separations(ph)            print B
#   section_cross(ph, reference)       print C
#   section_same_node(ph, plan)        print D
#   section_fitted(model, plan, table, solver, weight, step, freq_tolerance, mag_floor)
#                                      print E
#   main()                             loop over days
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
    parser = argparse.ArgumentParser(description="Measured voltage-angle relationships between the physics model's meters, before any fitting")
    # Same inputs as main_physics.py
    parser.add_argument("--processed-dir", default=os.path.join("data", "processed"), help="Folder with <YYYY-MM>/<meter>.parquet")
    # Topology
    parser.add_argument("--topology-dir", default="topology", help="Caltech topology folder copied into the repo")
    # Caltech code
    parser.add_argument("--caltech-dir", default="caltech_utils", help="Caltech utils folder copied into the repo")
    # Config
    parser.add_argument("--config", default="physics_config.json", help="Reference model and injection map (names the meters and the component)")
    # Days
    parser.add_argument("--days", nargs="+", required=True, help="Fixed-PST calendar days, like 2025-06-12; the first is the comparison day")
    # Reference meter
    parser.add_argument("--reference", default="egauge_21", help="Meter whose voltage angles the others are compared with")
    # Frequency tolerance
    parser.add_argument("--freq-tolerance", type=float, default=0.01, help="Hertz; as in main_physics.py")
    # Fitted comparison
    parser.add_argument("--fitted", action="store_true", help="Also rerun the estimator and compare measured with fitted voltage angles (section E)")
    # Solver for the fitted part
    parser.add_argument("--solver", default="numpy", choices=["numpy", "cvxpy"], help="Solver for --fitted")
    # Step for the fitted part
    parser.add_argument("--step", type=int, default=1, help="Solve every step-th slot in --fitted")
    # Current weight
    parser.add_argument("--current-weight", type=float, default=None, help="Weight on current residuals for --fitted, default from config")
    # Magnitude floor
    parser.add_argument("--mag-floor", type=float, default=1.0, help="Volts; phasors (measured or fitted) below this have no usable angle and are left out of angle comparisons and ratios")
    # Parse
    args = parser.parse_args()
    # Step check
    if args.step < 1:
        # Refuse
        parser.error("--step must be at least 1")
    # Floor check
    if args.mag_floor <= 0:
        # Refuse
        parser.error("--mag-floor must be positive")
    # Done
    return args


def wrap(deg):
    """Wrap angles in degrees to (-180, 180]."""
    # Shift, modulo, shift back
    return -((-np.asarray(deg, dtype=float) + 180.0) % 360.0) + 180.0


def circ(deg):
    """
    Circular summary of angles in degrees, ignoring NaN: the circular mean
    direction, the concentration R (mean resultant length), the 5th and
    95th percentile of the wrapped deviation from the circular mean, and
    the count. Returns a dict, or None when there is nothing.
    """
    # Finite only
    d = np.asarray(deg, dtype=float)
    # Keep
    d = d[np.isfinite(d)]
    # Nothing
    if len(d) == 0:
        # None
        return None
    # Unit vectors
    z = np.exp(1j * np.deg2rad(d))
    # Mean vector
    m = z.mean()
    # Direction
    mean = float(np.rad2deg(np.angle(m)))
    # Concentration
    R = float(np.abs(m))
    # Wrapped deviations from the mean
    dev = wrap(d - mean)
    # Done
    return {"mean": mean, "R": R, "p5": float(np.percentile(dev, 5)), "p95": float(np.percentile(dev, 95)), "n": int(len(d))}


def ctext(stats):
    """One text cell for a circ() result: mean, R, and the 5th to 95th percentile deviation."""
    # Nothing
    if stats is None:
        # Cell
        return f"{'n/a':>31}"
    # Cell
    return f"{stats['mean']:+7.1f} R{stats['R']:.3f} [{stats['p5']:+5.1f},{stats['p95']:+5.1f}] n{stats['n']:>5}"


def day_bounds(day):
    """Fixed-PST calendar day to UTC start and end texts (08:00 UTC to 08:00 UTC the next day)."""
    # Start
    start = pd.Timestamp(day) + pd.Timedelta(hours=8)
    # End
    end = start + pd.Timedelta(days=1)
    # Texts
    return start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S")


def meter_phasors(table, plan, freq_tolerance, mag_floor):
    """
    Aligned voltage phasors per meter from the measurement plan, in phase
    order a, b, c, using the channel-to-phase map the plan took from the
    network file. Phasors with magnitude below mag_floor are set to NaN,
    since the angle of a near-zero phasor is undefined. Returns (phasors,
    counts, node_of): phasors maps meter to a (3, T) complex array (NaN
    where absent, unusable or below the floor); counts maps meter to the
    reasons over its three channels: present (rms finite), no_angle,
    no_freq, no_toff, off_bin (frequency disagrees with the meter's
    voltage frequency), below_floor, usable, toff_median; node_of maps
    meter to its node index.
    """
    # Slots
    T = len(table)
    # Outputs
    phasors, counts, node_of = {}, {}, {}
    # Each planned voltage term
    for v in plan["voltages"]:
        # Meter
        m = v["meter"]
        # First time: empty array and zero counts
        if m not in phasors:
            # Three phases
            phasors[m] = np.full((3, T), np.nan + 0j)
            # Counts
            counts[m] = {"present": 0, "no_angle": 0, "no_freq": 0, "no_toff": 0, "off_bin": 0, "below_floor": 0, "usable": 0, "toff_median": float("nan")}
            # Node
            node_of[m] = v["node_idx"]
        # Column names, as aligned_phasor builds them
        rms_col, ang_col, freq_col, toff_col = (f"{m}{sp.SEP}{v['channel']}{sp.SEP}rms", f"{m}{sp.SEP}{v['channel']}{sp.SEP}ang0", f"{m}{sp.SEP}{v['channel']}{sp.SEP}freq", f"{m}{sp.SEP}{sp.TOFF}")
        # A column per quantity, NaN when the column is absent
        col = lambda c: table[c].to_numpy(dtype=float) if c in table.columns else np.full(T, np.nan)
        # Values
        rms, ang, freq, toff = col(rms_col), col(ang_col), col(freq_col), col(toff_col)
        # Present readings
        present = np.isfinite(rms)
        # Reasons a present reading cannot be aligned, counted in this order so each reading is counted once
        no_angle = present & ~np.isfinite(ang)
        # Frequency missing
        no_freq = present & np.isfinite(ang) & ~np.isfinite(freq)
        # Offset missing
        no_toff = present & np.isfinite(ang) & np.isfinite(freq) & ~np.isfinite(toff)
        # Aligned phasor, the baseline's own function
        ph = sp.aligned_phasor(table, m, v["channel"], freq_tolerance)
        # Off-bin: alignable but rejected by the frequency rule
        off_bin = present & np.isfinite(ang) & np.isfinite(freq) & np.isfinite(toff) & ~np.isfinite(ph)
        # Below the magnitude floor
        below = np.isfinite(ph) & (np.abs(ph) < mag_floor)
        # Apply the floor
        ph = np.where(below, np.nan + 0j, ph)
        # Store at the phase the network file assigns to the channel
        phasors[m][v["phase_idx"]] = ph
        # Counts
        c = counts[m]
        # Add
        c["present"] += int(present.sum()); c["no_angle"] += int(no_angle.sum()); c["no_freq"] += int(no_freq.sum()); c["no_toff"] += int(no_toff.sum()); c["off_bin"] += int(off_bin.sum()); c["below_floor"] += int(below.sum()); c["usable"] += int(np.isfinite(ph).sum())
        # Capture offset
        if toff_col in table.columns:
            # Median
            c["toff_median"] = float(np.nanmedian(toff))
    # Done
    return phasors, counts, node_of


def section_alignment(ph, counts, T, mag_floor):
    """Print section A: per meter, readings present and why some are unusable after alignment, with the magnitude floor applied."""
    # Title
    print(f"\n  A. alignment ({T} slots x 3 phases per meter; a present reading is unusable when its angle, frequency or capture offset is missing, when its frequency is off the meter's voltage frequency, or when its magnitude is below the floor of {mag_floor} V)")
    # Header
    print(f"  {'meter':<10} {'present':>8} {'no angle':>9} {'no freq':>8} {'no toff':>8} {'off-bin':>8} {'< floor':>8} {'usable':>8} {'toff median s':>14}")
    # Each meter
    for m in sorted(ph, key=lambda s: int(s.split('_')[1])):
        # Counts
        c = counts[m]
        # Line
        print(f"  {m:<10} {c['present']:>8} {c['no_angle']:>9} {c['no_freq']:>8} {c['no_toff']:>8} {c['off_bin']:>8} {c['below_floor']:>8} {c['usable']:>8} {c['toff_median']:>14.4f}")


def section_separations(ph):
    """Print section B: signed within-meter phase separations b-a, c-b, a-c, circular summaries."""
    # Title
    print("\n  B. within-meter phase separations, signed, degrees (healthy positive sequence: near -120 each; a swap of two labels shows as +120; a cyclic relabeling is not detectable here)")
    # Header
    print(f"  {'meter':<10} {'b - a':>31} {'c - b':>31} {'a - c':>31}")
    # Each meter
    for m in sorted(ph, key=lambda s: int(s.split('_')[1])):
        # Angles
        a = np.rad2deg(np.angle(ph[m]))
        # Three separations
        cells = [ctext(circ(wrap(a[1] - a[0]))), ctext(circ(wrap(a[2] - a[1]))), ctext(circ(wrap(a[0] - a[2])))]
        # Line
        print(f"  {m:<10} {cells[0]} {cells[1]} {cells[2]}")


def section_cross(ph, reference):
    """Print section C: per meter and phase, angle relative to the reference meter's same phase, and the magnitude ratio."""
    # Reference present
    if reference not in ph:
        # Say
        print(f"\n  C. cross-meter angles: reference {reference} has no voltage terms in the plan")
        # Done
        return
    # Title
    print(f"\n  C. cross-meter angles relative to {reference}, same phase, degrees, and magnitude ratio (meters behind transformers are not expected at zero; compare days)")
    # Header
    print(f"  {'meter':<10} {'phase a':>31} {'phase b':>31} {'phase c':>31} {'ratio a/b/c':>20}")
    # Reference angles and magnitudes
    ra, rm = np.rad2deg(np.angle(ph[reference])), np.abs(ph[reference])
    # Each meter
    for m in sorted(ph, key=lambda s: int(s.split('_')[1])):
        # Skip the reference
        if m == reference:
            # Next
            continue
        # Angles
        a = np.rad2deg(np.angle(ph[m]))
        # Cells per phase
        cells = [ctext(circ(wrap(a[p] - ra[p]))) for p in range(3)]
        # Ratios
        ratios = "/".join(f"{np.nanmedian(np.abs(ph[m][p]) / rm[p]):.3f}" if np.isfinite(np.abs(ph[m][p]) / rm[p]).any() else "n/a" for p in range(3))
        # Line
        print(f"  {m:<10} {cells[0]} {cells[1]} {cells[2]} {ratios:>20}")


def section_same_node(ph, node_of):
    """Print section D: meters sharing an electrical node, per phase angle difference and magnitude ratio."""
    # Group meters by node
    groups = {}
    # Each meter
    for m, n in node_of.items():
        # Add
        groups.setdefault(n, []).append(m)
    # Pairs
    pairs = [(g[i], g[j]) for g in groups.values() if len(g) > 1 for i in range(len(g)) for j in range(i + 1, len(g))]
    # Title
    print("\n  D. same-node pairs (expected near 0 degrees and ratio 1)")
    # None
    if not pairs:
        # Say
        print("  no two meters share a node in this plan")
        # Done
        return
    # Header
    print(f"  {'pair':<22} {'phase a':>31} {'phase b':>31} {'phase c':>31} {'ratio a/b/c':>20}")
    # Each pair
    for m1, m2 in pairs:
        # Angles
        a1, a2 = np.rad2deg(np.angle(ph[m1])), np.rad2deg(np.angle(ph[m2]))
        # Cells
        cells = [ctext(circ(wrap(a1[p] - a2[p]))) for p in range(3)]
        # Ratios
        ratios = "/".join(f"{np.nanmedian(np.abs(ph[m1][p]) / np.abs(ph[m2][p])):.3f}" if np.isfinite(np.abs(ph[m1][p]) / np.abs(ph[m2][p])).any() else "n/a" for p in range(3))
        # Line
        print(f"  {m1 + ' - ' + m2:<22} {cells[0]} {cells[1]} {cells[2]} {ratios:>20}")


def section_fitted(model, plan, table, solver, weight, step, freq_tolerance, mag_floor):
    """
    Print section E: rerun the estimator on every step-th slot and compare,
    per voltage term, the measured and fitted voltage angle (measured minus
    fitted, wrapped) and the magnitude ratio, circular summaries per meter
    and phase. This section requires the fit; the others do not.
    """
    # Title
    print(f"\n  E. measured minus fitted voltage angle (requires rerunning the estimator; solver {solver}, every {step}th slot), degrees, and measured/fitted magnitude ratio")
    # Admittance
    Y = model["Y"].astype(np.complex128)
    # State size
    n_state = 3 * model["n_nodes"]
    # Arrays
    m = sp.assemble(model, plan, table, freq_tolerance)
    # Positions
    positions = np.arange(len(table))[::step]
    # Solver
    solve = sp.solve_slot_cvxpy if solver == "cvxpy" else sp.solve_slot_numpy
    # Per-term angle differences and ratios
    MV = m["V_meas"].shape[0]
    # Arrays
    dang = np.full((MV, len(positions)), np.nan)
    # Ratios
    ratio = np.full((MV, len(positions)), np.nan)
    # Solved count
    solved = 0
    # Term-slots excluded by the magnitude floor
    excluded = 0
    # Each slot
    for k, pos in enumerate(positions):
        # Availability
        v_ok, i_ok = m["V_ok"][:, pos], m["I_ok"][:, pos]
        # The baseline's skip rules
        if v_ok.sum() == 0:
            # Next
            continue
        # Solve
        V_hat = solve(Y, m["V_meas"][v_ok, pos], m["V_idx"][v_ok], m["V_nom"][v_ok], m["I_meas"][i_ok, pos], m["I_rows"][i_ok], m["I_nom"][i_ok], weight, n_state)
        # Failed
        if V_hat is None:
            # Next
            continue
        # Count
        solved += 1
        # Fitted at the measured rows
        V_fit = V_hat[m["V_idx"]]
        # Rows where both measured and fitted are above the floor; a near-zero phasor has no usable angle
        use = v_ok & (np.abs(m["V_meas"][:, pos]) >= mag_floor) & (np.abs(V_fit) >= mag_floor)
        # Excluded by the floor
        excluded += int((v_ok & ~use).sum())
        # Angle difference where usable
        dang[use, k] = wrap(np.rad2deg(np.angle(m["V_meas"][use, pos])) - np.rad2deg(np.angle(V_fit[use])))
        # Ratio
        ratio[use, k] = np.abs(m["V_meas"][use, pos]) / np.abs(V_fit[use])
    # Say
    print(f"  solved {solved} of {len(positions)} slots; {excluded} measured term-slots left out because the measured or fitted magnitude was below {mag_floor} V")
    # Header
    print(f"  {'meter':<10} {'phase a':>31} {'phase b':>31} {'phase c':>31} {'ratio a/b/c':>20}")
    # Group terms by meter
    by_meter = {}
    # Each term
    for j, v in enumerate(plan["voltages"]):
        # Add
        by_meter.setdefault(v["meter"], {})[v["phase_idx"]] = j
    # Each meter
    for mt in sorted(by_meter, key=lambda s: int(s.split('_')[1])):
        # Cells
        cells = [ctext(circ(dang[by_meter[mt][p]])) if p in by_meter[mt] else f"{'n/a':>31}" for p in range(3)]
        # Ratios
        ratios = "/".join(f"{np.nanmedian(ratio[by_meter[mt][p]]):.3f}" if p in by_meter[mt] and np.isfinite(ratio[by_meter[mt][p]]).any() else "n/a" for p in range(3))
        # Line
        print(f"  {mt:<10} {cells[0]} {cells[1]} {cells[2]} {ratios:>20}")


def main():
    """Build the plan once, then for every day read the phasors and print the sections."""
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
    # Measurement plan, which carries the channel-to-phase map
    plan = sp.measurement_plan(model, config)
    # Say
    print(f"voltage meters in the plan: {sorted({v['meter'] for v in plan['voltages']}, key=lambda s: int(s.split('_')[1]))}; reference {args.reference}; alignment: angle shifted to the slot time with the meter's voltage frequency, readings off the voltage frequency by more than {args.freq_tolerance} Hz dropped")
    # Note on the first day
    print(f"the first day, {args.days[0]}, is the comparison day for the cross-meter angles")
    # Each day
    for day in args.days:
        # Bounds
        start, end = day_bounds(day)
        # Banner
        print(f"\n== {day} (fixed PST): {start} to {end} UTC ==")
        # Data
        table = sp.load_block(args.processed_dir, plan["meters"], start, end)
        # Empty
        if table.empty:
            # Say
            print("  no processed data")
            # Next
            continue
        # Phasors
        ph, counts, node_of = meter_phasors(table, plan, args.freq_tolerance, args.mag_floor)
        # Sections
        section_alignment(ph, counts, len(table), args.mag_floor)
        # B
        section_separations(ph)
        # C
        section_cross(ph, args.reference)
        # D
        section_same_node(ph, node_of)
        # E
        if args.fitted:
            # Fit
            section_fitted(model, plan, table, args.solver, weight, args.step, args.freq_tolerance, args.mag_floor)


# Entry point
if __name__ == "__main__":
    # Run
    main()
