# =============================================================================
# File:        main_summary.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-08
#
# Purpose:     Print a plain-language summary of one or more scored runs,
#              read from the score files main_score.py already wrote
#              (score.json, score_grid.json, score_era.json,
#              score_grid_era.json, whichever exist). Nothing is rescored
#              and no threshold changes: this is a reader. For each run it
#              prints what the run was (architecture, edges, graph file,
#              horizon, seed), then for every score file (scoring version
#              and statistic named) and every mode: the episodes the model
#              detected by id with their delays in seconds from the
#              episode's first logged change (negative delays kept as
#              written, which means the alarm started inside the one-minute
#              allowance before the first change), the episodes persistence
#              detected, the quiet block's false alarms per day and share of
#              slots in alarm, the event blocks' false alarms per day and
#              share of slots in alarm, the pooled false alarms per day
#              labeled as pooled, and the longest merged alarm span labeled
#              as a span, separately from the share of slots in alarm.
#              Then the episodes in scope that this run's model did not
#              detect in any of the selected files and modes, and any
#              episode out of scope. With several runs, a comparison table
#              follows, preceded by the scoring settings each file used
#              (alpha, minimum meters, merge gap, event window, common
#              mask, restriction, calibration mapping) with differences
#              flagged, so rows are not read as if scored under identical
#              rules when they were not; the same settings are printed under
#              each file in the single-run block (calibration label and era
#              rules separately, and the grid statistic's design label,
#              reference percentile, minimum count, and floors):
#              one row per run and score file, with the detected episode
#              ids and the quiet-block false alarms per day and share of
#              slots in alarm for each mode.
#              With --plot, three figures are written to --plot-dir from the
#              same score files (matplotlib, nothing rescored):
#                detections.png   one panel per score file and mode; two
#                                 rows per run (its model, its own
#                                 persistence), columns the scored episodes;
#                                 a filled cell is a detection, shaded by
#                                 delay (darker is later), with the delay in
#                                 seconds (a negative delay is a match
#                                 inside the pre-event allowance); a light
#                                 empty cell is a miss, a hatched cell is an
#                                 episode out of scope, a grey cell means
#                                 that result is not available
#                quiet_block.png  per score file and mode: bars of the
#                                 quiet-block false alarms per day and, in a
#                                 second row, the share of quiet-block slots
#                                 in alarm, model beside persistence, one
#                                 bar group per run; version 1 files only,
#                                 since version 2 reuses the version 1
#                                 quiet block
#                delays.png       per mode: for each episode any model
#                                 detected, one point per run and score file
#                                 at its delay in seconds, so the spread
#                                 across runs is visible; negative delays
#                                 shown as they are
#
#              Single file on purpose: it is a check, not pipeline code.
#              Lives in checks/ and is run from the repo root.
#
# Usage:
#   python checks/main_summary.py --run-dir data/processed/models/gcn_dyn_aw001
#   python checks/main_summary.py --run-dir RUN1 RUN2 RUN3
#   python checks/main_summary.py --run-dir RUN --files score.json score_era.json --modes mag ang
#   python checks/main_summary.py --run-dir RUN1 RUN2 --plot --plot-dir figures
#
# Functions:
#   parse_args()              command-line options with defaults
#   run_options(run_dir)      the training options from any residual summary file in the run
#   describe(options, name)   one line saying what the run was
#   detected(r)               detected episodes of one result as (id, delay) pairs, by id
#   scope(r)                  episode ids in scope of one result
#   settings_line(st)         the scoring settings of one file that matter for comparing rows
#   ids_text(pairs)           "3 (650 s), 8 (520 s)" or "none"
#   pct(x)                    a fraction as a percent string, or "n/a"
#   rate(x)                   a per-day rate to one decimal, or "n/a"
#   split_of(r, name)         the per-split entry whose key starts with name
#   split_line(r, name)       false alarms per day and share of slots in alarm for one split
#   summarize_run(run_dir, files, modes)
#                             print one run's summary, return rows for the table and the plots
#   plot_detections(rows, modes, out)
#                             the detection grid figure
#   plot_quiet(rows, modes, out)
#                             the quiet-block false alarm figure
#   plot_delays(rows, modes, out)
#                             the delay spread figure
#   main()                    run over every run folder, print the table, write the plots
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: file names
import glob
# Standard library: JSON
import json
# Standard library: paths
import os

# The score files main_score.py writes, in reading order
DEFAULT_FILES = ["score.json", "score_grid.json", "score_era.json", "score_grid_era.json"]


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Plain-language summary of scored runs, read from their score files")
    # Runs
    parser.add_argument("--run-dir", nargs="+", required=True, help="One or more run folders")
    # Files
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="Score files to read in each run folder; missing ones are skipped")
    # Modes
    parser.add_argument("--modes", nargs="+", default=["ang", "mag"], help="Modes to report, in this order")
    # Plots
    parser.add_argument("--plot", action="store_true", help="Also write the three figures")
    # Where
    parser.add_argument("--plot-dir", default="figures", help="Folder for the figures")
    # Parse
    return parser.parse_args()


def run_options(run_dir):
    """The training options recorded in any residual summary file of the run, or an empty dict."""
    # Candidates: the per-block summary files
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        # Skip score files
        if os.path.basename(path).startswith("score"):
            # Next
            continue
        # Read
        try:
            # Parse
            with open(path) as f:
                # Body
                body = json.load(f)
        except (OSError, ValueError):
            # Next
            continue
        # Options present
        if isinstance(body, dict) and isinstance(body.get("options"), dict):
            # Found
            return body["options"]
    # None
    return {}


def describe(options, name):
    """One line saying what the run was, from its options."""
    # Pieces, each only when known
    kind = {"gcn": "GCN", "gat": "GAT", "sage": "GraphSAGE", "gru": "GRU"}.get(options.get("kind"), options.get("kind", "unknown model"))
    # Edges
    edges = options.get("edges")
    # Graph file name, without the hash
    graph = (options.get("graph_id") or "").split(":")[0]
    # Selection rule
    selection = options.get("graph_selection", "fixed" if graph else None)
    # Horizon and seed
    horizon, seed = options.get("horizon"), options.get("seed")
    # Build
    parts = [kind]
    # Edges
    if edges:
        # Words
        parts.append({"true": "true edges", "random": "random edges", "none": "no edges"}.get(edges, edges))
    # Graph
    if graph:
        # Words
        parts.append(f"{graph} ({'time-indexed' if selection == 'last_history_slot' else 'fixed'})")
    # Horizon
    if horizon is not None:
        # Words
        parts.append(f"horizon {horizon} slot{'s' if horizon != 1 else ''} ({10 * horizon} s ahead)")
    # Seed
    if seed is not None:
        # Words
        parts.append(f"seed {seed}")
    # Masked
    if options.get("masked"):
        # Words
        parts.append("masked objective")
    # Line
    return f"Run {name}: " + ", ".join(parts)


def detected(r):
    """Detected episodes of one result, as a list of (id, delay seconds) sorted by id."""
    # Collect
    return sorted((e["episode"], e["delay_s"]) for e in r["episode_results"] if e["detected"])


def scope(r):
    """Episode ids in scope of one result."""
    # Collect
    return sorted(e["episode"] for e in r["episode_results"])


def settings_line(st):
    """The scoring settings of one file that matter when comparing rows, as a dict."""
    # Pick: detection rules, calibration, and the grid statistic's references
    return {"alpha": st.get("alpha"), "min_meters": st.get("min_meters"), "merge_slots": st.get("merge_slots"),
            "before_minutes": st.get("before_minutes"), "after_minutes": st.get("after_minutes"),
            "common": st.get("common"), "restrict_common": st.get("restrict_common"),
            "calib_label": st.get("calib_label"), "era_calib": st.get("era_calib"),
            "design_label": st.get("design_label"), "ref_percentile": st.get("ref_percentile"), "ref_min_count": st.get("ref_min_count"),
            "ref_floor_mag": st.get("ref_floor_mag"), "ref_floor_ang": st.get("ref_floor_ang")}


def ids_text(pairs):
    """Episode ids with delays, or none."""
    # Empty
    if not pairs:
        # Say
        return "none"
    # Join; a negative delay is an alarm that started inside the pre-event allowance
    return ", ".join(f"{i} ({d:+.0f} s)" if d < 0 else f"{i} ({d:.0f} s)" for i, d in pairs)


def pct(x):
    """A fraction as a percent with two decimals, or n/a."""
    # Missing
    return "n/a" if x is None else f"{100.0 * x:.2f}%"


def rate(x):
    """A per-day rate with two decimals, or n/a."""
    # Missing
    return "n/a" if x is None else f"{x:.2f}"


def split_of(r, name):
    """The per-split entry whose key starts with name (the score file keys splits by their table prefix, e.g. test_normal_), or None."""
    # Scan
    for key, val in r.get("per_split", {}).items():
        # Match by prefix
        if key.startswith(name):
            # Found
            return val
    # None
    return None


def split_line(r, name):
    """False alarms per day and share of slots in alarm for one split of a result, with the longest merged span."""
    # The split
    s = split_of(r, name)
    # Missing
    if s is None:
        # Say
        return "not scored"
    # Span
    span = s.get("merged_span_max_s")
    # Text
    return f"{rate(s.get('false_alarms_per_day'))} per day, {pct(s.get('alarm_slot_fraction'))} of slots in alarm" + (f", longest merged alarm span {span / 3600.0:.1f} h" if span else "")


def summarize_run(run_dir, files, modes):
    """Print one run's summary and return rows for the comparison table."""
    # Name
    name = os.path.basename(run_dir.rstrip("/"))
    # Options
    options = run_options(run_dir)
    # Header
    print("\n" + describe(options, name))
    # Rows for the table
    rows = []
    # Episodes ever detected by the model in this run, and every episode in scope anywhere
    ever, in_scope, out_scope = set(), set(), set()
    # Loop over score files
    for fname in files:
        # Path
        path = os.path.join(run_dir, fname)
        # Missing
        if not os.path.exists(path):
            # Say once
            print(f"  {fname}: not present")
            # Next
            continue
        # Read
        with open(path) as f:
            # Body
            body = json.load(f)
        # Settings
        st = body["settings"]
        # Label
        label = f"scoring version {st.get('scoring_version', 1)}, {st.get('statistic', 'channel')} statistic"
        # Print, with the version 2 caveat
        print(f"  {label} ({fname})" + (" (quiet block figures are the version 1 result reused, as the scoring notes)" if st.get("scoring_version", 1) == 2 else ""))
        # The settings this file was scored with
        print("    settings: " + ", ".join(f"{k}={v}" for k, v in settings_line(st).items() if v is not None))
        # Row, with the settings that matter for comparing rows
        row = {"run": name, "file": label, "modes": {}, "settings": settings_line(st)}
        # Loop over modes
        for mode in modes:
            # Present
            if mode not in body.get("modes", {}):
                # Say
                print(f"    {mode}: not scored")
                # Next
                continue
            # Results
            model, pers = body["modes"][mode]["model"], body["modes"][mode]["persistence"]
            # Detected
            md, pd_ = detected(model), detected(pers)
            # Remember
            ever.update(i for i, _ in md)
            # Scope of this result
            in_scope.update(scope(model))
            # Episodes the result lists as out of scope, when the file records them
            out_scope.update(e["episode"] for e in model.get("episodes_partial_excluded", []))
            # Mode name
            word = {"mag": "magnitude", "ang": "angle", "both": "either"}.get(mode, mode)
            # Lines
            print(f"    {word}: model detected {len(md)} of {len(scope(model))} episodes in scope: {ids_text(md)}")
            # Persistence
            print(f"      persistence detected {len(pd_)}: {ids_text(pd_)}")
            # Quiet block
            pq = split_of(pers, 'test_normal') or {}
            # Lines
            print(f"      quiet block, model: {split_line(model, 'test_normal')}; persistence {rate(pq.get('false_alarms_per_day'))} per day, {pct(pq.get('alarm_slot_fraction'))} of slots in alarm")
            # Event blocks
            print(f"      event blocks, model: {split_line(model, 'test_events')}")
            # Pooled
            print(f"      pooled over all test blocks, model: {rate(model.get('false_alarms_per_day'))} false alarms per day")
            # Table cell
            quiet = split_of(model, "test_normal") or {}
            # Persistence quiet block
            pquiet = split_of(pers, "test_normal") or {}
            # Store, with what the plots need
            row["modes"][mode] = {"ids": [i for i, _ in md], "fa": quiet.get("false_alarms_per_day"), "frac": quiet.get("alarm_slot_fraction"),
                                  "model": md, "pers": pd_, "pers_fa": pquiet.get("false_alarms_per_day"), "pers_frac": pquiet.get("alarm_slot_fraction"),
                                  "scope": scope(model), "pers_scope": scope(pers), "partial": [e["episode"] for e in model.get("episodes_partial_excluded", [])]}
        # Keep
        rows.append(row)
    # Not detected in the selected files and modes, among the episodes in scope
    never = sorted(e for e in in_scope if e not in ever)
    # Say
    if in_scope:
        # Line
        print(f"  in scope but not detected by this run's model in the selected files and modes ({', '.join(files)}; {', '.join(modes)}): {never if never else 'none'}")
    else:
        # Nothing read
        print("  no score results available for this run in the selected files and modes")
    # Out of scope
    if out_scope:
        # Line
        print(f"  episodes out of scope (only partly covered by the scored blocks): {sorted(out_scope)}")
    # Done
    return rows


def plot_detections(rows, modes, out):
    """The detection grid: one panel per score file and mode; two rows per run (model, its own persistence), columns the episodes."""
    # Lazy import so the text summary needs no matplotlib
    import matplotlib
    # No display
    matplotlib.use("Agg")
    # Plotting
    import matplotlib.pyplot as plt
    # Score file labels in order of appearance
    files = list(dict.fromkeys(r["file"] for r in rows))
    # Runs in order
    runs = list(dict.fromkeys(r["run"] for r in rows))
    # Episode columns: every id in scope or partial anywhere, sorted
    eps = sorted({e for r in rows for m in r["modes"].values() for e in m["scope"] + m["partial"] + m["pers_scope"]})
    # Nothing to draw
    if not eps:
        # Skip
        return
    # Row labels: model and persistence per run
    labels = [f"{run} model" for run in runs for _ in (0,)]
    # Interleave
    labels = [l for run in runs for l in (f"{run}: model", f"{run}: persistence")]
    # Figure grid: files down, modes across
    fig, axes = plt.subplots(len(files), len(modes), figsize=(0.55 * len(eps) + 3.5, 0.3 * len(labels) * len(files) + 1.2 * len(files)), squeeze=False)
    # Loop
    for i, fl in enumerate(files):
        # Rows of this file
        sub = {r["run"]: r for r in rows if r["file"] == fl}
        # Loop over modes
        for j, mode in enumerate(modes):
            # Axis
            ax = axes[i][j]
            # Draw cells
            for y, label in enumerate(labels):
                # Which run and source
                run, source = label.split(": ")
                # Result available
                cell = sub[run]["modes"].get(mode) if run in sub else None
                # Loop over episodes
                for x, ep in enumerate(eps):
                    # Not available
                    if cell is None:
                        # Grey
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, color="0.8", ec="white"))
                        # Next
                        continue
                    # Pairs and scope for this source
                    pairs = dict(cell["model"] if source == "model" else cell["pers"])
                    # Scope
                    sc = cell["scope"] if source == "model" else cell["pers_scope"]
                    # Out of scope
                    if ep not in sc:
                        # Hatched
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, hatch="///", ec="0.7"))
                    elif ep in pairs:
                        # Shade by delay, 0 s light to 1800 s dark
                        shade = 0.85 - 0.6 * min(max(pairs[ep], 0.0), 1800.0) / 1800.0
                        # Color: blue family for the model, grey-blue for persistence
                        color = (shade * 0.3, shade * 0.6, shade) if source == "model" else (shade * 0.5, shade * 0.55, shade * 0.65)
                        # Cell
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, ec="white"))
                        # Delay text
                        ax.text(x + 0.5, y + 0.5, f"{pairs[ep]:+.0f}" if pairs[ep] < 0 else f"{pairs[ep]:.0f}", ha="center", va="center", fontsize=6, color="white")
                    else:
                        # Missed: empty light cell
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, ec="0.85"))
            # Axes
            ax.set_xlim(0, len(eps)); ax.set_ylim(len(labels), 0)
            # Ticks
            ax.set_xticks([x + 0.5 for x in range(len(eps))]); ax.set_xticklabels([str(e) for e in eps], fontsize=7)
            # Rows
            ax.set_yticks([y + 0.5 for y in range(len(labels))]); ax.set_yticklabels(labels, fontsize=6)
            # Title
            ax.set_title(f"{fl}, { {'mag': 'magnitude', 'ang': 'angle'}.get(mode, mode)}", fontsize=7)
            # Episode axis label
            ax.set_xlabel("episode", fontsize=7)
    # One caption for the whole figure
    fig.text(0.5, 0.005, "filled = detected, number = delay in s from the episode's first logged change (negative = alarm inside the pre-event allowance); empty = missed; hatched = out of scope; grey = result not available", ha="center", fontsize=6)
    # Layout, leaving room for the caption
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    # Save
    fig.savefig(out, dpi=150)
    # Close
    plt.close(fig)


def plot_quiet(rows, modes, out):
    """Quiet-block false alarms per day and share of slots in alarm, model beside persistence, per score file and mode."""
    # Lazy import
    import matplotlib
    # No display
    matplotlib.use("Agg")
    # Plotting
    import matplotlib.pyplot as plt
    # Numerics
    import numpy as np
    # Version 1 files only: version 2 reuses the version 1 quiet block, so its panels would repeat
    files = [f for f in dict.fromkeys(r["file"] for r in rows) if "version 1" in f] or list(dict.fromkeys(r["file"] for r in rows))
    # Runs
    runs = list(dict.fromkeys(r["run"] for r in rows))
    # Two rows of panels (rate, share) per file, modes across
    fig, axes = plt.subplots(2 * len(files), len(modes), figsize=(3.5 * len(modes) + 2, 2.4 * 2 * len(files) + 1), squeeze=False)
    # Positions
    x = np.arange(len(runs))
    # Loop
    for i, fl in enumerate(files):
        # Rows of this file
        sub = {r["run"]: r for r in rows if r["file"] == fl}
        # Loop over modes
        for j, mode in enumerate(modes):
            # Values, zero kept as zero, only a missing result becomes nan
            def pick(key):
                # Per run
                vals = []
                # Loop
                for r in runs:
                    # Value or None
                    v = sub[r]["modes"][mode].get(key) if r in sub and mode in sub[r]["modes"] else None
                    # Keep
                    vals.append(float("nan") if v is None else v)
                # Done
                return vals
            # Mark missing results with a cross at zero
            def mark(ax, vals, xs):
                # Loop
                for v, xx in zip(vals, xs):
                    # Missing
                    if v != v:
                        # Cross
                        ax.plot(xx, 0, "x", color="red", ms=6)
            # Rate panel
            ax = axes[2 * i][j]
            # Bars
            ax.bar(x - 0.2, pick("fa"), 0.4, label="model", color=(0.2, 0.45, 0.75))
            # Persistence
            ax.bar(x + 0.2, pick("pers_fa"), 0.4, label="persistence", color="0.6")
            # Missing marks
            mark(ax, pick("fa"), x - 0.2); mark(ax, pick("pers_fa"), x + 0.2)
            # Labels
            ax.set_ylabel("false alarms per day", fontsize=7); ax.set_title(f"{fl}, { {'mag': 'magnitude', 'ang': 'angle'}.get(mode, mode)}: quiet block", fontsize=7)
            # Ticks
            ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=6, rotation=30, ha="right"); ax.tick_params(axis="y", labelsize=6); ax.legend(fontsize=6)
            # Share panel
            ax = axes[2 * i + 1][j]
            # Bars in percent
            ax.bar(x - 0.2, [100.0 * v for v in pick("frac")], 0.4, label="model", color=(0.2, 0.45, 0.75))
            # Persistence
            ax.bar(x + 0.2, [100.0 * v for v in pick("pers_frac")], 0.4, label="persistence", color="0.6")
            # Missing marks
            mark(ax, pick("frac"), x - 0.2); mark(ax, pick("pers_frac"), x + 0.2)
            # Labels
            ax.set_ylabel("quiet-block slots in alarm, % (red x = not available)", fontsize=7)
            # Ticks
            ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=6, rotation=30, ha="right"); ax.tick_params(axis="y", labelsize=6)
    # Layout
    fig.tight_layout()
    # Save
    fig.savefig(out, dpi=150)
    # Close
    plt.close(fig)


def plot_delays(rows, modes, out):
    """Per mode, the delay of every detection as a point per run and score file, by episode."""
    # Lazy import
    import matplotlib
    # No display
    matplotlib.use("Agg")
    # Plotting
    import matplotlib.pyplot as plt
    # Panels across modes
    fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes) + 1, 3.5), squeeze=False)
    # Runs and files for markers and colors
    runs = list(dict.fromkeys(r["run"] for r in rows))
    # Files
    files = list(dict.fromkeys(r["file"] for r in rows))
    # Marker per file
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    # Color per run
    cmap = plt.get_cmap("tab10")
    # Episode positions from the files
    eps = sorted({e for r in rows for m in r["modes"].values() for e in m["scope"]})
    # Nothing
    if not eps:
        # Skip
        plt.close(fig)
        # Done
        return
    # Loop over modes
    for j, mode in enumerate(modes):
        # Axis
        ax = axes[0][j]
        # Loop over rows
        for r in rows:
            # Skip modes absent
            if mode not in r["modes"]:
                # Next
                continue
            # Points
            for ep, d in r["modes"][mode]["model"]:
                # Small horizontal offset per run so points do not overlap
                off = (runs.index(r["run"]) - (len(runs) - 1) / 2) * 0.08
                # Plot
                ax.plot(eps.index(ep) + off, d, markers[files.index(r["file"]) % len(markers)], color=cmap(runs.index(r["run"]) % 10), ms=5, alpha=0.85)
        # Zero line: the first logged change
        ax.axhline(0, color="0.5", lw=0.8)
        # Axes
        ax.set_xticks(range(len(eps))); ax.set_xticklabels([str(e) for e in eps], fontsize=7)
        # Labels
        ax.set_xlabel("episode", fontsize=7); ax.set_ylabel("delay from the episode's first logged change, s (negative = inside the pre-event allowance)", fontsize=6); ax.tick_params(axis="y", labelsize=6)
        # Title
        ax.set_title(f"{ {'mag': 'magnitude', 'ang': 'angle'}.get(mode, mode)}: color = run, marker = score file", fontsize=7)
    # Legend entries: runs by color, files by marker
    handles = [plt.Line2D([], [], color=cmap(k % 10), marker="o", ls="", label=run) for k, run in enumerate(runs)]
    # Files
    handles += [plt.Line2D([], [], color="0.3", marker=markers[k % len(markers)], ls="", label=fl) for k, fl in enumerate(files)]
    # Place
    fig.legend(handles=handles, fontsize=6, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02 - 0.03 * ((len(handles) + 2) // 3)))
    # Layout
    fig.tight_layout()
    # Save
    fig.savefig(out, dpi=150, bbox_inches="tight")
    # Close
    plt.close(fig)


def main():
    """Run over every run folder, print the comparison table when there are several, and write the plots when asked."""
    # Options
    args = parse_args()
    # All rows
    rows = []
    # Loop
    for run_dir in args.run_dir:
        # Summarize
        rows.extend(summarize_run(run_dir, args.files, args.modes))
    # Table for several runs
    if len(args.run_dir) > 1 and rows:
        # Scoring settings per file label, and whether rows with the same label used the same settings
        print("\nScoring settings per file (rows below with the same label share these unless flagged):")
        # Group
        for fl in dict.fromkeys(r["file"] for r in rows):
            # Settings seen under this label
            seen = [r["settings"] for r in rows if r["file"] == fl]
            # Reference
            ref = seen[0]
            # Print
            print(f"  {fl}: " + ", ".join(f"{k}={v}" for k, v in ref.items()))
            # Differences
            for r in rows:
                # Same label, different settings
                if r["file"] == fl and r["settings"] != ref:
                    # Flag
                    diff = {k: (ref[k], r["settings"][k]) for k in ref if ref[k] != r["settings"][k]}
                    # Print
                    print(f"    DIFFERENT SETTINGS in {r['run']}: {diff}")
        # Header
        print("\nComparison (quiet block = April to August 2025; FA/day = false alarms per day; in alarm = share of quiet-block slots in alarm)")
        # Column header
        head = f"{'run':22s} {'scoring':38s}"
        # Per mode
        for mode in args.modes:
            # Add
            head += f" | {mode + ' episodes':16s} {'FA/day':>7s} {'in alarm':>9s}"
        # Print
        print(head)
        # Rows
        for row in rows:
            # Start
            line = f"{row['run'][:22]:22s} {row['file'][:38]:38s}"
            # Per mode
            for mode in args.modes:
                # Cell
                c = row["modes"].get(mode)
                # Missing
                if c is None:
                    # Blank
                    line += f" | {'':16s} {'':>7s} {'':>9s}"
                else:
                    # Ids
                    ids = ",".join(str(i) for i in c["ids"]) or "none"
                    # Add
                    line += f" | {ids:16s} {rate(c['fa']):>7s} {pct(c['frac']):>9s}"
            # Print
            print(line)
    # Plots
    if args.plot and rows:
        # Folder
        os.makedirs(args.plot_dir, exist_ok=True)
        # Detection grid
        plot_detections(rows, args.modes, os.path.join(args.plot_dir, "detections.png"))
        # Quiet block
        plot_quiet(rows, args.modes, os.path.join(args.plot_dir, "quiet_block.png"))
        # Delays
        plot_delays(rows, args.modes, os.path.join(args.plot_dir, "delays.png"))
        # Say
        print(f"\nwrote {args.plot_dir}/detections.png, quiet_block.png, delays.png")


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
