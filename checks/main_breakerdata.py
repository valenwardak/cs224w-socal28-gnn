# =============================================================================
# File:        main_breakerdata.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-11-12
#
# Purpose:     Read-only comparison of the breaker records of egauge_3 and
#              egauge_4 with the meters' own measurements, with the sensor
#              placement derived from the source files rather than assumed.
#              The switch files record cb_143 and cb_145 as open (NO) from
#              the record start until 2024-02-01 00:00:00 standard time and
#              closed (NC) after, and these two breakers are also the
#              elements on which the two meters measure current. The meter
#              graph isolates both meters while the records say open.
#
#              For each meter the check derives and prints, from the network
#              file, the electrical graph and the period topology records:
#              the voltage measurement bus and the current element; which
#              end of the breaker the voltage bus is (the breaker's fbus or
#              its tbus); the buses that form the meter's node while the
#              breaker is recorded open and while it is recorded closed; the
#              complete component cut off by the open breaker, found by a
#              search through every element of a network-file snapshot with
#              the named breaker forced open and the other switching
#              elements at their listed statuses (a merged node can continue
#              through lines or transformers, so the own node is not assumed
#              to be the whole component); this bounded snapshot search does
#              not apply the time-series switch states of any recorded open
#              interval, so it is not a verification of every such interval;
#              the generators, inverters and loads documented
#              on the own node and elsewhere in that component; and the
#              non-conducting switching elements on the component's
#              boundary, with status and time-series availability. These facts, not a
#              general rule, say what a voltage reading during a recorded
#              open interval can and cannot mean for that meter.
#
#              It then aligns every slot with the breaker-state intervals in
#              UTC (records shifted by the pipeline's +8 h) and reports per
#              interval, per month inside the interval and per channel (S1,
#              S2, S3 current; L1, L2, L3 voltage): slots with a reading
#              present, with a valid finite magnitude, with a valid nonzero
#              magnitude, and the 5th, 50th, 95th and 99th percentiles and
#              the maximum. An open breaker carries no current, so valid,
#              correctly mapped, correctly timed nonzero current during a
#              recorded open interval is a disagreement between record and
#              measurement; neither voltage nor current alone authorizes an
#              inferred replacement switching history, and the check draws
#              no conclusion and writes no file. Missing months and missing
#              records are reported as unresolved.
#
# Usage (from the repo root):
#   python checks/main_breakerdata.py --features-dir /content/features
#   python checks/main_breakerdata.py --features-dir /content/features --meters egauge_3 egauge_4 --months 2023-08 2023-09 2023-10 2023-11 2023-12 2024-01 2024-02 2024-03
#
# Functions:
#   parse_args()                       command-line options with defaults
#   breaker_intervals(ts_dir, switch, record_end)
#                                      the switch's state intervals as (start_utc, end_utc, state, naive start)
#   interval_of(t, intervals)          the interval index of each UTC slot time, -1 outside
#   element_record(net, name)          (kind, record) of a named network element
#   cutoff_component(net, breaker, start_bus)
#                                      every bus reachable with the breaker open, through the whole network file
#   placement(meter_rec, net, periods, p_open, p_closed, with_series)
#                                      derived placement facts for one meter, own node and full component apart
#   channel_stats(rms, present, valid) present, valid, nonzero counts and percentiles for one channel
#   main()                             print the placement, the intervals and the per-channel table
# =============================================================================

# Standard library: command-line options
import argparse
# Standard library: switch records
import csv
# Standard library: JSON
import json
# Standard library: paths
import os
# Standard library: import path
import sys
# Third party: numerics
import numpy as np
# Third party: timestamps
import pandas as pd

# The repo root, one level above checks/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Import from the root
sys.path.insert(0, ROOT)

# Project: month loading with the same validity rules as the models, and the fixed UTC offset
import socal_modeldata as sd

# Default months: the record start to the month after the recorded closing
DEFAULT_MONTHS = ["2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03"]
# End of the record, standard time
RECORD_END = "2025-09-04 00:00:00"
# Switching element kinds in the network file
SWITCH_KINDS = ("Switch", "CB", "SwitchMultiPosition", "VFI")
# Source and load kinds
SOURCE_KINDS = ("Generator", "Inverter", "PV", "FuelCell")
# Two-ended element kinds
TWO_ENDED = ("Line", "Transformer") + SWITCH_KINDS


def parse_args():
    """Read the command-line options. Every setting has a default."""
    # Parser
    parser = argparse.ArgumentParser(description="Read-only comparison of breaker records with the meters' own measurements, with derived sensor placement")
    # Features
    parser.add_argument("--features-dir", default=os.path.join("data", "features"), help="Folder with <YYYY-MM>.npz and manifest.json")
    # Graph file with the meter records and the period topology
    parser.add_argument("--graph", default="graph_meters_periods.json", help="Time-indexed graph file (meter records, period topology)")
    # Network file
    parser.add_argument("--network", default=os.path.join("topology", "network_files", "circuit3", "2023-08-01T00h00m00.000000s.json"), help="Caltech network file")
    # Switch records
    parser.add_argument("--topology-dir", default="topology", help="Caltech's topology folder")
    # Meters
    parser.add_argument("--meters", nargs="+", default=["egauge_3", "egauge_4"], help="Meters to check")
    # Months
    parser.add_argument("--months", nargs="+", default=DEFAULT_MONTHS, help="Months YYYY-MM (the files' fixed standard-time months) to read")
    # Periods used to show the node with the breaker open and closed
    parser.add_argument("--period-open", type=int, default=8, help="A topology period in which the breakers are recorded open")
    # Closed
    parser.add_argument("--period-closed", type=int, default=9, help="A topology period in which the breakers are recorded closed")
    # Parse
    return parser.parse_args()


def breaker_intervals(ts_dir, switch, record_end=RECORD_END):
    """The switch's state intervals from <switch>-tbus_status.csv as (start_utc_s, end_utc_s, state, naive_start), records read as fixed standard time and shifted by the pipeline's offset."""
    # Path
    path = os.path.join(ts_dir, f"{switch}-tbus_status.csv")
    # Missing file
    if not os.path.isfile(path):
        # None
        return None
    # Read
    with open(path) as f:
        # Rows after the header
        rows = [r for r in csv.reader(f)][1:]
    # Records in time order
    recs = sorted((pd.Timestamp(t), s) for t, s in rows)
    # Bounds
    bounds = [t for t, _ in recs] + [pd.Timestamp(record_end)]
    # Intervals
    out = []
    # Loop
    for i, (t, s) in enumerate(recs):
        # UTC seconds
        a = int(bounds[i].timestamp()) + sd.UTC_OFFSET_SECONDS
        # End
        b = int(bounds[i + 1].timestamp()) + sd.UTC_OFFSET_SECONDS
        # Keep
        out.append((a, b, s, str(bounds[i])))
    # Done
    return out


def interval_of(t, intervals):
    """Index of the state interval holding each UTC slot time, -1 outside every interval."""
    # None yet
    idx = np.full(len(t), -1, dtype=np.int64)
    # Each interval
    for i, (a, b, _, _) in enumerate(intervals):
        # Inside
        idx[(t >= a) & (t < b)] = i
    # Done
    return idx


def element_record(net, name):
    """(kind, record) of the named element in the network file, or (None, None)."""
    # Every kind
    for kind, items in net.items():
        # Lists of records
        if isinstance(items, list):
            # Search
            for e in items:
                # Match
                if isinstance(e, dict) and e.get("name") == name:
                    # Found
                    return kind, e
    # None
    return None, None


def cutoff_component(net, breaker, start_bus):
    """
    The complete set of buses reachable from start_bus in a network-file
    snapshot with the named breaker forced open and every other switching
    element at its listed status: lines and transformers always conduct;
    switching elements conduct only when their listed status is NC; the
    breaker itself does not. The time-series states of a recorded open
    interval are not applied. Returns (buses, elements inside, boundary
    switching elements not conducting), each as sorted lists.
    """
    # Adjacency
    adj = {}
    # Elements inside the component and boundary candidates
    inside, boundary = {}, {}
    # Every two-ended kind
    for kind in TWO_ENDED:
        # Records
        for e in net.get(kind, []):
            # Ends
            ends = [e.get("fbus")] + [t.get("name") for t in e.get("tbus", [])]
            # Clean
            ends = [b for b in ends if b]
            # Conducts?
            if kind in SWITCH_KINDS:
                # Status
                st = e.get("tbus", [{}])[0].get("status") if e.get("tbus") else None
                # The breaker itself is open; others conduct only when NC
                conducts = e.get("name") != breaker and st == "NC"
            # Lines and transformers
            else:
                # Always
                conducts = True
            # Record
            for b in ends:
                # Neighbors
                adj.setdefault(b, [])
                # Add conducting links
                if conducts:
                    # Each other end
                    for c in ends:
                        # Not itself
                        if c != b:
                            # Link
                            adj[b].append((c, kind, e.get("name")))
            # Non-conducting switching elements are boundary candidates
            if not conducts:
                # Keep
                boundary[e.get("name")] = (kind, ends, e.get("tbus", [{}])[0].get("status") if e.get("tbus") else None)
    # Breadth-first search
    seen = {start_bus}
    # Queue
    queue = [start_bus]
    # Loop
    while queue:
        # Pop
        b = queue.pop(0)
        # Neighbors
        for c, kind, name in adj.get(b, []):
            # Element inside
            inside[name] = kind
            # New bus
            if c not in seen:
                # Mark
                seen.add(c)
                # Visit
                queue.append(c)
    # Boundary elements touching the component
    touching = {n: v for n, v in boundary.items() if set(v[1]) & seen}
    # Done
    return sorted(seen), sorted(inside.items()), touching


def placement(meter_rec, net, periods, p_open, p_closed, with_series=()):
    """Derived placement facts for one meter: a list of lines to print, from the network file and the period topology records; with_series names the switching elements that have a status time series."""
    # Lines
    out = []
    # Voltage bus and current elements
    vbus = meter_rec.get("voltage_bus")
    # Elements
    elements = meter_rec.get("current_elements", [])
    # Each current element
    for el in elements:
        # Record
        kind, rec = element_record(net, el)
        # Unknown
        if rec is None:
            # Line
            out.append(f"  current element {el}: not found in the network file; placement unresolved")
            # Next
            continue
        # Ends
        fbus = rec.get("fbus")
        # Far ends
        tbus = [t.get("name") for t in rec.get("tbus", [])]
        # Which side the voltage bus is
        side = "the fbus end" if vbus == fbus else ("the tbus end" if vbus in tbus else "neither end (voltage bus not on this element)")
        # Line
        out.append(f"  current element {el} ({kind}): fbus {fbus}, tbus {tbus}; the voltage bus {vbus} is at {side}")
    # Node buses with the breaker open and closed, from the period topology records
    meter_name = meter_rec.get("name") or ""
    # Topology entries
    t_open = periods[p_open].get("topology", {}).get(meter_name, {})
    # Closed
    t_closed = periods[p_closed].get("topology", {}).get(meter_name, {})
    # Buses
    b_open = set(t_open.get("node_buses", []))
    # Closed
    b_closed = set(t_closed.get("node_buses", []))
    # Line
    out.append(f"  node buses with the breaker recorded open (period {p_open}): {sorted(b_open)} (component {t_open.get('component')}); recorded closed (period {p_closed}): {sorted(b_closed)} (component {t_closed.get('component')})")
    # The complete component cut off by the open breaker, searched through the whole network file (not only the meter's own node)
    comp_buses, comp_elements, comp_boundary = cutoff_component(net, elements[0], vbus) if elements and vbus else ([], [], {})
    # Compare the full component with the meter's own node
    extra = sorted(set(comp_buses) - b_open)
    # Lines
    out.append(f"  own node while open (period {p_open} topology record): {sorted(b_open)}; complete cut-off component in the network-file snapshot with {elements[0] if elements else '?'} forced open and other switching elements at their listed statuses: {comp_buses}; buses beyond the own node: {extra if extra else 'none, the component is the own node'}")
    # Elements inside the component
    out.append(f"  elements inside the cut-off component: {[f'{k} {n}' for n, k in comp_elements] if comp_elements else 'none'}; buses rejoined when the breaker closes: {sorted(b_closed - b_open)}")
    # Documented sources and loads, own node and full component reported apart
    srcs_node, srcs_comp, loads_node, loads_comp = [], [], [], []
    # Each kind
    for kind in SOURCE_KINDS + ("Load",):
        # Records
        for e in net.get(kind, []):
            # Bus
            b = e.get("bus") or e.get("fbus")
            # Text
            txt = f"{kind} {e.get('name')} at {b}" + (f" ({e.get('gen_type')})" if e.get("gen_type") else "")
            # Own node
            if b in b_open:
                # Record
                (loads_node if kind == "Load" else srcs_node).append(txt)
            # Full component beyond the node
            elif b in comp_buses:
                # Record
                (loads_comp if kind == "Load" else srcs_comp).append(txt)
    # Lines
    out.append(f"  documented sources: on the own node {srcs_node if srcs_node else 'none'}; elsewhere in the component {srcs_comp if srcs_comp else 'none'}. Loads: on the own node {loads_node if loads_node else 'none'}; elsewhere {loads_comp if loads_comp else 'none'}")
    # Boundary switching elements of the full component (not conducting), with status and time-series availability
    ties = [f"{kind} {name} {ends} status {st} ({'time series' if name in with_series else 'no time series, held at the listed status'})" for name, (kind, ends, st) in sorted(comp_boundary.items()) if name not in elements]
    # Line
    out.append(f"  non-conducting switching elements on the component's boundary (ties that would connect it elsewhere if closed): {ties if ties else 'none'}")
    # Search scope
    out.append("  search scope: a network-file snapshot with the named breaker forced open and every other Line, Transformer, Switch, CB, SwitchMultiPosition and VFI record at its listed status; the time-series states of the recorded open interval are not applied, so this is a bounded snapshot check, not a verification of every recorded open interval; period-specific verification would need the relevant switching states applied explicitly")
    # What a voltage reading can mean here, derived
    if vbus in comp_buses and vbus in b_open:
        # Island side
        out.append(f"  reading of a voltage during a recorded open interval: the voltage bus is on the cut-off side, so a live voltage there would have to come from a documented source inside the component listed above, through a boundary tie listed above if it were closed, or from the breaker not being open; which of these, the files do not say")
    # Feeder side
    else:
        # Line
        out.append("  reading of a voltage during a recorded open interval: the voltage bus is on the feeder side, so a live voltage says nothing about the breaker state")
    # Done
    return out


def channel_stats(rms, present, valid):
    """For one channel over a set of slots: present count, valid count, valid-nonzero count, 5th, 50th, 95th, 99th percentiles and maximum of the valid magnitudes."""
    # Values
    vals = rms[valid]
    # Nonzero
    nz = int((vals > 0).sum())
    # Percentiles
    q = np.percentile(vals, [5, 50, 95, 99]).tolist() if len(vals) else [None] * 4
    # Max
    mx = float(vals.max()) if len(vals) else None
    # Done
    return int(present.sum()), int(valid.sum()), nz, q, mx


def main():
    """Print the placement, the intervals and the per-interval, per-month, per-channel table for each meter."""
    # Options
    args = parse_args()
    # Manifest
    manifest = sd.load_manifest(args.features_dir)
    # Channels
    channels = list(manifest["channels"])
    # Graph file
    with open(args.graph) as f:
        # Parse
        body = json.load(f)
    # Meter records and periods
    meters, periods = body["meters"], body["periods"]
    # Network
    with open(args.network) as f:
        # Parse
        net = json.load(f)
    # Switch folder
    ts_dir = os.path.join(args.topology_dir, "parameter_timeseries")
    # Months loaded once
    loaded = {month: sd.load_month(args.features_dir, month, allow_missing=True, manifest=manifest) for month in args.months}
    # Per meter
    for meter in args.meters:
        # Record
        rec = meters.get(meter)
        # Must exist
        if rec is None:
            # Say
            print(f"\n{meter}: not in {args.graph}; unresolved")
            # Next
            continue
        # Name for the topology lookup
        rec = dict(rec, name=meter)
        # Mapping
        print(f"\n{meter}: voltage bus {rec.get('voltage_bus')}, current elements {rec.get('current_elements')}, channel-to-phase {rec.get('phase')}, current ratings {rec.get('current_rating')}")
        # Switching elements with a status time series
        with_series = {fn[:-len("-tbus_status.csv")] for fn in os.listdir(ts_dir) if fn.endswith("-tbus_status.csv")}
        # Derived placement
        for line in placement(rec, net, periods, args.period_open, args.period_closed, with_series):
            # Print
            print(line)
        # Meter index
        mi = manifest["meters"].index(meter)
        # Each current element with a status file
        for el in rec.get("current_elements", []):
            # Intervals
            iv = breaker_intervals(ts_dir, el)
            # Missing
            if iv is None:
                # Say
                print(f"  {el}: no status file; breaker state unresolved from the records")
                # Next
                continue
            # Intervals in both conventions
            print(f"  {el} state intervals (standard time -> UTC):")
            # Each
            for a, b, s, naive in iv:
                # Line
                print(f"    {s:<3} from {naive} standard time = {pd.Timestamp(a, unit='s', tz='UTC')} UTC until {pd.Timestamp(b, unit='s', tz='UTC')} UTC")
            # Header
            print(f"    {'interval':<8} {'state':<5} {'month':<8} {'chan':<5} {'present':>8} {'valid':>8} {'nonzero':>8} {'p5':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9}")
            # Each month
            for month in args.months:
                # Arrays
                data = loaded[month]
                # No file
                if data is None:
                    # Say
                    print(f"    {'':<8} {'':<5} {month:<8} no feature file; unresolved for this month")
                    # Next
                    continue
                # Interval of each slot
                idx = interval_of(data["t"], iv)
                # Each interval present in the month
                for i in sorted(set(idx.tolist()) - {-1}):
                    # Slots
                    sel = idx == i
                    # State
                    state = iv[i][2]
                    # Current channels first, then voltage
                    for ci, ch in sorted(enumerate(channels), key=lambda x: (not x[1].startswith("S"), x[0])):
                        # Stats
                        pres, val, nz, q, mx = channel_stats(data["rms"][sel, mi, ci], data["present"][sel, mi, ci], data["mag_valid"][sel, mi, ci])
                        # Formatter
                        f = lambda v: "n/a" if v is None else f"{v:9.2f}"
                        # Line
                        print(f"    {i:<8} {state:<5} {month:<8} {ch:<5} {pres:>8} {val:>8} {nz:>8} {f(q[0]):>9} {f(q[1]):>9} {f(q[2]):>9} {f(q[3]):>9} {f(mx):>9}")
                # Slots outside every interval
                if (idx == -1).any():
                    # Say
                    print(f"    {'':<8} {'':<5} {month:<8} {int((idx == -1).sum())} slots outside the record's intervals")
    # Reading guide, no conclusion drawn
    print("\nreading: in an interval recorded NO, valid nonzero current on the breaker's own channels with plausible magnitudes is a disagreement between the record and the measurement for that interval; a voltage reading is read against the derived placement above; a month with no file or no readings is unresolved. A disagreement does not say which side is right or when a breaker changed state, and no replacement switching history is inferred here; mapping, placement, timestamp alignment and measurement validity remain the things to weigh.")


# Run only when executed directly
if __name__ == "__main__":
    # Go
    main()
