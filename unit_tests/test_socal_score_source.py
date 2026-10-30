# =============================================================================
# File:        unit_tests/test_socal_score_source.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-29
#
# Purpose:     Numerical equivalence of incremental score_source with the
#              original all-in-memory implementation, plus column-selective
#              residual loads used to keep test_normal out of RAM.
# =============================================================================

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import socal_score as ss


METERS = ["egauge_3", "egauge_4"]
CHANNELS = ["L1", "S1"]
RES_KEYS = ("mag_res", "ang_res", "pers_mag_res", "pers_ang_res")
VAL_KEYS = ("mag_valid", "ang_valid", "pers_mag_valid", "pers_ang_valid")


def residual_frame(times, mag, ang, pers_mag=None, pers_ang=None, mag_valid=None):
    """Residual table with independent model and persistence columns."""
    n = len(times)
    if pers_mag is None:
        pers_mag = mag
    if pers_ang is None:
        pers_ang = ang
    if mag_valid is None:
        mag_valid = np.ones(n, dtype=bool)
    cols = {}
    for meter in METERS:
        for ch in CHANNELS:
            cols[f"{meter}|{ch}|mag_res"] = np.asarray(mag, dtype=np.float32)
            cols[f"{meter}|{ch}|ang_res"] = np.asarray(ang, dtype=np.float32)
            cols[f"{meter}|{ch}|pers_mag_res"] = np.asarray(pers_mag, dtype=np.float32)
            cols[f"{meter}|{ch}|pers_ang_res"] = np.asarray(pers_ang, dtype=np.float32)
            cols[f"{meter}|{ch}|mag_valid"] = mag_valid
            cols[f"{meter}|{ch}|ang_valid"] = mag_valid
            cols[f"{meter}|{ch}|pers_mag_valid"] = mag_valid
            cols[f"{meter}|{ch}|pers_ang_valid"] = mag_valid
    table = pd.DataFrame(cols, index=pd.to_datetime(times, unit="s", utc=True))
    table.index.name = "t_utc"
    return table


def write_events(path, rows):
    """Minimal events.csv: t_utc, scored, episode, device."""
    pd.DataFrame(rows).to_csv(path, index=False)


def settings(events, mode="both", common=True, alpha=0.1, min_meters=2):
    """Fixed scoring settings used by both implementations."""
    return {
        "alpha": alpha,
        "mode": mode,
        "min_meters": min_meters,
        "merge_slots": 6,
        "before_minutes": 1.0,
        "after_minutes": 10.0,
        "calib_label": "validation_2025-02-03_2025-02-16",
        "test_prefixes": ["test_events_", "test_normal_"],
        "events": events,
        "n_boot": 50,
        "seed": 0,
        "common": common,
    }


def score_source_all_in_memory(run_dir, prefix, settings, other_prefix=None):
    """
    Original score_source: load every test table into RAM, then detect.
    Kept here as the numerical reference; production code no longer does this.
    """
    before = pd.Timedelta(minutes=settings["before_minutes"])
    after = pd.Timedelta(minutes=settings["after_minutes"])
    cal = ss.load_table(run_dir, settings["calib_label"])
    meters, channels = ss.table_axes(cal)
    arrays = ss.residual_arrays(cal, meters, channels, prefix)
    if other_prefix is not None and settings["common"]:
        ss.common_mask(arrays, ss.residual_arrays(cal, meters, channels, other_prefix))
    thr = ss.calibrate(arrays, settings["alpha"])
    missing = {kind: [f"{meters[mi]}|{channels[ci]}" for mi, ci in zip(*np.nonzero(~np.isfinite(thr[f"thr_{kind}"])))] for kind in ("mag", "ang")}
    episodes = ss.load_episodes(settings["events"])
    labels = [lab for p in settings["test_prefixes"] for lab in ss.list_tables(run_dir, p)]
    tables = []
    for label in labels:
        table = ss.load_table(run_dir, label)
        arr = ss.residual_arrays(table, meters, channels, prefix)
        if other_prefix is not None and settings["common"]:
            ss.common_mask(arr, ss.residual_arrays(table, meters, channels, other_prefix))
        tables.append((label, arr))
    blocks = [(pd.to_datetime(arr["t"][0], utc=True), pd.to_datetime(arr["t"][-1], utc=True)) for _, arr in tables]
    scoped, partial = ss.episodes_in_scope(episodes, blocks, before, after)
    alarms, monitored_s, per_block = [], 0.0, []
    for label, arr in tables:
        fires = ss.exceedances(arr, thr, settings["mode"])
        score = ss.slot_scores(fires)
        mon = ss.monitored_slots(arr, thr, settings["mode"], settings["min_meters"])
        mon &= ~ss.window_mask(arr["t"], partial, before, after)
        alarm = (score >= settings["min_meters"]) & mon
        eps = ss.alarm_episodes(arr["t"], alarm, settings["merge_slots"])
        alarms.extend(eps)
        monitored_s += float(mon.sum()) * ss.SLOT_SECONDS
        spans = [float((e - s0).total_seconds()) + ss.SLOT_SECONDS for s0, e in eps]
        per_block.append({"label": label, "slots": int(len(arr["t"])), "monitored_slots": int(mon.sum()), "alarm_slots": int(alarm.sum()),
                          "alarm_slot_fraction": (float(alarm.sum()) / float(mon.sum())) if mon.sum() else None,
                          "alarm_episodes": len(eps), "merged_span_median_s": float(np.median(spans)) if spans else None, "merged_span_max_s": float(max(spans)) if spans else None})
    results, unmatched = ss.match(alarms, scoped, before, after)
    n_ep, n_det, n_false = len(results), sum(r["detected"] for r in results), len(unmatched)
    per_split = {}
    for p in settings["test_prefixes"]:
        spans = [(b, blocks[i]) for i, b in enumerate(per_block) if b["label"].startswith(p)]
        sec = sum(b["monitored_slots"] for b, _ in spans) * ss.SLOT_SECONDS
        a_slots = sum(b["alarm_slots"] for b, _ in spans)
        n_unm = sum(1 for a0, _ in unmatched if any(b0 <= a0 <= b1 for _, (b0, b1) in spans))
        split_spans = [float((e - a0).total_seconds()) + ss.SLOT_SECONDS for a0, e in alarms if any(b0 <= a0 <= b1 for _, (b0, b1) in spans)]
        per_split[p] = {"false_alarm_episodes": n_unm, "monitored_days": sec / 86400.0, "false_alarms_per_day": (n_unm / (sec / 86400.0)) if sec > 0 else None,
                        "alarm_slot_fraction": (a_slots / (sec / ss.SLOT_SECONDS)) if sec > 0 else None,
                        "alarm_episodes": len(split_spans), "merged_span_median_s": float(np.median(split_spans)) if split_spans else None, "merged_span_max_s": float(max(split_spans)) if split_spans else None}
    days = monitored_s / 86400.0
    recall = n_det / n_ep if n_ep else None
    precision = n_det / (n_det + n_false) if (n_det + n_false) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else (0.0 if n_ep else None)
    delays = [r["delay_s"] for r in results if r["detected"]]
    boot = ss.bootstrap([r["detected"] for r in results], n_false, settings["n_boot"], settings["seed"])
    return {
        "source": "persistence" if prefix == "pers_" else "model",
        "statistic": settings.get("statistic", "channel"),
        "restriction": None,
        "grid": None,
        "calibration": {"label": settings["calib_label"], "alpha": settings["alpha"],
                        "coverage_mag_mean": float(np.nanmean(thr["cov_mag"])) if np.isfinite(thr["cov_mag"]).any() else None,
                        "coverage_ang_mean": float(np.nanmean(thr["cov_ang"])) if np.isfinite(thr["cov_ang"]).any() else None,
                        "coverage_mag": thr["cov_mag"].tolist(), "coverage_ang": thr["cov_ang"].tolist(),
                        "n_mag": thr["n_mag"].tolist(), "n_ang": thr["n_ang"].tolist(),
                        "channels_without_threshold": missing},
        "thresholds": {"meters": meters, "channels": channels, "mag": thr["thr_mag"].tolist(), "ang": thr["thr_ang"].tolist()},
        "blocks": per_block,
        "episodes_total": int(len(episodes)), "episodes_in_scope": n_ep, "episodes_detected": n_det,
        "episodes_partial_excluded": [{"episode": int(e["episode"]), "first": e["first"].isoformat()} for _, e in partial.iterrows()],
        "episode_results": [{"episode": r["episode"], "first": r["first"].isoformat(), "detected": r["detected"], "delay_s": r["delay_s"], "alarm_start": r["alarm_start"].isoformat() if r["alarm_start"] is not None else None} for r in results],
        "false_alarm_episodes": n_false, "monitored_days": days, "false_alarms_per_day": (n_false / days) if days > 0 else None,
        "per_split": per_split,
        "false_alarm_list": [(s.isoformat(), e.isoformat()) for s, e in unmatched],
        "recall": recall, "precision": precision, "f1": f1,
        "delay_median_s": float(np.median(delays)) if delays else None, "delay_max_s": float(np.max(delays)) if delays else None,
        "bootstrap": boot,
    }


def assert_nested_close(test, got, expect, path=""):
    """Walk the score_source dict and compare numbers, lists, and strings."""
    if isinstance(expect, dict):
        test.assertEqual(set(got.keys()), set(expect.keys()), path)
        for k in expect:
            assert_nested_close(test, got[k], expect[k], f"{path}.{k}")
        return
    if isinstance(expect, list):
        test.assertEqual(len(got), len(expect), path)
        for i, (g, e) in enumerate(zip(got, expect)):
            assert_nested_close(test, g, e, f"{path}[{i}]")
        return
    if isinstance(expect, float):
        if np.isnan(expect):
            test.assertTrue(np.isnan(got), path)
        else:
            test.assertTrue(np.isclose(got, expect, rtol=0, atol=1e-6, equal_nan=True), f"{path}: {got} != {expect}")
        return
    test.assertEqual(got, expect, path)


class LoadHelpersTests(unittest.TestCase):
    """Column-selective loaders match a full pandas read."""

    def test_load_residual_arrays_matches_residual_arrays(self):
        times = 1_700_000_000 + np.arange(20, dtype=np.int64) * 10
        mag = np.linspace(0.01, 0.5, 20)
        ang = np.linspace(0.5, 8.0, 20)
        pers_mag = mag * 0.8
        table = residual_frame(times, mag, ang, pers_mag=pers_mag)
        with tempfile.TemporaryDirectory() as tmp:
            table.to_parquet(os.path.join(tmp, "block.parquet"))
            loaded = ss.load_table(tmp, "block")
            for prefix in ("", "pers_"):
                from_frame = ss.residual_arrays(loaded, METERS, CHANNELS, prefix)
                from_disk = ss.load_residual_arrays(tmp, "block", METERS, CHANNELS, prefix)
                np.testing.assert_allclose(from_disk["mag"], from_frame["mag"], equal_nan=True)
                np.testing.assert_allclose(from_disk["ang"], from_frame["ang"], equal_nan=True)
                np.testing.assert_array_equal(from_disk["mag_valid"], from_frame["mag_valid"])
                np.testing.assert_array_equal(from_disk["ang_valid"], from_frame["ang_valid"])
                np.testing.assert_array_equal(pd.to_datetime(from_disk["t"], utc=True), pd.to_datetime(from_frame["t"], utc=True))

    def test_table_span_matches_index(self):
        times = 1_700_000_000 + np.arange(7, dtype=np.int64) * 10
        table = residual_frame(times, np.ones(7), np.ones(7))
        with tempfile.TemporaryDirectory() as tmp:
            table.to_parquet(os.path.join(tmp, "block.parquet"))
            first, last = ss.table_span(tmp, "block")
            loaded = ss.load_table(tmp, "block")
            self.assertEqual(first, loaded.index[0])
            self.assertEqual(last, loaded.index[-1])

    def test_parquet_axes_match_table_axes(self):
        times = 1_700_000_000 + np.arange(3, dtype=np.int64) * 10
        table = residual_frame(times, np.ones(3), np.ones(3))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "block.parquet")
            table.to_parquet(path)
            self.assertEqual(ss.parquet_axes(path), ss.table_axes(ss.load_table(tmp, "block")))


class ScoreSourceEquivalenceTests(unittest.TestCase):
    """Incremental score_source matches the original all-in-memory pipeline."""

    def _run_dir(self, tmp):
        """Calib, one event block, one quiet block, and an events table."""
        t0 = 1_700_000_000
        calib_t = t0 + np.arange(80, dtype=np.int64) * 10
        event_t = t0 + 10_000 + np.arange(120, dtype=np.int64) * 10
        quiet_t = t0 + 50_000 + np.arange(200, dtype=np.int64) * 10
        rng = np.random.default_rng(1)
        calib = residual_frame(calib_t, rng.uniform(0.05, 0.15, 80), rng.uniform(0.5, 1.5, 80), pers_mag=rng.uniform(0.05, 0.2, 80))
        mag_e = rng.uniform(0.05, 0.12, 120)
        mag_e[40:46] = 2.0
        pers_e = rng.uniform(0.05, 0.12, 120)
        pers_e[40:46] = 2.0
        events = residual_frame(event_t, mag_e, np.full(120, 0.8), pers_mag=pers_e)
        mag_q = rng.uniform(0.05, 0.12, 200)
        mag_q[10:14] = 3.0
        mag_q[180:186] = 3.0
        pers_q = rng.uniform(0.05, 0.12, 200)
        pers_q[10:14] = 0.08
        pers_q[180:186] = 3.0
        quiet = residual_frame(quiet_t, mag_q, np.full(200, 0.8), pers_mag=pers_q)
        calib.to_parquet(os.path.join(tmp, "validation_2025-02-03_2025-02-16.parquet"))
        events.to_parquet(os.path.join(tmp, "test_events_2023-09-15_2023-09-19.parquet"))
        quiet.to_parquet(os.path.join(tmp, "test_normal_2025-04-01_2025-08-31.parquet"))
        ev_time = pd.Timestamp(event_t[42], unit="s", tz="UTC").isoformat()
        write_events(os.path.join(tmp, "events.csv"), [
            {"t_utc": ev_time, "scored": True, "episode": 1, "device": "cb_121"},
        ])
        return settings(os.path.join(tmp, "events.csv"))

    def test_equivalence_common_and_modes(self):
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            base = self._run_dir(tmp)
            for common in (True, False):
                for mode in ("mag", "ang", "both"):
                    cfg = dict(base, common=common, mode=mode)
                    for prefix, other in (("", "pers_"), ("pers_", "")):
                        got = ss.score_source(tmp, prefix, cfg, other_prefix=other, log=logs.append)
                        expect = score_source_all_in_memory(tmp, prefix, cfg, other_prefix=other)
                        assert_nested_close(self, got, expect, f"common={common} mode={mode} prefix={prefix!r}")

    def test_logs_progress(self):
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_dir(tmp)
            ss.score_source(tmp, "", cfg, other_prefix="pers_", log=logs.append)
            text = "\n".join(logs)
            self.assertIn("calibrating", text)
            self.assertIn("rss ", text)
            self.assertIn("test_events_", text)
            self.assertIn("test_normal_", text)


if __name__ == "__main__":
    unittest.main()
