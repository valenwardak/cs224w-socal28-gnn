# =============================================================================
# File:        unit_tests/test_socal_score_split.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-29
#
# Purpose:     Tests for residual scoring output: monthly pieces, resume,
#              streaming combine without holding the split in RAM, and the
#              parquet schema main_score.py reads.
# =============================================================================

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

import socal_modeldata as sd
import socal_models as sm
import socal_score as ss


METERS = ["egauge_3", "egauge_4"]
CHANNELS = ["L1", "S1"]
RES_KEYS = ("mag_res", "ang_res", "pers_mag_res", "pers_ang_res")
VAL_KEYS = ("mag_valid", "ang_valid", "pers_mag_valid", "pers_ang_valid")


def residual_frame(times, meters, channels, mag=0.1, ang=1.0):
    """One month-shaped residual table with the score_split column layout."""
    cols = {}
    n = len(times)
    for meter in meters:
        for ch in channels:
            for k in RES_KEYS:
                cols[f"{meter}|{ch}|{k}"] = np.full(n, mag if "mag" in k else ang, dtype=np.float32)
            for k in VAL_KEYS:
                cols[f"{meter}|{ch}|{k}"] = np.ones(n, dtype=bool)
    table = pd.DataFrame(cols, index=pd.to_datetime(times, unit="s", utc=True))
    table.index.name = "t_utc"
    return table


def write_month_npz(features_dir, month, n, meters, channels):
    """A contiguous month file with valid readings, times inside the named month."""
    start, _ = sd.month_bounds(month)
    t = start + np.arange(n, dtype=np.int64) * sd.SLOT_SECONDS
    m, c = len(meters), len(channels)
    rms = np.full((n, m, c), 100.0, dtype=np.float32)
    ang = np.zeros((n, m, c), dtype=np.float32)
    mask = np.ones((n, m, c), dtype=bool)
    angle_ok = np.ones((n, m, c), dtype=bool)
    np.savez(os.path.join(features_dir, f"{month}.npz"), t=t, rms=rms, ang=ang, mask=mask, angle_ok=angle_ok)


def tiny_splits(start, end, start_utc, end_utc_exclusive):
    """One test_normal block spanning the given inclusive PST dates."""
    return {
        "train": [],
        "validation": [],
        "test_events": [],
        "test_normal": [{
            "start": start,
            "end": end,
            "start_utc": start_utc,
            "end_utc_exclusive": end_utc_exclusive,
        }],
    }


def gru_options():
    """Options score_split reads for a tiny GRU; predictions are not under test."""
    return {
        "kind": "gru",
        "history": 8,
        "horizon": 1,
        "max_gap": 4,
        "batch_size": 16,
    }


class CombineResidualPiecesTests(unittest.TestCase):
    """Streaming combine of month parquet files, the path that used to OOM."""

    def test_combine_preserves_schema_for_main_score(self):
        """Combined parquet has t_utc index and residual columns load_table expects."""
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            t0 = np.array([1_700_000_000, 1_700_000_010], dtype=np.int64)
            t1 = np.array([1_700_000_020, 1_700_000_030, 1_700_000_040], dtype=np.int64)
            a = residual_frame(t0, METERS, CHANNELS, mag=0.2, ang=3.0)
            b = residual_frame(t1, METERS, CHANNELS, mag=0.4, ang=5.0)
            p0 = os.path.join(tmp, "2024-10.parquet")
            p1 = os.path.join(tmp, "2024-11.parquet")
            a.to_parquet(p0)
            b.to_parquet(p1)
            dest = os.path.join(tmp, "test_normal_2024-10-01_2024-11-30.parquet")
            n = sm.combine_residual_pieces([p0, p1], dest, logs.append)
            self.assertEqual(n, 5)
            table = ss.load_table(tmp, "test_normal_2024-10-01_2024-11-30")
            self.assertEqual(table.index.name, "t_utc")
            self.assertEqual(len(table), 5)
            self.assertTrue(table.index.is_monotonic_increasing)
            meters, channels = ss.table_axes(table)
            self.assertEqual(meters, METERS)
            self.assertEqual(channels, CHANNELS)
            arr = ss.residual_arrays(table, meters, channels, prefix="")
            self.assertEqual(arr["mag"].shape, (5, 2, 2))
            self.assertTrue(np.all(arr["mag_valid"]))
            labels = ss.list_tables(tmp, "test_normal_")
            self.assertEqual(labels, ["test_normal_2024-10-01_2024-11-30"])

    def test_combine_matches_pandas_concat(self):
        """Streamed combine equals pd.concat of the same month files."""
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            frames = []
            paths = []
            for i, month in enumerate(("2024-10", "2024-11", "2024-12")):
                times = 1_700_000_000 + i * 1000 + np.arange(4, dtype=np.int64) * 10
                frame = residual_frame(times, METERS, CHANNELS, mag=0.1 * (i + 1))
                path = os.path.join(tmp, f"{month}.parquet")
                frame.to_parquet(path)
                frames.append(frame)
                paths.append(path)
            dest = os.path.join(tmp, "combined.parquet")
            sm.combine_residual_pieces(paths, dest, logs.append)
            got = pd.read_parquet(dest)
            expect = pd.concat(frames).sort_index()
            pd.testing.assert_frame_equal(got, expect, check_dtype=False)

    def test_combine_empty_raises(self):
        """No month files is an error, not a silent missing parquet."""
        with self.assertRaises(RuntimeError):
            sm.combine_residual_pieces([], "/tmp/none.parquet", lambda _: None)

    def test_combine_failure_removes_temp(self):
        """A failed combine must not leave dest or dest.tmp behind."""
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "block.parquet")
            missing = os.path.join(tmp, "nope.parquet")
            with self.assertRaises(Exception):
                sm.combine_residual_pieces([missing], dest, logs.append)
            self.assertFalse(os.path.exists(dest))
            self.assertFalse(os.path.exists(dest + ".tmp"))
            self.assertTrue(any("FAILED combining" in line for line in logs))

    def test_summary_from_pieces(self):
        """JSON summary is computed from month files, not a concatenated frame."""
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, month in enumerate(("2024-10", "2024-11")):
                times = 1_700_000_000 + i * 100 + np.arange(3, dtype=np.int64) * 10
                frame = residual_frame(times, METERS, CHANNELS, mag=float(i + 1), ang=10.0)
                path = os.path.join(tmp, f"{month}.parquet")
                frame.to_parquet(path)
                paths.append(path)
            summary = sm.write_residual_summary(paths, "lab", {"kind": "rgcn"}, 6, tmp, logs.append)
            self.assertEqual(summary["slots"], 6)
            self.assertEqual(summary["model"], "rgcn")
            self.assertAlmostEqual(summary["mag_res_median"], 1.5, places=5)
            self.assertTrue(os.path.exists(os.path.join(tmp, "lab.json")))


class ScoreSplitResumeTests(unittest.TestCase):
    """Full score_split over two months: write pieces, resume, combined schema."""

    def _setup_features(self, tmp):
        """Two months of tiny npz files, a GRU, and a split that covers both."""
        features = os.path.join(tmp, "features")
        os.makedirs(features)
        write_month_npz(features, "2024-10", 80, METERS, CHANNELS)
        write_month_npz(features, "2024-11", 80, METERS, CHANNELS)
        manifest = {"meters": METERS, "channels": CHANNELS}
        with open(os.path.join(features, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        splits = tiny_splits("2024-10-01", "2024-11-30", "2024-10-01T08:00:00+00:00", "2024-12-01T08:00:00+00:00")
        scale = np.ones((len(METERS), len(CHANNELS)), dtype=np.float32)
        model = sm.make_model("gru", manifest, hidden=8, embed=4, layers=1, dropout=0.0, seed=0, predict_delta=True)
        out_dir = os.path.join(tmp, "run")
        return features, splits, scale, manifest, model, out_dir

    def test_multi_month_score_writes_combined_parquet(self):
        """Scoring two months writes month pieces and one combined table main_score can read."""
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            features, splits, scale, manifest, model, out_dir = self._setup_features(tmp)
            sm.score_split(model, features, splits, "test_normal", scale, manifest, torch.device("cpu"), gru_options(), out_dir, logs.append)
            label = "test_normal_2024-10-01_2024-11-30"
            dest = os.path.join(out_dir, f"{label}.parquet")
            self.assertTrue(os.path.exists(dest), msg="\n".join(logs))
            self.assertTrue(os.path.exists(os.path.join(out_dir, f"{label}.json")))
            piece_dir = sm.residual_piece_dir(out_dir, label)
            self.assertTrue(os.path.exists(os.path.join(piece_dir, "2024-10.parquet")))
            self.assertTrue(os.path.exists(os.path.join(piece_dir, "2024-11.parquet")))
            table = ss.load_table(out_dir, label)
            self.assertEqual(table.index.name, "t_utc")
            self.assertGreater(len(table), 0)
            meters, channels = ss.table_axes(table)
            arr = ss.residual_arrays(table, meters, channels, prefix="")
            pers = ss.residual_arrays(table, meters, channels, prefix="pers_")
            self.assertEqual(arr["mag"].shape[0], len(table))
            self.assertEqual(pers["mag"].shape, arr["mag"].shape)
            self.assertTrue(any("combining 2 month files" in line for line in logs))
            self.assertTrue(any("rss " in line for line in logs))

    def test_resume_skips_completed_months(self):
        """A second score_split leaves existing month files untouched and rebuilds the combined parquet."""
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            features, splits, scale, manifest, model, out_dir = self._setup_features(tmp)
            sm.score_split(model, features, splits, "test_normal", scale, manifest, torch.device("cpu"), gru_options(), out_dir, logs.append)
            label = "test_normal_2024-10-01_2024-11-30"
            piece_dir = sm.residual_piece_dir(out_dir, label)
            oct_path = os.path.join(piece_dir, "2024-10.parquet")
            nov_path = os.path.join(piece_dir, "2024-11.parquet")
            dest = os.path.join(out_dir, f"{label}.parquet")
            oct_mtime = os.path.getmtime(oct_path)
            nov_mtime = os.path.getmtime(nov_path)
            with open(oct_path, "rb") as f:
                oct_bytes = f.read()
            os.remove(dest)
            os.remove(os.path.join(out_dir, f"{label}.json"))
            logs2 = []
            sm.score_split(model, features, splits, "test_normal", scale, manifest, torch.device("cpu"), gru_options(), out_dir, logs2.append)
            self.assertEqual(os.path.getmtime(oct_path), oct_mtime)
            self.assertEqual(os.path.getmtime(nov_path), nov_mtime)
            with open(oct_path, "rb") as f:
                self.assertEqual(f.read(), oct_bytes)
            self.assertTrue(any("resume" in line for line in logs2))
            self.assertTrue(os.path.exists(dest))
            table = ss.load_table(out_dir, label)
            self.assertGreater(len(table), 0)

    def test_incomplete_months_raise_without_combined_file(self):
        """Missing a month piece must not publish a partial combined parquet."""
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            features, splits, scale, manifest, model, out_dir = self._setup_features(tmp)
            label = "test_normal_2024-10-01_2024-11-30"
            piece_dir = sm.residual_piece_dir(out_dir, label)
            os.makedirs(piece_dir)
            residual_frame(np.array([1_700_000_000], dtype=np.int64), METERS, CHANNELS).to_parquet(
                os.path.join(piece_dir, "2024-10.parquet")
            )
            original = sm.sd.iter_blocks
            sm.sd.iter_blocks = lambda *args, **kwargs: iter(())
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    sm.score_split(model, features, splits, "test_normal", scale, manifest, torch.device("cpu"), gru_options(), out_dir, logs.append)
            finally:
                sm.sd.iter_blocks = original
            self.assertIn("still need months", str(ctx.exception))
            self.assertIn("2024-11", str(ctx.exception))
            self.assertFalse(os.path.exists(os.path.join(out_dir, f"{label}.parquet")))

    def test_existing_combined_parquet_is_skipped(self):
        """score_split does not rebuild a block that already has its parquet."""
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            features, splits, scale, manifest, model, out_dir = self._setup_features(tmp)
            sm.score_split(model, features, splits, "test_normal", scale, manifest, torch.device("cpu"), gru_options(), out_dir, logs.append)
            label = "test_normal_2024-10-01_2024-11-30"
            dest = os.path.join(out_dir, f"{label}.parquet")
            with open(dest, "rb") as f:
                before = f.read()
            mtime = os.path.getmtime(dest)
            logs2 = []
            sm.score_split(model, features, splits, "test_normal", scale, manifest, torch.device("cpu"), gru_options(), out_dir, logs2.append)
            self.assertEqual(os.path.getmtime(dest), mtime)
            with open(dest, "rb") as f:
                self.assertEqual(f.read(), before)
            self.assertTrue(any(line.startswith("skip ") for line in logs2))


if __name__ == "__main__":
    unittest.main()
