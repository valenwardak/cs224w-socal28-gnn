"""
Tests for download resume, cleanup, and missing-data reporting in socal_download.
"""

import json
import os
import tempfile
import unittest
from datetime import date

import socal_download as sd


def _make_complete_download(root, meter, label, channels, data_type="phasors"):
    """Create a DONE folder that passes validate_download."""
    folder = os.path.join(root, meter, label)
    subdir = sd.DATA_SUBDIR[data_type]
    os.makedirs(os.path.join(folder, subdir), exist_ok=True)
    for channel in channels:
        path = os.path.join(folder, subdir, sd.channel_csv_name(meter, channel))
        with open(path, "w") as f:
            f.write("t\n")
    sd.write_request(
        folder,
        {
            "meter": meter,
            "data_type": data_type,
            "channels": channels,
            "start": label.split("_")[0],
            "end": label.split("_")[1],
        },
    )
    sd.mark_done(folder)
    return folder


def _make_incomplete_download(root, meter, label, channels):
    """Folder without DONE and without all channel CSVs (simulates a failed run)."""
    folder = os.path.join(root, meter, label)
    os.makedirs(os.path.join(folder, sd.PHASORS_SUBDIR), exist_ok=True)
    sd.write_request(
        folder,
        {
            "meter": meter,
            "data_type": "phasors",
            "channels": channels,
            "start": label.split("_")[0],
            "end": label.split("_")[1],
        },
    )
    return folder


class TestPrepareDownloadFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_done_folder_is_not_purged(self):
        meter = "egauge_7"
        label = "2024-01-01_2024-01-31"
        folder = _make_complete_download(self.tmp, meter, label, ["L1", "L2"])
        csv_path = os.path.join(folder, sd.PHASORS_SUBDIR, "egauge_7-L1.csv")
        self.assertTrue(os.path.isfile(csv_path))

        should_download = sd.prepare_download_folder(folder)
        self.assertFalse(should_download)
        self.assertTrue(sd.is_done(folder))
        self.assertTrue(os.path.isfile(csv_path))

    def test_incomplete_folder_is_purged_on_prepare(self):
        meter = "egauge_7"
        label = "2024-02-01_2024-02-01_L1"
        folder = _make_incomplete_download(self.tmp, meter, label, ["L1"])
        self.assertTrue(os.path.isdir(folder))

        should_download = sd.prepare_download_folder(folder)
        self.assertTrue(should_download)
        self.assertFalse(os.path.exists(folder))

    def test_remove_partial_download_skips_done(self):
        meter = "egauge_7"
        label = "2024-03-01_2024-03-31"
        folder = _make_complete_download(self.tmp, meter, label, ["S1"])
        sd.remove_partial_download(folder)
        self.assertTrue(sd.is_done(folder))
        self.assertTrue(
            os.path.isfile(os.path.join(folder, sd.PHASORS_SUBDIR, "egauge_7-S1.csv"))
        )


class TestMissingDownloadReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.meter = "egauge_7"

    def test_incomplete_done_folder_reported_in_missing(self):
        label = "2024-04-01_2024-04-30"
        folder = _make_complete_download(self.tmp, self.meter, label, ["L1", "L2"])
        os.remove(os.path.join(folder, sd.PHASORS_SUBDIR, "egauge_7-L2.csv"))

        missing = sd.find_missing_channels(self.tmp, [self.meter])
        entries = missing["missing"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["meter"], self.meter)
        self.assertEqual(entries[0]["label"], label)
        self.assertEqual(entries[0]["channels"], ["L2"])

    def test_recorded_leaf_failure_appears_in_missing_json(self):
        label = "2024-05-01_2024-05-01_S2"
        start = date(2024, 5, 1)
        sd.record_missing(self.tmp, self.meter, start, start, ["S2"], label)

        path = os.path.join(self.tmp, sd.MISSING_FILE)
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(len(data["missing"]), 1)
        self.assertEqual(data["missing"][0]["label"], label)

    def test_resolved_failure_removed_from_missing_after_valid_done(self):
        label = "2024-06-01_2024-06-01_L3"
        start = date(2024, 6, 1)
        sd.record_missing(self.tmp, self.meter, start, start, ["L3"], label)
        _make_complete_download(self.tmp, self.meter, label, ["L3"])

        missing = sd.find_missing_channels(self.tmp, [self.meter])
        self.assertEqual(missing["missing"], [])


if __name__ == "__main__":
    unittest.main()
