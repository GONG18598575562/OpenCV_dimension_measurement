"""RecordManager 模块单元测试。"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.calibration import Calibration
from src.contour_detector import DetectedObject
from src.measurement import Measurement
from src.record_manager import RecordManager


def make_object(width=300.0, height=150.0, area=44700.0):
    return DetectedObject(
        center=(250.0, 155.0),
        width_px=width,
        height_px=height,
        angle=10.0,
        area_px=area,
        box_points=np.zeros((4, 2)),
    )


class TestRecordManager(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "records.csv"
        self.manager = RecordManager(file_path=self.path)
        self.measurement = Measurement(
            Calibration(pixels_per_mm=5.0)
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_fields(self):
        record = self.manager.save_record(
            obj=make_object(),
            measurement=self.measurement,
            mode="manual",
        )

        self.assertEqual(record["mode"], "manual")
        self.assertAlmostEqual(record["width_mm"], 60.0, places=1)
        self.assertAlmostEqual(record["height_mm"], 30.0, places=1)
        self.assertIn("pixels_per_mm", record)

    def test_rect_and_contour_area_distinct(self):
        """矩形面积 = 长宽乘积；轮廓面积 = contourArea 换算。"""
        obj = make_object(width=300.0, height=150.0, area=30000.0)
        record = self.manager.save_record(
            obj=obj, measurement=self.measurement
        )

        self.assertAlmostEqual(record["rect_area_mm2"], 1800.0, places=1)
        self.assertAlmostEqual(record["contour_area_mm2"], 1200.0, places=1)
        self.assertNotEqual(
            record["rect_area_mm2"], record["contour_area_mm2"]
        )

    def test_object_id_auto_increment(self):
        r1 = self.manager.save_record(
            make_object(), self.measurement
        )
        r2 = self.manager.save_record(
            make_object(), self.measurement
        )
        self.assertEqual(r1["object_id"], 1)
        self.assertEqual(r2["object_id"], 2)

    def test_explicit_object_id_continues_counter(self):
        record = self.manager.save_record(
            make_object(), self.measurement, object_id=7
        )
        nxt = self.manager.save_record(
            make_object(), self.measurement
        )
        self.assertEqual(record["object_id"], 7)
        self.assertEqual(nxt["object_id"], 8)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.manager.save_record(
                make_object(), self.measurement, mode="bogus"
            )

    def test_invalid_object_id(self):
        with self.assertRaises(ValueError):
            self.manager.save_record(
                make_object(), self.measurement, object_id=0
            )

    def test_csv_written_with_header(self):
        self.manager.save_record(make_object(), self.measurement)

        with self.path.open(encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        self.assertEqual(len(rows), 1)
        self.assertIn("mode", rows[0])
        self.assertIn("contour_area_mm2", rows[0])


if __name__ == "__main__":
    unittest.main()
