"""Calibration 模块单元测试。"""

import math
import tempfile
import unittest
from pathlib import Path

from src.calibration import Calibration


class TestCalibrate(unittest.TestCase):
    def test_calculate_ratio(self):
        calibration = Calibration()
        ratio = calibration.calibrate(150.0, 30.0)
        self.assertAlmostEqual(ratio, 5.0)
        self.assertAlmostEqual(calibration.pixels_per_mm, 5.0)

    def test_calibrate_records_metadata(self):
        calibration = Calibration()
        calibration.calibrate(
            150.0,
            30.0,
            reference={
                "real_length_mm": 30.0,
                "pixel_length": 150.0,
                "edge": "宽度",
            },
        )
        self.assertEqual(calibration.method, "reference_object")
        self.assertEqual(calibration.reference["edge"], "宽度")
        self.assertIsNotNone(calibration.calibrated_at)

    def test_calibrate_default_reference(self):
        calibration = Calibration()
        calibration.calibrate(150.0, 30.0)
        self.assertAlmostEqual(
            calibration.reference["real_length_mm"], 30.0
        )

    def test_invalid_pixel_length(self):
        calibration = Calibration()
        with self.assertRaises(ValueError):
            calibration.calibrate(0, 30.0)

    def test_invalid_real_length(self):
        calibration = Calibration()
        with self.assertRaises(ValueError):
            calibration.calibrate(150.0, -1.0)

    def test_non_finite_input(self):
        calibration = Calibration()
        with self.assertRaises(ValueError):
            calibration.calibrate(float("nan"), 30.0)


class TestConversion(unittest.TestCase):
    def test_pixel_to_mm(self):
        calibration = Calibration(pixels_per_mm=5.0)
        self.assertAlmostEqual(calibration.pixel_to_mm(250), 50.0)

    def test_mm_to_pixel(self):
        calibration = Calibration(pixels_per_mm=5.0)
        self.assertAlmostEqual(calibration.mm_to_pixel(50), 250.0)

    def test_uncalibrated_raises(self):
        calibration = Calibration()
        with self.assertRaises(RuntimeError):
            calibration.pixel_to_mm(10)

    def test_invalid_ratio_rejected(self):
        with self.assertRaises(ValueError):
            Calibration(pixels_per_mm=-1)

    def test_zero_pixel_length_rejected(self):
        calibration = Calibration(pixels_per_mm=5.0)
        with self.assertRaises(ValueError):
            calibration.pixel_to_mm(-5)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "calibration.yaml"

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_load_roundtrip(self):
        calibration = Calibration(
            pixels_per_mm=3.5,
            resolution=(640, 480),
            method="reference_object",
            reference={"real_length_mm": 50.0, "edge": "高度"},
        )
        calibration.calibrated_at = "2026-10-20T10:00:00+08:00"
        calibration.save(self.path)

        loaded = Calibration.load(self.path)
        self.assertAlmostEqual(loaded.pixels_per_mm, 3.5)
        self.assertEqual(loaded.resolution, (640, 480))
        self.assertEqual(loaded.method, "reference_object")
        self.assertEqual(loaded.reference["edge"], "高度")
        self.assertEqual(loaded.calibrated_at, "2026-10-20T10:00:00+08:00")

    def test_saved_file_contains_version(self):
        Calibration(pixels_per_mm=2.0).save(self.path)
        import yaml

        with self.path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)

        self.assertEqual(data["version"], Calibration.CONFIG_VERSION)

    def test_load_legacy_file_without_extra_fields(self):
        self.path.write_text(
            "pixels_per_mm: 2.0\nunit: mm\n", encoding="utf-8"
        )
        loaded = Calibration.load(self.path)
        self.assertAlmostEqual(loaded.pixels_per_mm, 2.0)
        self.assertIsNone(loaded.resolution)
        self.assertIsNone(loaded.reference)

    def test_load_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            Calibration.load(
                Path(self._tmp.name) / "missing.yaml"
            )

    def test_load_corrupt_file(self):
        self.path.write_text("!!!: [[ broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            Calibration.load(self.path)

    def test_load_invalid_reference_type(self):
        self.path.write_text(
            "pixels_per_mm: 2.0\nreference: [1, 2]\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            Calibration.load(self.path)

    def test_save_without_calibration(self):
        with self.assertRaises(RuntimeError):
            Calibration().save(self.path)

    def test_clear_removes_file(self):
        Calibration(pixels_per_mm=2.0).save(self.path)
        self.assertTrue(Calibration.clear(self.path))
        self.assertFalse(self.path.exists())
        self.assertFalse(Calibration.clear(self.path))


class TestResolutionValidation(unittest.TestCase):
    def test_invalid_resolution(self):
        with self.assertRaises(ValueError):
            Calibration(pixels_per_mm=1.0, resolution=(0, 480))

    def test_non_finite_ratio(self):
        with self.assertRaises(ValueError):
            Calibration(pixels_per_mm=math.inf)


if __name__ == "__main__":
    unittest.main()
