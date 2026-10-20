"""Measurement 模块单元测试。"""

import unittest

from src.calibration import Calibration
from src.measurement import Measurement


class TestMeasurement(unittest.TestCase):
    def test_conversion(self):
        measurement = Measurement(Calibration(pixels_per_mm=5.0))
        result = measurement.measure(300, 150)
        self.assertAlmostEqual(result.width_mm, 60.0)
        self.assertAlmostEqual(result.height_mm, 30.0)

    def test_long_short_sides(self):
        measurement = Measurement(Calibration(pixels_per_mm=5.0))
        result = measurement.measure(150, 300)
        self.assertAlmostEqual(result.length_mm, 60.0)
        self.assertAlmostEqual(result.short_side_mm, 30.0)

    def test_uncalibrated_raises(self):
        with self.assertRaises(RuntimeError):
            Measurement(Calibration())

    def test_invalid_ratio_raises(self):
        with self.assertRaises(ValueError):
            Measurement(Calibration(pixels_per_mm=0))

    def test_nonpositive_size_raises(self):
        measurement = Measurement(Calibration(pixels_per_mm=5.0))
        with self.assertRaises(ValueError):
            measurement.measure(0, 100)
        with self.assertRaises(ValueError):
            measurement.measure(100, -1)


if __name__ == "__main__":
    unittest.main()
