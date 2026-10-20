
from dataclasses import dataclass

from src.calibration import Calibration


@dataclass
class MeasurementResult:
    """物体尺寸测量结果，单位为毫米。"""

    width_mm: float
    height_mm: float

    @property
    def length_mm(self) -> float:
        """返回较长的一边。"""
        return max(self.width_mm, self.height_mm)

    @property
    def short_side_mm(self) -> float:
        """返回较短的一边。"""
        return min(self.width_mm, self.height_mm)


class Measurement:
    """根据标定比例，将物体的像素尺寸转换为实际尺寸。"""

    def __init__(self, calibration: Calibration):
        if calibration.pixels_per_mm is None:
            raise RuntimeError("请先完成像素比例标定")

        if calibration.pixels_per_mm <= 0:
            raise ValueError("像素比例必须大于 0")

        self.calibration = calibration

    def measure(
        self,
        width_px: float,
        height_px: float,
    ) -> MeasurementResult:
        """
        将物体的像素宽度和高度换算为毫米。

        Args:
            width_px: 物体宽度，单位 px。
            height_px: 物体高度，单位 px。
        """
        if width_px <= 0 or height_px <= 0:
            raise ValueError("物体像素宽度和高度必须大于 0")

        width_mm = self.calibration.pixel_to_mm(width_px)
        height_mm = self.calibration.pixel_to_mm(height_px)

        return MeasurementResult(
            width_mm=width_mm,
            height_mm=height_mm,
        )


def main():
    """独立测试尺寸测量模块。"""

    # 假设标定结果为 5 px/mm。
    calibration = Calibration(pixels_per_mm=5.0)
    measurement = Measurement(calibration)

    result = measurement.measure(
        width_px=300,
        height_px=150,
    )

    print(f"物体宽度：{result.width_mm:.2f} mm")
    print(f"物体高度：{result.height_mm:.2f} mm")
    print(f"长边尺寸：{result.length_mm:.2f} mm")
    print(f"短边尺寸：{result.short_side_mm:.2f} mm")


if __name__ == "__main__":
    main()
