
import csv
from datetime import datetime
from pathlib import Path

from src.contour_detector import DetectedObject
from src.measurement import Measurement


class RecordManager:
    """负责将物体测量结果保存到 CSV 文件。"""

    FIELDNAMES = [
        "timestamp",
        "object_id",
        "mode",
        "width_px",
        "height_px",
        "width_mm",
        "height_mm",
        "length_mm",
        "short_side_mm",
        "rect_area_mm2",
        "contour_area_mm2",
        "pixels_per_mm",
    ]

    def __init__(self, file_path="data/measurements.csv"):
        self.file_path = Path(file_path)
        self._next_id = 1

    def save_record(
        self,
        obj: DetectedObject,
        measurement: Measurement,
        mode: str = "auto",
        object_id: int | None = None,
    ) -> dict:
        """
        计算物体实际尺寸，并追加保存一条记录。

        Args:
            obj: 检测到的物体。
            measurement: 测量模块。
            mode: 测量模式，"auto" 自动检测，"manual" 手动框选。
            object_id: 物体编号；不传时自动递增分配。

        说明：rect_area_mm2 是旋转矩形面积（长宽乘积），
        contour_area_mm2 是物体轮廓面积换算，二者并不相同。
        """
        if mode not in ("auto", "manual"):
            raise ValueError("mode 必须是 'auto' 或 'manual'")

        if object_id is None:
            object_id = self._next_id

        if object_id < 1:
            raise ValueError("object_id 必须大于或等于 1")

        self._next_id = max(self._next_id, object_id) + 1

        result = measurement.measure(
            width_px=obj.width_px,
            height_px=obj.height_px,
        )

        pixels_per_mm = measurement.calibration.pixels_per_mm
        pixels_per_mm2 = pixels_per_mm**2

        record = {
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "object_id": object_id,
            "mode": mode,
            "width_px": round(obj.width_px, 1),
            "height_px": round(obj.height_px, 1),
            "width_mm": round(result.width_mm, 2),
            "height_mm": round(result.height_mm, 2),
            "length_mm": round(result.length_mm, 2),
            "short_side_mm": round(result.short_side_mm, 2),
            "rect_area_mm2": round(
                obj.width_px * obj.height_px / pixels_per_mm2,
                1,
            ),
            "contour_area_mm2": round(
                obj.area_px / pixels_per_mm2,
                1,
            ),
            "pixels_per_mm": round(pixels_per_mm, 4),
        }

        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_exists = (
            self.file_path.exists()
            and self.file_path.stat().st_size > 0
        )

        with self.file_path.open(
            "a",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELDNAMES,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(record)

        return record


def main():
    """独立测试记录保存功能。"""

    import numpy as np

    from src.calibration import Calibration

    calibration = Calibration(pixels_per_mm=5.0)
    measurement = Measurement(calibration)

    obj = DetectedObject(
        center=(250.0, 155.0),
        width_px=300.0,
        height_px=150.0,
        angle=10.0,
        area_px=44700.0,
        box_points=np.array(
            [[100, 80], [400, 80], [400, 230], [100, 230]],
            dtype=float,
        ),
    )

    manager = RecordManager()
    record = manager.save_record(
        object_id=1,
        obj=obj,
        measurement=measurement,
    )

    print("测量记录保存成功：")
    for key, value in record.items():
        print(f"{key}: {value}")

    print(f"\nCSV 文件位置：{manager.file_path.resolve()}")


if __name__ == "__main__":
    main()
