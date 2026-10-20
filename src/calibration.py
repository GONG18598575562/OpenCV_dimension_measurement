
import argparse
import math
from datetime import datetime
from pathlib import Path

import yaml


class Calibration:
    """像素与实际尺寸之间的比例标定模块。"""

    CONFIG_VERSION = 1

    def __init__(
        self,
        pixels_per_mm=None,
        resolution=None,
        method=None,
        reference=None,
        calibrated_at=None,
    ):
        if pixels_per_mm is not None:
            pixels_per_mm = float(pixels_per_mm)
            if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
                raise ValueError("像素比例必须是有限的正数")

        self.pixels_per_mm = pixels_per_mm

        # 标定时的图像分辨率 (宽, 高)。像素比例只在同一分辨率下有效。
        if resolution is not None:
            width, height = resolution
            width, height = int(width), int(height)
            if width <= 0 or height <= 0:
                raise ValueError("分辨率必须是正数")

            resolution = (width, height)

        self.resolution = resolution

        # 标定方法描述，例如 "reference_object"。
        self.method = (
            str(method) if method is not None else None
        )

        # 参考物信息：真实长度、像素长度、参考边。
        if reference is not None:
            if not isinstance(reference, dict):
                raise ValueError("reference 必须是字典")

            reference = dict(reference)

        self.reference = reference

        # 标定时间（ISO 8601 字符串）。
        self.calibrated_at = (
            str(calibrated_at)
            if calibrated_at is not None
            else None
        )

    def calibrate(
        self,
        reference_pixel_length: float,
        reference_real_length_mm: float,
        reference: dict | None = None,
    ) -> float:
        """根据参考物的像素长度和真实长度计算 px/mm。"""
        pixel_length = float(reference_pixel_length)
        real_length = float(reference_real_length_mm)

        if not math.isfinite(pixel_length) or pixel_length <= 0:
            raise ValueError("参考物像素长度必须是有限的正数")

        if not math.isfinite(real_length) or real_length <= 0:
            raise ValueError("参考物真实长度必须是有限的正数")

        if reference is not None:
            if not isinstance(reference, dict):
                raise ValueError("reference 必须是字典")

            self.reference = dict(reference)
        else:
            self.reference = {
                "pixel_length": pixel_length,
                "real_length_mm": real_length,
            }

        self.calibrated_at = (
            datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
        )
        self.method = "reference_object"

        self.pixels_per_mm = pixel_length / real_length
        return self.pixels_per_mm

    def pixel_to_mm(self, pixel_length: float) -> float:
        """将像素长度转换为毫米。"""
        if self.pixels_per_mm is None:
            raise RuntimeError("尚未完成标定")

        pixel_length = float(pixel_length)
        if not math.isfinite(pixel_length) or pixel_length < 0:
            raise ValueError("像素长度必须是有限的非负数")

        return pixel_length / self.pixels_per_mm

    def mm_to_pixel(self, length_mm: float) -> float:
        """将毫米长度转换为像素。"""
        if self.pixels_per_mm is None:
            raise RuntimeError("尚未完成标定")

        length_mm = float(length_mm)
        if not math.isfinite(length_mm) or length_mm < 0:
            raise ValueError("实际长度必须是有限的非负数")

        return length_mm * self.pixels_per_mm

    def save(self, file_path="config/calibration.yaml") -> None:
        """保存标定结果到 YAML 文件。"""
        if self.pixels_per_mm is None:
            raise RuntimeError("尚未完成标定，无法保存")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self.CONFIG_VERSION,
            "method": self.method or "reference_object",
            "pixels_per_mm": float(self.pixels_per_mm),
            "unit": "mm",
        }

        if self.reference is not None:
            data["reference"] = self.reference

        if self.resolution is not None:
            data["resolution"] = [
                self.resolution[0],
                self.resolution[1],
            ]

        if self.calibrated_at is not None:
            data["calibrated_at"] = self.calibrated_at

        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                data,
                file,
                allow_unicode=True,
                sort_keys=False,
            )

        print(f"标定结果已保存至：{path}")
        print(f"标定比例：{self.pixels_per_mm:.6f} px/mm")

    @staticmethod
    def clear(file_path="config/calibration.yaml") -> bool:
        """删除标定文件；返回是否实际删除了文件。"""
        path = Path(file_path)

        if path.exists():
            path.unlink()
            print(f"标定文件已清除：{path}")
            return True

        print(f"标定文件不存在，无需清除：{path}")
        return False

    @classmethod
    def load(cls, file_path="config/calibration.yaml"):
        """从 YAML 文件加载标定结果。"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"标定文件不存在：{path}")

        with path.open("r", encoding="utf-8") as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as error:
                raise ValueError(
                    f"标定文件解析失败（文件可能损坏）：{error}"
                )

        if not isinstance(data, dict):
            raise ValueError("标定文件格式不正确")

        if "pixels_per_mm" not in data:
            raise ValueError("标定文件缺少 pixels_per_mm")

        resolution = data.get("resolution")

        if resolution is not None:
            try:
                resolution = (int(resolution[0]), int(resolution[1]))
            except (TypeError, ValueError, IndexError):
                raise ValueError("标定文件 resolution 字段格式不正确")

        reference = data.get("reference")

        if reference is not None:
            if not isinstance(reference, dict):
                raise ValueError(
                    "标定文件 reference 字段格式不正确"
                )

            reference = dict(reference)

        return cls(
            pixels_per_mm=data["pixels_per_mm"],
            resolution=resolution,
            method=data.get("method"),
            reference=reference,
            calibrated_at=data.get("calibrated_at"),
        )


def main():
    parser = argparse.ArgumentParser(
        description="单目图像平面尺寸测量工具——标定程序"
    )
    parser.add_argument(
        "--pixel-length",
        type=float,
        help="参考物在图像中的测量长度，单位 px",
    )
    parser.add_argument(
        "--real-length-mm",
        type=float,
        help="参考物的实际长度，单位 mm",
    )
    parser.add_argument(
        "--output",
        default="config/calibration.yaml",
        help="标定结果保存路径",
    )
    args = parser.parse_args()

    # 没有提供参数时，交互式输入，避免自动写入示例比例。
    if args.pixel_length is None and args.real_length_mm is None:
        try:
            args.pixel_length = float(
                input("请输入参考物在图像中的长度（px）：")
            )
            args.real_length_mm = float(
                input("请输入参考物的真实长度（mm）：")
            )
        except ValueError:
            parser.error("请输入有效的数字")

    elif args.pixel_length is None or args.real_length_mm is None:
        parser.error(
            "--pixel-length 和 --real-length-mm 必须同时提供"
        )

    try:
        calibration = Calibration()
        ratio = calibration.calibrate(
            args.pixel_length,
            args.real_length_mm,
        )
        print(f"计算结果：{ratio:.6f} px/mm")
        calibration.save(args.output)

    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
