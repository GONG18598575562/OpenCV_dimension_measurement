
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectedObject:
    """检测到的物体信息，基于最小外接旋转矩形，尺寸单位为像素。"""

    center: tuple[float, float]
    width_px: float
    height_px: float
    angle: float
    area_px: float
    box_points: np.ndarray

    @property
    def long_side_px(self) -> float:
        """返回旋转矩形的较长边。"""
        return max(self.width_px, self.height_px)

    @property
    def short_side_px(self) -> float:
        """返回旋转矩形的较短边。"""
        return min(self.width_px, self.height_px)


class ContourDetector:
    """从二值图像中检测物体轮廓。"""

    def __init__(
        self,
        min_area: float = 1000.0,
        max_area: float | None = None,
    ):
        if min_area < 0:
            raise ValueError("min_area 不能小于 0")

        if max_area is not None and max_area <= min_area:
            raise ValueError("max_area 必须大于 min_area")

        self.min_area = min_area
        self.max_area = max_area

    def detect(
        self,
        binary_image: np.ndarray,
    ) -> list[DetectedObject]:
        """
        检测二值图像中的轮廓。

        白色区域通常作为待检测前景。
        使用最小外接旋转矩形（minAreaRect）描述物体，
        物体倾斜时也能得到真实的长宽。
        返回按轮廓面积从大到小排列的物体列表。
        """
        if binary_image is None or binary_image.size == 0:
            raise ValueError("输入二值图像为空")

        if binary_image.ndim != 2:
            raise ValueError("输入必须是单通道二值图像")

        # findContours 会修改输入数据，因此复制一份。
        contour_input = binary_image.copy()

        contours, _ = cv2.findContours(
            contour_input,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        objects = []

        for contour in contours:
            area = float(cv2.contourArea(contour))

            if area < self.min_area:
                continue

            if (
                self.max_area is not None
                and area > self.max_area
            ):
                continue

            (center_x, center_y), (width, height), angle = (
                cv2.minAreaRect(contour)
            )

            if width <= 0 or height <= 0:
                continue

            # 角度规范化到 [-45, 45)，宽高语义保持稳定，便于帧间平滑。
            while angle >= 45:
                angle -= 90
                width, height = height, width

            while angle < -45:
                angle += 90
                width, height = height, width

            box_points = cv2.boxPoints(
                (
                    (center_x, center_y),
                    (width, height),
                    angle,
                )
            )

            objects.append(
                DetectedObject(
                    center=(
                        float(center_x),
                        float(center_y),
                    ),
                    width_px=float(width),
                    height_px=float(height),
                    angle=float(angle),
                    area_px=area,
                    box_points=box_points,
                )
            )

        objects.sort(
            key=lambda item: item.area_px,
            reverse=True,
        )

        return objects


def main():
    """使用合成图像测试轮廓检测。"""

    # 黑色背景。
    binary = np.zeros((400, 600), dtype=np.uint8)

    # 添加两个白色矩形，其中一个倾斜，模拟两个待测物体。
    cv2.rectangle(
        binary,
        (50, 80),
        (250, 230),
        255,
        -1,
    )

    tilted = np.array(
        [[380, 130], [530, 170], [500, 290], [350, 250]],
        np.int32,
    )
    cv2.fillPoly(binary, [tilted], 255)

    # 添加一个小噪声，应该被面积阈值过滤。
    cv2.rectangle(
        binary,
        (20, 20),
        (25, 25),
        255,
        -1,
    )

    detector = ContourDetector(min_area=500)
    objects = detector.detect(binary)

    print(f"检测到 {len(objects)} 个物体")

    for index, obj in enumerate(objects, start=1):
        print(
            f"物体 {index}: "
            f"中心=({obj.center[0]:.1f}, {obj.center[1]:.1f}), "
            f"宽={obj.width_px:.1f} px, "
            f"高={obj.height_px:.1f} px, "
            f"角度={obj.angle:.1f}°, "
            f"面积={obj.area_px:.1f} px²"
        )

    # 在图像上绘制最小外接旋转矩形。
    display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for index, obj in enumerate(objects, start=1):
        cv2.drawContours(
            display,
            [obj.box_points.astype(np.int32)],
            -1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f"Object {index}",
            (
                int(obj.center[0] - 30),
                int(obj.center[1]),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Binary Image", binary)
    cv2.imshow("Contour Detection", display)

    print("按 q 或 Esc 退出。")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
