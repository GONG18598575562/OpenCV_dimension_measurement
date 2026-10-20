
import cv2
import numpy as np

from src.calibration import Calibration
from src.contour_detector import DetectedObject


class Visualizer:
    """在图像上绘制最小外接旋转框、中心点和尺寸标注。"""

    def __init__(
        self,
        box_color=(0, 255, 0),
        text_color=(0, 0, 255),
        center_color=(255, 0, 0),
        thickness=2,
        font_scale=0.55,
    ):
        if thickness <= 0:
            raise ValueError("thickness 必须大于 0")

        if font_scale <= 0:
            raise ValueError("font_scale 必须大于 0")

        self.box_color = box_color
        self.text_color = text_color
        self.center_color = center_color
        self.thickness = thickness
        self.font_scale = font_scale

    def draw(
        self,
        image: np.ndarray,
        objects: list[DetectedObject],
        calibration: Calibration | None = None,
    ) -> np.ndarray:
        """
        绘制检测结果。

        Args:
            image: 输入的 BGR 彩色图像。
            objects: 轮廓检测模块返回的物体列表。
            calibration: 可选的像素比例标定对象。

        Returns:
            绘制后的图像副本，不修改输入图像。
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("输入必须是 BGR 彩色图像")

        if not isinstance(objects, (list, tuple)):
            raise ValueError("objects 必须是物体列表")

        if (
            calibration is not None
            and calibration.pixels_per_mm is not None
            and calibration.pixels_per_mm <= 0
        ):
            raise ValueError("像素比例必须大于 0")

        output = image.copy()
        image_height, image_width = output.shape[:2]

        for index, obj in enumerate(objects, start=1):
            box = obj.box_points.astype(np.int32)

            # 绘制最小外接旋转矩形。
            cv2.drawContours(
                output,
                [box],
                -1,
                self.box_color,
                self.thickness,
            )

            # 绘制中心点。
            center_x = int(round(obj.center[0]))
            center_y = int(round(obj.center[1]))
            cv2.circle(
                output,
                (center_x, center_y),
                4,
                self.center_color,
                -1,
            )

            # 默认显示像素尺寸。
            length_value = obj.long_side_px
            short_value = obj.short_side_px
            unit = "px"

            # 有有效标定比例时，显示毫米尺寸。
            if (
                calibration is not None
                and calibration.pixels_per_mm is not None
            ):
                length_value = calibration.pixel_to_mm(
                    length_value
                )
                short_value = calibration.pixel_to_mm(
                    short_value
                )
                unit = "mm"

            length_label = f"L: {length_value:.1f} {unit}"
            short_label = f"W: {short_value:.1f} {unit}"

            # 轮廓面积（区别于旋转矩形面积）。
            if (
                calibration is not None
                and calibration.pixels_per_mm is not None
            ):
                contour_area = obj.area_px / (
                    calibration.pixels_per_mm**2
                )
                area_label = f"A: {contour_area:.1f} mm2 (contour)"
            else:
                area_label = f"A: {obj.area_px:.0f} px2 (contour)"

            # 旋转框的几何边界，用于放置文字。
            box_x = int(box[:, 0].min())
            box_y = int(box[:, 1].min())

            label_y = box_y - 10

            # 如果框顶部空间不足，就将文字放在框内。
            if label_y < 20:
                label_y = box_y + 20

            text_x = min(
                max(box_x + 5, 5),
                max(image_width - 150, 5),
            )

            cv2.putText(
                output,
                f"Object {index}",
                (text_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.text_color,
                self.thickness,
                cv2.LINE_AA,
            )

            width_text_y = min(
                max(box_y + 20, label_y + 22),
                image_height - 30,
            )

            height_text_y = min(
                max(box_y + 42, label_y + 44),
                image_height - 8,
            )

            area_text_y = min(
                max(box_y + 64, label_y + 66),
                image_height - 8,
            )

            cv2.putText(
                output,
                length_label,
                (text_x, width_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.text_color,
                self.thickness,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                short_label,
                (text_x, height_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.text_color,
                self.thickness,
                cv2.LINE_AA,
            )

            cv2.putText(
                output,
                area_label,
                (text_x, area_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.text_color,
                self.thickness,
                cv2.LINE_AA,
            )

        return output


def main():
    """使用合成图像测试可视化模块。"""

    from src.contour_detector import ContourDetector
    from src.image_processor import ImageProcessor

    image = np.full((400, 600, 3), 230, dtype=np.uint8)

    # 倾斜的深色矩形物体。
    tilted = np.array(
        [[180, 120], [420, 160], [390, 300], [150, 260]],
        np.int32,
    )
    cv2.fillPoly(image, [tilted], (30, 30, 30))

    # 生成二值图像并检测轮廓。
    processor = ImageProcessor(invert=True)
    binary = processor.process(image)

    detector = ContourDetector(min_area=500)
    objects = detector.detect(binary)

    # 模拟标定结果：5 px/mm。
    calibration = Calibration(pixels_per_mm=5.0)

    visualizer = Visualizer()
    result = visualizer.draw(image, objects, calibration)

    cv2.imshow("Original Image", image)
    cv2.imshow("Visualization Test", result)

    print(f"检测到 {len(objects)} 个物体")
    print("按 q 或 Esc 退出。")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
