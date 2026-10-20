"""ContourDetector 模块单元测试（使用合成二值图像）。"""

import unittest

import cv2
import numpy as np

from src.contour_detector import ContourDetector, DetectedObject


def make_rotated_rect_image(
    size=(600, 600),
    center=(300, 300),
    wh=(300, 120),
    angle=30.0,
    background=0,
    foreground=255,
):
    image = np.full(size, background, np.uint8)
    box = cv2.boxPoints(((center), (wh), angle))
    cv2.fillPoly(image, [box.astype(np.int32)], foreground)
    return image


class TestDetect(unittest.TestCase):
    def test_recover_rotated_rect_dimensions(self):
        image = make_rotated_rect_image(wh=(300, 120), angle=30.0)
        objects = ContourDetector(min_area=500).detect(image)

        self.assertEqual(len(objects), 1)
        obj = objects[0]
        sides = sorted([obj.width_px, obj.height_px], reverse=True)
        self.assertAlmostEqual(sides[0], 300, delta=3)
        self.assertAlmostEqual(sides[1], 120, delta=3)

    def test_angle_normalized(self):
        image = make_rotated_rect_image(wh=(300, 120), angle=80.0)
        obj = ContourDetector(0).detect(image)[0]
        self.assertGreaterEqual(obj.angle, -45)
        self.assertLess(obj.angle, 45)

    def test_min_area_filter(self):
        image = np.zeros((200, 200), np.uint8)
        cv2.rectangle(image, (10, 10), (20, 20), 255, -1)
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)

        objects = ContourDetector(min_area=500).detect(image)
        self.assertEqual(len(objects), 1)

    def test_sorted_by_area_desc(self):
        image = np.zeros((400, 600), np.uint8)
        cv2.rectangle(image, (10, 10), (110, 110), 255, -1)
        cv2.rectangle(image, (200, 50), (550, 350), 255, -1)

        objects = ContourDetector(min_area=100).detect(image)
        self.assertEqual(len(objects), 2)
        self.assertGreaterEqual(
            objects[0].area_px, objects[1].area_px
        )

    def test_max_area_filter(self):
        image = make_rotated_rect_image(wh=(300, 120))
        detector = ContourDetector(min_area=100, max_area=2000)
        self.assertEqual(len(detector.detect(image)), 0)

    def test_box_points_shape(self):
        image = make_rotated_rect_image()
        obj = ContourDetector(0).detect(image)[0]
        self.assertEqual(obj.box_points.shape, (4, 2))

    def test_long_short_side_properties(self):
        obj = DetectedObject(
            center=(0.0, 0.0),
            width_px=50.0,
            height_px=120.0,
            angle=0.0,
            area_px=6000.0,
            box_points=np.zeros((4, 2)),
        )
        self.assertEqual(obj.long_side_px, 120.0)
        self.assertEqual(obj.short_side_px, 50.0)

    def test_invalid_inputs(self):
        detector = ContourDetector()
        with self.assertRaises(ValueError):
            detector.detect(None)
        with self.assertRaises(ValueError):
            detector.detect(np.zeros((10, 10, 3), np.uint8))

    def test_invalid_min_area(self):
        with self.assertRaises(ValueError):
            ContourDetector(min_area=-1)

    def test_max_area_not_greater_than_min(self):
        with self.assertRaises(ValueError):
            ContourDetector(min_area=100, max_area=50)


class TestCannyPipeline(unittest.TestCase):
    def test_canny_edges_are_detectable(self):
        """Canny 模式生成的前景图同样能检测出物体。"""
        from src.image_processor import ImageProcessor

        color = np.full((600, 600, 3), 230, np.uint8)
        cv2.rectangle(color, (150, 200), (450, 400), (40, 40, 40), -1)

        processor = ImageProcessor(threshold_mode="canny")
        edges = processor.process(color)

        self.assertGreater(int(edges.max()), 0)
        self.assertIsNotNone(processor.last_gray)

        objects = ContourDetector(min_area=500).detect(edges)
        self.assertEqual(len(objects), 1)
        obj = objects[0]
        self.assertAlmostEqual(obj.width_px, 300, delta=6)
        self.assertAlmostEqual(obj.height_px, 200, delta=6)


if __name__ == "__main__":
    unittest.main()
