
import math

import cv2

from src.contour_detector import DetectedObject


class ObjectSmoother:
    """
    帧间目标匹配与指数平滑（EMA）。

    手持测量时，逐帧独立检测会导致框位置和尺寸数字抖动。
    该模块将当前帧的检测结果与上一帧的目标按中心距离匹配，
    对匹配上的目标做指数平滑，未匹配的目标直接保留。
    """

    def __init__(
        self,
        alpha: float = 0.4,
        match_radius: float = 80.0,
        max_missed: int = 5,
    ):
        """
        Args:
            alpha: 平滑系数，越小越平滑，取值 (0, 1]。
            match_radius: 帧间中心匹配的最大距离，单位 px。
            max_missed: 目标连续丢失多少帧后删除。
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha 必须在 (0, 1] 范围内")

        if match_radius <= 0:
            raise ValueError("match_radius 必须大于 0")

        if max_missed < 0:
            raise ValueError("max_missed 不能小于 0")

        self.alpha = alpha
        self.match_radius = match_radius
        self.max_missed = max_missed

        # 内部跟踪列表：[DetectedObject, 连续未匹配帧数]
        self._tracks: list[list] = []

    def update(
        self,
        objects: list[DetectedObject],
    ) -> list[DetectedObject]:
        """传入当前帧检测结果，返回平滑后的结果。"""
        matched_indices: set[int] = set()
        smoothed: list[DetectedObject] = []

        for obj in objects:
            best_index = None
            best_distance = self.match_radius

            for index, (track, _) in enumerate(self._tracks):
                if index in matched_indices:
                    continue

                distance = math.dist(obj.center, track.center)

                if distance <= best_distance:
                    best_index = index
                    best_distance = distance

            if best_index is None:
                # 新目标，直接保留并建立跟踪。
                smoothed.append(obj)
                self._tracks.append([obj, 0])
            else:
                matched_indices.add(best_index)
                merged = self._merge(
                    self._tracks[best_index][0],
                    obj,
                )
                self._tracks[best_index] = [merged, 0]
                smoothed.append(merged)

        # 未匹配的跟踪增加丢失计数，删除过期跟踪。
        for index, track in enumerate(self._tracks):
            if index not in matched_indices:
                track[1] += 1

        self._tracks = [
            track
            for track in self._tracks
            if track[1] <= self.max_missed
        ]

        return smoothed

    def _merge(
        self,
        track: DetectedObject,
        obj: DetectedObject,
    ) -> DetectedObject:
        """对匹配上的目标做指数平滑。"""
        a = self.alpha

        def lerp(old, new):
            return old + (new - old) * a

        center_x = lerp(track.center[0], obj.center[0])
        center_y = lerp(track.center[1], obj.center[1])
        width = lerp(track.width_px, obj.width_px)
        height = lerp(track.height_px, obj.height_px)
        area = lerp(track.area_px, obj.area_px)

        # 角度按 180° 周期取最短差值，避免 -45°/45° 边界跳变。
        angle_delta = (
            (obj.angle - track.angle + 90) % 180
        ) - 90
        angle = track.angle + angle_delta * a

        box_points = cv2.boxPoints(
            (
                (center_x, center_y),
                (width, height),
                angle,
            )
        )

        return DetectedObject(
            center=(center_x, center_y),
            width_px=width,
            height_px=height,
            angle=angle,
            area_px=area,
            box_points=box_points,
        )
