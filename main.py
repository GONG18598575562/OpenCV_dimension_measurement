
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.calibration import Calibration
from src.contour_detector import ContourDetector, DetectedObject
from src.image_processor import ImageProcessor
from src.measurement import Measurement
from src.record_manager import RecordManager
from src.tracker import ObjectSmoother
from src.visualization import Visualizer

CAPTURE_DIR = Path("data/captures")

# 调试视图循环顺序：关闭 -> 灰度图 -> 边缘/二值图。
VIEW_SEQUENCE = ("off", "gray", "edge")

SAME_PLANE_NOTICE_CN = (
    "被测物体必须与标定参考物处于同一平面，"
    "否则测量结果可能存在误差。"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="单目图像平面尺寸测量工具"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="摄像头索引，默认 0",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="config/calibration.yaml",
        help="标定文件路径",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="请求的图像宽度，默认 640",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="请求的图像高度，默认 480",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="请求的摄像头帧率，默认 30",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1000.0,
        help="最小轮廓面积，单位 px²",
    )
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="otsu",
        choices=ImageProcessor.VALID_MODES,
        help="前景提取方式：otsu / adaptive / canny",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=50,
        help="Canny 低阈值，默认 50",
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=150,
        help="Canny 高阈值，默认 150",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="反转二值化结果，适用于深色背景上的浅色物体",
    )
    return parser.parse_args()


def load_calibration(path: Path):
    """
    加载标定文件。

    文件不存在或损坏时不中断程序，返回 None
    并以"未标定模式"运行（只显示像素尺寸）。
    """
    if not path.exists():
        print(f"找不到标定文件：{path}")
        print(
            "将以未标定模式运行（只显示像素尺寸），"
            "按 C 可随时进行标定。"
        )
        return None

    try:
        calibration = Calibration.load(path)
    except (ValueError, OSError, TypeError) as error:
        print(f"加载标定文件失败：{error}")
        print(
            "将以未标定模式运行（只显示像素尺寸），"
            "按 C 重新标定，按 X 删除损坏的标定文件。"
        )
        return None

    if calibration.pixels_per_mm is None or calibration.pixels_per_mm <= 0:
        print("标定比例无效，将以未标定模式运行，请按 C 重新标定。")
        return None

    print(
        "已加载标定："
        f"{calibration.pixels_per_mm:.4f} px/mm"
        f"（方法：{calibration.method or '未知'}，"
        f"时间：{calibration.calibrated_at or '未知'}）"
    )
    return calibration


def check_resolution(calibration: Calibration, frame) -> None:
    """校验当前画面分辨率是否与标定时一致。"""
    if calibration.resolution is None:
        print(
            "提示：标定文件未记录分辨率，无法校验，"
            "请重新标定以获得更可靠的结果。"
        )
        return

    actual = (frame.shape[1], frame.shape[0])

    if actual != calibration.resolution:
        print(
            "警告：当前分辨率 "
            f"{actual[0]}x{actual[1]} 与标定时 "
            f"{calibration.resolution[0]}x{calibration.resolution[1]}"
            " 不一致，测量结果不可信，请重新标定！"
        )


def save_screenshot(display) -> Path:
    """保存带标注的测量截图。"""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    path = CAPTURE_DIR / (
        f"capture_{datetime.now():%Y%m%d_%H%M%S}.png"
    )

    if not cv2.imwrite(str(path), display):
        raise RuntimeError(f"截图写入失败：{path}")

    return path


def run_calibration(frame, output_path: Path):
    """
    在冻结画面上框选参考物完成标定（应用内）。

    Returns:
        标定成功并确认保存时返回 Calibration，否则返回 None。
    """
    print("\n[标定] 请在弹出的画面中框选参考物，Enter 确认，Esc 取消。")

    roi = cv2.selectROI(
        "Calibration - select reference object",
        frame,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyAllWindows()

    x, y, width, height = roi

    if width <= 0 or height <= 0:
        print("[标定] 未选择有效区域，已取消。")
        return None

    print(f"[标定] 选区：x={x}, y={y}, 宽={width} px，高={height} px")

    try:
        choice = input(
            "[标定] 已知真实长度的边（1=宽度 2=高度）："
        ).strip()

        if choice == "1":
            pixel_length, edge = float(width), "宽度"
        elif choice == "2":
            pixel_length, edge = float(height), "高度"
        else:
            print("[标定] 无效选择，已取消。")
            return None

        real_length_mm = float(
            input(f"[标定] 参考物{edge}的真实长度（mm）：")
        )
    except ValueError:
        print("[标定] 输入无效，已取消。")
        return None

    calibration = Calibration()

    try:
        ratio = calibration.calibrate(
            reference_pixel_length=pixel_length,
            reference_real_length_mm=real_length_mm,
            reference={
                "real_length_mm": real_length_mm,
                "pixel_length": pixel_length,
                "edge": edge,
            },
        )
    except ValueError as error:
        print(f"[标定] 标定失败：{error}")
        return None

    # 记录标定时的画面分辨率。
    calibration.resolution = (frame.shape[1], frame.shape[0])

    print(f"[标定] 比例：{ratio:.6f} px/mm")
    print(
        f"[标定] 分辨率："
        f"{calibration.resolution[0]}x{calibration.resolution[1]}"
    )

    confirm = input(
        f"[标定] 保存到 {output_path}？输入 y 保存，其他键取消："
    ).strip().lower()

    if confirm != "y":
        print("[标定] 已取消保存，原标定文件未被修改。")
        return None

    try:
        calibration.save(output_path)
    except OSError as error:
        print(f"[标定] 保存失败：{error}")
        return None

    return calibration


def manual_measure(
    frame,
    processor: ImageProcessor,
    detector: ContourDetector,
    measurement: Measurement,
    visualizer: Visualizer,
    record_manager: RecordManager,
) -> None:
    """
    冻结当前画面，手动框选待测物体并测量。

    框选区域内检测不到轮廓时，直接把框选区域当作物体。
    测量结果以 manual 模式写入 CSV 记录。
    """
    roi = cv2.selectROI(
        "Manual - select object, Enter to confirm",
        frame,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyAllWindows()

    x, y, width, height = roi

    if width <= 0 or height <= 0:
        print("未选择有效区域，已取消手动测量。")
        return

    binary = processor.process(frame)
    roi_binary = binary[y : y + height, x : x + width]

    objects = detector.detect(roi_binary)

    if objects:
        obj = objects[0]
        offset = np.array([x, y], dtype=float)
        obj.center = (
            obj.center[0] + x,
            obj.center[1] + y,
        )
        obj.box_points = obj.box_points + offset
    else:
        obj = DetectedObject(
            center=(x + width / 2, y + height / 2),
            width_px=float(width),
            height_px=float(height),
            angle=0.0,
            area_px=float(width * height),
            box_points=np.array(
                [
                    [x, y],
                    [x + width, y],
                    [x + width, y + height],
                    [x, y + height],
                ],
                dtype=float,
            ),
        )

    result = measurement.measure(
        width_px=obj.width_px,
        height_px=obj.height_px,
    )

    display = visualizer.draw(
        frame.copy(),
        [obj],
        measurement.calibration,
    )

    cv2.putText(
        display,
        f"L: {result.length_mm:.2f} mm | "
        f"W: {result.short_side_mm:.2f} mm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Manual Measurement", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    try:
        record = record_manager.save_record(
            obj=obj,
            measurement=measurement,
            mode="manual",
        )
        print(
            f"手动测量已保存："
            f"{record['length_mm']:.2f} x "
            f"{record['short_side_mm']:.2f} mm，"
            f"记录文件：{record_manager.file_path.resolve()}"
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"保存测量记录失败：{error}")


def draw_status(
    display,
    calibration,
    auto_mode: bool,
    object_count: int,
) -> None:
    """在画面上绘制标定状态、测量模式与限制提示。"""
    if calibration is not None:
        status = (
            f"CAL: {calibration.pixels_per_mm:.4f} px/mm"
        )
        if calibration.resolution is not None:
            status += (
                f" @{calibration.resolution[0]}"
                f"x{calibration.resolution[1]}"
            )
        status_color = (0, 255, 0)
    else:
        status = "UNCALIBRATED - press C to calibrate (px only)"
        status_color = (0, 0, 255)

    cv2.putText(
        display,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
        cv2.LINE_AA,
    )

    mode_text = (
        f"MODE: {'AUTO' if auto_mode else 'MANUAL-ONLY'}"
        f"  Objects: {object_count}"
    )

    cv2.putText(
        display,
        mode_text,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 测量限制提示（OpenCV 字体不支持中文，故用英文）。
    cv2.putText(
        display,
        "NOTE: object must be on the SAME plane"
        " as the calibration reference",
        (10, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 200, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        "C: calibrate | M: manual | S: shot | D: record"
        " | A: auto | P: view | B: invert | X: clear | Q: quit",
        (10, display.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    calibration_path = Path(args.calibration)

    # 1. 加载标定结果；失败时进入未标定模式，不输出虚假尺寸。
    calibration = load_calibration(calibration_path)
    measurement = (
        Measurement(calibration) if calibration else None
    )

    # 2. 初始化各模块。
    processor = ImageProcessor(
        threshold_mode=args.threshold_mode,
        invert=args.invert,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
    )
    detector = ContourDetector(min_area=args.min_area)
    visualizer = Visualizer()
    record_manager = RecordManager()
    smoother = ObjectSmoother()

    auto_mode = True
    view_index = 0
    debug_window_open = False

    try:
        # 3. 打开摄像头。
        with Camera(
            device_index=args.index,
            width=args.width,
            height=args.height,
            fps=args.fps,
        ) as camera:
            first_frame = camera.read()

            if calibration is not None:
                check_resolution(calibration, first_frame)

            print("\n程序已启动。")
            print("操作说明：")
            print("  C / R：标定 / 重新标定（框选参考物）")
            print("  X：清除标定参数")
            print("  M：冻结画面，手动框选物体测量")
            print("  A：切换自动轮廓检测")
            print("  S：保存当前测量截图")
            print("  D：保存当前检测到的测量记录（CSV）")
            print("  P：调试视图循环（关闭/灰度图/边缘图）")
            print("  B：切换前景黑白（仅阈值模式）")
            print("  q / Esc：退出")
            print(f"\n重要提示：{SAME_PLANE_NOTICE_CN}")

            binary_window_open = False

            while True:
                frame = camera.read()

                # 4. 图像预处理、轮廓检测与帧间平滑。
                binary = processor.process(frame)

                if auto_mode:
                    objects = smoother.update(
                        detector.detect(binary)
                    )
                else:
                    objects = []

                # 5. 绘制检测框、尺寸标注和状态栏。
                display = visualizer.draw(
                    frame,
                    objects,
                    calibration,
                )
                draw_status(
                    display,
                    calibration,
                    auto_mode,
                    len(objects),
                )

                cv2.imshow("Dimension Measurement", display)

                # 6. 调试视图（灰度图 / 边缘图）。
                if view_index > 0:
                    view_name = VIEW_SEQUENCE[view_index]

                    debug_image = (
                        processor.last_gray
                        if view_name == "gray"
                        else (
                            processor.last_edges
                            if processor.last_edges is not None
                            else binary
                        )
                    )

                    cv2.imshow(
                        f"Debug View - {view_name}",
                        debug_image,
                    )
                    debug_window_open = True
                elif debug_window_open:
                    cv2.destroyWindow(
                        f"Debug View - {VIEW_SEQUENCE[1]}"
                    )
                    cv2.destroyWindow(
                        f"Debug View - {VIEW_SEQUENCE[2]}"
                    )
                    debug_window_open = False

                key = cv2.waitKey(1) & 0xFF

                # q 或 Esc：退出。
                if key == ord("q") or key == 27:
                    break

                # C 或 R：标定 / 重新标定。
                elif key in (ord("c"), ord("r")):
                    new_calibration = run_calibration(
                        frame,
                        calibration_path,
                    )

                    if new_calibration is not None:
                        calibration = new_calibration
                        measurement = Measurement(calibration)

                # X：清除标定参数。
                elif key == ord("x"):
                    Calibration.clear(calibration_path)
                    calibration = None
                    measurement = None
                    print(
                        "标定已清除，进入未标定模式"
                        "（只显示像素尺寸）。"
                    )

                # A：切换自动轮廓检测。
                elif key == ord("a"):
                    auto_mode = not auto_mode
                    print(
                        "自动检测："
                        + ("开启" if auto_mode else "关闭")
                    )

                # M：手动框选测量（需要有效标定）。
                elif key == ord("m"):
                    if measurement is None:
                        print(
                            "尚未完成标定，无法测量真实尺寸。"
                            "请先按 C 标定。"
                        )
                        continue

                    manual_measure(
                        frame,
                        processor,
                        detector,
                        measurement,
                        visualizer,
                        record_manager,
                    )

                # S：保存当前测量截图。
                elif key == ord("s"):
                    try:
                        path = save_screenshot(display)
                        print(f"截图已保存：{path.resolve()}")
                    except (OSError, RuntimeError) as error:
                        print(f"保存截图失败：{error}")

                # D：保存当前检测到的测量记录。
                elif key == ord("d"):
                    if measurement is None:
                        print(
                            "尚未完成标定，"
                            "不会记录看似有效的真实尺寸。"
                            "请先按 C 标定。"
                        )
                        continue

                    if not objects:
                        print("当前没有检测到可保存的物体。")
                        continue

                    try:
                        for obj in objects:
                            record = record_manager.save_record(
                                obj=obj,
                                measurement=measurement,
                                mode="auto",
                            )
                            print(
                                f"已保存物体 {record['object_id']}："
                                f"{record['length_mm']:.2f} x "
                                f"{record['short_side_mm']:.2f} mm"
                            )

                        print(
                            "记录文件："
                            f"{record_manager.file_path.resolve()}"
                        )
                    except (
                        OSError,
                        ValueError,
                        RuntimeError,
                    ) as error:
                        print(f"保存测量记录失败：{error}")

                # P：调试视图循环。
                elif key == ord("p"):
                    view_index = (view_index + 1) % len(
                        VIEW_SEQUENCE
                    )

                # B：切换前景黑白模式（仅阈值模式有效）。
                elif key == ord("b"):
                    if args.threshold_mode == "canny":
                        print("canny 模式下 invert 不生效。")
                    else:
                        processor.invert = not processor.invert
                        print(
                            "前景模式："
                            + (
                                "反转"
                                if processor.invert
                                else "正常"
                            )
                        )

    except (RuntimeError, cv2.error) as error:
        print(f"运行错误：{error}")

    finally:
        cv2.destroyAllWindows()
        print("程序已退出，窗口已关闭。")


if __name__ == "__main__":
    main()
