
import argparse
from pathlib import Path

import cv2

from src.camera import Camera
from src.calibration import Calibration


def parse_args():
    parser = argparse.ArgumentParser(
        description="通过摄像头画面框选参考物，完成尺寸标定"
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output",
        default="config/calibration.yaml",
        help="标定文件保存路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)

    print("=" * 50)
    print("单目图像平面尺寸测量工具 - 交互式标定")
    print("操作说明：")
    print("  c：冻结当前画面并选择参考物")
    print("  q：退出标定")
    print("=" * 50)

    try:
        with Camera(device_index=args.index) as camera:
            while True:
                frame = camera.read()
                display = frame.copy()

                cv2.putText(
                    display,
                    "Press C to capture, Q to quit",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Calibration - Live Camera", display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    print("已取消标定。")
                    return

                if key == ord("c"):
                    captured = frame.copy()
                    break

        cv2.destroyAllWindows()

        # 在冻结的图像中框选参考物
        print("\n请在弹出的图像中框选参考物，然后按 Enter 确认。")
        print("按 Esc 或取消选择可放弃本次标定。")

        x, y, width, height = cv2.selectROI(
            "Select Reference Object",
            captured,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyAllWindows()

        if width <= 0 or height <= 0:
            print("未选择有效区域，标定已取消。")
            return

        print(f"选择区域：x={x}, y={y}, 宽={width} px，高={height} px")
        print("\n请选择你已知真实长度的那条边：")
        print("  1：框选区域的宽度")
        print("  2：框选区域的高度")

        choice = input("请输入 1 或 2：").strip()

        if choice == "1":
            pixel_length = float(width)
            edge_name = "宽度"
        elif choice == "2":
            pixel_length = float(height)
            edge_name = "高度"
        else:
            print("无效选择，标定已取消。")
            return

        real_length_mm = float(
            input(f"请输入参考物{edge_name}的真实长度（mm）：")
        )

        calibration = Calibration()
        ratio = calibration.calibrate(
            reference_pixel_length=pixel_length,
            reference_real_length_mm=real_length_mm,
            reference={
                "real_length_mm": real_length_mm,
                "pixel_length": pixel_length,
                "edge": edge_name,
            },
        )

        # 记录标定时的画面分辨率，测量时校验分辨率是否一致。
        calibration.resolution = (
            captured.shape[1],
            captured.shape[0],
        )

        print(f"\n参考边像素长度：{pixel_length:.2f} px")
        print(f"参考边真实长度：{real_length_mm:.2f} mm")
        print(f"计算标定比例：{ratio:.6f} px/mm")

        confirm = input(
            f"确认保存到 {output_path}？输入 y 保存，其他键取消："
        ).strip().lower()

        if confirm == "y":
            calibration.save(output_path)
            print("标定完成！现在可以运行 main.py 进行测量。")
        else:
            print("已取消保存，原有标定文件未被修改。")

    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f"标定失败：{error}")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
