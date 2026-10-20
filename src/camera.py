
import cv2


class Camera:
    """摄像头读取模块，负责打开设备、读取图像和释放资源。"""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """
        Args:
            device_index: 摄像头设备索引，例如 0、1、2。
            width: 希望使用的图像宽度。
            height: 希望使用的图像高度。
            fps: 希望使用的帧率。
        """
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self) -> None:
        """打开摄像头；打开失败时抛出异常。"""
        if self.is_opened():
            return

        self.cap = cv2.VideoCapture(
            self.device_index,
            cv2.CAP_V4L2,
        )

        if not self.cap.isOpened():
            self.release()
            raise RuntimeError(
                f"无法打开摄像头，设备索引：{self.device_index}"
            )

        # 请求摄像头使用指定分辨率和帧率。
        # 摄像头不一定支持，因此后续要读取实际值。
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        actual_height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(
            f"摄像头已打开：索引 {self.device_index}，"
            f"实际分辨率 {actual_width}x{actual_height}，"
            f"帧率 {actual_fps:.1f} fps"
        )

    def read(self):
        """
        读取一帧图像。

        Returns:
            OpenCV 图像，类型为 NumPy 数组，颜色格式为 BGR。

        Raises:
            RuntimeError: 摄像头未打开或图像读取失败。
        """
        if not self.is_opened():
            raise RuntimeError("摄像头尚未打开，请先调用 open()")

        success, frame = self.cap.read()

        if not success or frame is None:
            raise RuntimeError("摄像头读取图像失败")

        return frame

    def is_opened(self) -> bool:
        """判断摄像头是否处于打开状态。"""
        return (
            self.cap is not None
            and self.cap.isOpened()
        )

    def release(self) -> None:
        """释放摄像头资源。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """支持 with Camera(...) as camera 的用法。"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """离开 with 代码块时自动释放资源。"""
        self.release()


def main():
    """独立测试摄像头模块。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="测试 Camera 摄像头读取模块"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="摄像头设备索引，默认 0",
    )
    args = parser.parse_args()

    try:
        with Camera(device_index=args.index) as camera:
            print("摄像头测试开始，按 q 退出。")

            while True:
                frame = camera.read()

                cv2.imshow("Camera Test", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except RuntimeError as error:
        print(f"摄像头错误：{error}")

    finally:
        cv2.destroyAllWindows()
        print("摄像头资源已释放，测试结束。")


if __name__ == "__main__":
    main()
