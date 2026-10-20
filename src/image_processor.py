
import cv2
import numpy as np


class ImageProcessor:
    """
    图像预处理模块：灰度转换、去噪和前景提取。

    threshold_mode 支持三种方式：
    - "otsu"：Otsu 全局自适应阈值（默认）。
    - "adaptive"：局部自适应阈值，适合光照不均的场景。
    - "canny"：Canny 边缘检测 + 形态学闭运算连接断裂边缘。

    process() 会把中间结果缓存到 last_gray / last_edges，
    供调试视图（灰度图、边缘图）显示。
    """

    VALID_MODES = ("otsu", "adaptive", "canny")

    def __init__(
        self,
        blur_kernel_size: int = 5,
        threshold_mode: str = "otsu",
        invert: bool = False,
        canny_low: int = 50,
        canny_high: int = 150,
        morph_kernel_size: int = 5,
        morph_iterations: int = 2,
    ):
        if (
            blur_kernel_size <= 0
            or blur_kernel_size % 2 == 0
        ):
            raise ValueError(
                "blur_kernel_size 必须是正奇数，例如 3、5、7"
            )

        if threshold_mode not in self.VALID_MODES:
            raise ValueError(
                f"threshold_mode 必须是 {self.VALID_MODES} 之一"
            )

        if canny_low < 0 or canny_high < canny_low:
            raise ValueError(
                "Canny 阈值需满足 0 <= canny_low <= canny_high"
            )

        if (
            morph_kernel_size <= 0
            or morph_kernel_size % 2 == 0
        ):
            raise ValueError(
                "morph_kernel_size 必须是正奇数，例如 3、5、7"
            )

        if morph_iterations < 0:
            raise ValueError("morph_iterations 不能小于 0")

        self.blur_kernel_size = blur_kernel_size
        self.threshold_mode = threshold_mode
        self.invert = invert
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations

        # 调试视图缓存的中间结果。
        self.last_gray = None
        self.last_edges = None

    def to_gray(self, image: np.ndarray) -> np.ndarray:
        """将 BGR 彩色图像或灰度图像转换为灰度图。"""
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if image.ndim == 2:
            return image.copy()

        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        raise ValueError("不支持的图像格式")

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """使用高斯模糊降低图像噪声。"""
        if gray is None or gray.size == 0:
            raise ValueError("输入灰度图像为空")

        if gray.ndim != 2:
            raise ValueError("denoise 需要二维灰度图像")

        return cv2.GaussianBlur(
            gray,
            (self.blur_kernel_size, self.blur_kernel_size),
            0,
        )

    def detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Canny 边缘检测 + 形态学闭运算连接断裂边缘。"""
        edges = cv2.Canny(
            gray,
            self.canny_low,
            self.canny_high,
        )

        kernel = np.ones(
            (self.morph_kernel_size, self.morph_kernel_size),
            np.uint8,
        )

        return cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.morph_iterations,
        )

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """将灰度图像转换为二值图像（仅阈值模式）。"""
        if gray is None or gray.size == 0:
            raise ValueError("输入灰度图像为空")

        if gray.ndim != 2:
            raise ValueError("binarize 需要二维灰度图像")

        if self.threshold_mode == "canny":
            raise ValueError(
                "canny 模式请使用 detect_edges()"
            )

        if self.threshold_mode == "otsu":
            mode = (
                cv2.THRESH_BINARY_INV
                if self.invert
                else cv2.THRESH_BINARY
            )

            _, binary = cv2.threshold(
                gray,
                0,
                255,
                mode + cv2.THRESH_OTSU,
            )
            return binary

        adaptive_mode = (
            cv2.THRESH_BINARY_INV
            if self.invert
            else cv2.THRESH_BINARY
        )

        block_size = 11
        if min(gray.shape[:2]) <= block_size:
            raise ValueError(
                "自适应阈值要求图像尺寸大于 11x11"
            )

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            adaptive_mode,
            block_size,
            2,
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        完整预处理流程，返回前景图像。

        canny 模式返回边缘图（invert 不生效），
        其他模式返回二值图。
        """
        gray = self.to_gray(image)
        blurred = self.denoise(gray)

        # 缓存灰度图，供调试视图显示。
        self.last_gray = gray

        if self.threshold_mode == "canny":
            edges = self.detect_edges(blurred)
            self.last_edges = edges
            return edges

        self.last_edges = None
        return self.binarize(blurred)


def main():
    """使用合成图像测试预处理流程。"""
    image = np.full((400, 600, 3), 230, dtype=np.uint8)

    # 模拟深色矩形物体。
    cv2.rectangle(
        image,
        (150, 100),
        (450, 300),
        (30, 30, 30),
        -1,
    )

    for mode in ImageProcessor.VALID_MODES:
        processor = ImageProcessor(threshold_mode=mode)
        result = processor.process(image)
        print(f"模式 {mode}: 输出形状 {result.shape}")

    processor = ImageProcessor()
    binary = processor.process(image)

    cv2.imshow("Original Image", image)
    cv2.imshow("Binary Image", binary)
    cv2.imshow("Gray Image", processor.last_gray)

    print("图像预处理测试已运行。")
    print("按 q 或 Esc 退出。")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
