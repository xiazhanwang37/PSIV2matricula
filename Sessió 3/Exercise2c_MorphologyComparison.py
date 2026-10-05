# -*- coding: utf-8 -*-
"""练习二 c：比较形态学增强前后的 SciPy LoG 响应。

运行：
    python Exercise2c_MorphologyComparison.py

同一张正面/侧面车牌、使用相同 sigma 和响应百分位，比较：
原图 LoG vs. 黑帽变换增强后 LoG。只显示，不保存数据文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from Exercise2b_ScipyGaussianLaplacian import estimate_sigmas, read_and_segment


def make_panel(image: np.ndarray, title: str) -> np.ndarray:
    """将灰度图转换为 BGR 并加上短标题。"""
    if image.ndim == 2:
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        panel = image.copy()
    cv2.putText(panel, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (0, 0, 255), 1, cv2.LINE_AA)
    return panel


def response_and_threshold(image: np.ndarray, sigma: tuple[float, float],
                           percentile: float):
    """返回正向 LoG 响应和按响应百分位生成的二值图。"""
    response = -ndi.gaussian_laplace(image, sigma=sigma)
    response = np.maximum(response, 0)
    positive = response[response > 0]
    cutoff = float(np.percentile(positive, percentile)) if positive.size else 0.0
    binary = np.uint8(response >= cutoff) * 255
    return response, binary, cutoff


def process(view: str, image_id: str, label: str, percentile: float) -> None:
    plate, ex1_candidates = read_and_segment(view, image_id)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    char_h, char_w, sigma_y, sigma_x = estimate_sigmas(ex1_candidates)
    sigma = (sigma_y, sigma_x)

    # 原始灰度中字符较暗，取反后使字符成为亮结构。
    raw_input = 1.0 - gray

    # 黑帽=闭运算结果减原图，能突出亮背景上的暗色字符。
    gray8 = np.uint8(np.clip(gray * 255, 0, 255))
    kernel_width = max(15, int(round(plate.shape[1] * 0.06)))
    if kernel_width % 2 == 0:
        kernel_width += 1
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, 3)
    )
    blackhat = cv2.morphologyEx(gray8, cv2.MORPH_BLACKHAT, blackhat_kernel)
    enhanced_input = blackhat.astype(np.float32) / 255.0

    raw_response, raw_binary, raw_cutoff = response_and_threshold(
        raw_input, sigma, percentile
    )
    enhanced_response, enhanced_binary, enhanced_cutoff = response_and_threshold(
        enhanced_input, sigma, percentile
    )

    def normalize_response(response: np.ndarray) -> np.ndarray:
        return cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    row1 = cv2.hconcat([
        make_panel(gray8, "Original gray"),
        make_panel(blackhat, "Blackhat enhanced"),
        make_panel(normalize_response(raw_response), "Raw LoG response"),
    ])
    row2 = cv2.hconcat([
        make_panel(raw_binary, "Raw LoG threshold"),
        make_panel(normalize_response(enhanced_response), "Enhanced LoG response"),
        make_panel(enhanced_binary, "Enhanced LoG threshold"),
    ])
    display = cv2.vconcat([row1, row2])
    cv2.putText(display, f"{label} | sigma_y={sigma_y:.1f}, sigma_x={sigma_x:.1f}, "
                        f"character estimate={char_h:.0f}x{char_w:.0f}px",
                (4, display.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 128, 0), 1, cv2.LINE_AA)
    window = f"Exercise 2c - {label}"
    cv2.imshow(window, display)

    raw_count = cv2.countNonZero(raw_binary)
    enhanced_count = cv2.countNonZero(enhanced_binary)
    print(f"{label}: sigma_y={sigma_y:.2f}, sigma_x={sigma_x:.2f}")
    print(f"  原图 LoG: cutoff={raw_cutoff:.5f}, 保留 {raw_count} 个响应像素")
    print(f"  黑帽增强后 LoG: cutoff={enhanced_cutoff:.5f}, 保留 {enhanced_count} 个响应像素")


def main() -> None:
    parser = argparse.ArgumentParser(description="练习二 c：形态学增强前后对比")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="两种 LoG 响应使用相同百分位阈值，默认 95")
    args = parser.parse_args()
    if not 50 <= args.percentile < 100:
        parser.error("percentile 应在 50 到 100 之间")

    process("Frontal", "0216KZP", "Frontal 0216KZP", args.percentile)
    process("Lateral", "0182GLK", "Lateral 0182GLK", args.percentile)
    print("按任意键关闭对照窗口。重点比较两张 LoG threshold 图中的字符响应和背景误检。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
