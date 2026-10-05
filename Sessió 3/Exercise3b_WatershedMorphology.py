# -*- coding: utf-8 -*-
"""练习三 b：比较黑帽形态学增强前后的 Watershed。

运行：
    python Exercise3b_WatershedMorphology.py

每个窗口上排为原图直接 Otsu + Watershed，下排为黑帽增强后再 Otsu +
Watershed。脚本只显示，不保存或修改车牌数据。
"""

from __future__ import annotations

import cv2
import numpy as np

from Exercise2b_ScipyGaussianLaplacian import read_and_segment


def panel(image: np.ndarray, title: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image.copy()
    cv2.putText(image, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (0, 0, 255), 1, cv2.LINE_AA)
    return image


def get_watershed(binary: np.ndarray, color_image: np.ndarray):
    """从前景二值图建立种子并运行 watershed，返回开运算、种子、边界图。"""
    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if distance.max() > 0:
        _, sure_fg = cv2.threshold(
            distance, 0.35 * float(distance.max()), 255, cv2.THRESH_BINARY
        )
    else:
        sure_fg = np.zeros_like(binary)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(opening, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    result_markers = cv2.watershed(color_image.copy(), markers)
    boundaries = color_image.copy()
    boundaries[result_markers == -1] = (0, 0, 255)
    return opening, sure_fg, boundaries


def process(view: str, image_id: str, label: str) -> np.ndarray:
    plate, _ = read_and_segment(view, image_id)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # A. 基线：原始灰度图直接使用 Otsu 反向阈值。
    _, raw_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, _, raw_boundaries = get_watershed(raw_binary, plate)

    # B. 黑帽变换增强亮背景上的暗字符，再对黑帽响应做普通 Otsu 阈值。
    kernel_width = max(15, int(round(plate.shape[1] * 0.06)))
    if kernel_width % 2 == 0:
        kernel_width += 1
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, 3)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
    blackhat = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4)).apply(blackhat)
    _, enhanced_binary = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, _, enhanced_boundaries = get_watershed(enhanced_binary, plate)

    first_row = cv2.hconcat([
        panel(plate, "Original plate"),
        panel(raw_binary, "Raw Otsu binary"),
        panel(raw_boundaries, "Raw watershed boundaries"),
    ])
    second_row = cv2.hconcat([
        panel(blackhat, "Blackhat + CLAHE"),
        panel(enhanced_binary, "Enhanced Otsu binary"),
        panel(enhanced_boundaries, "Enhanced watershed boundaries"),
    ])
    display = cv2.vconcat([first_row, second_row])
    cv2.putText(display, label, (4, display.shape[0] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 128, 0), 1, cv2.LINE_AA)
    return display


def main() -> None:
    frontal = process("Frontal", "0216KZP", "Frontal 0216KZP")
    lateral = process("Lateral", "0182GLK", "Lateral 0182GLK")
    cv2.imshow("Exercise 3b - Frontal", frontal)
    cv2.imshow("Exercise 3b - Lateral", lateral)
    print("按任意键关闭窗口。比较上排原始 Otsu 与下排黑帽增强后的结果。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
