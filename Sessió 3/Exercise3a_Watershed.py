# -*- coding: utf-8 -*-
"""练习三 a：观察 Watershed 的中间处理结果。

运行：
    python Exercise3a_Watershed.py

脚本读取一张正面、一张侧面车牌，显示阈值、开运算、距离变换、
前景种子和分水岭边界。只显示结果，不保存或修改数据文件。
"""

from __future__ import annotations

import cv2
import numpy as np

from Exercise2b_ScipyGaussianLaplacian import read_and_segment


def panel(image: np.ndarray, title: str) -> np.ndarray:
    """将图像统一为 BGR 并加上面板标题。"""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image.copy()
    cv2.putText(image, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (0, 0, 255), 1, cv2.LINE_AA)
    return image


def watershed_stages(view: str, image_id: str, label: str) -> np.ndarray:
    plate, _ = read_and_segment(view, image_id)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # 1. 暗色字符成为白色前景。
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2. 开运算清理小白噪点。
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=2
    )

    # 3. 距离变换：白色前景内部越靠中心，距离值越大。
    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # 4. 取距离较大的内部区域作为确定前景种子。
    max_distance = float(distance.max())
    if max_distance > 0:
        _, sure_fg = cv2.threshold(
            distance, 0.5 * max_distance, 255, cv2.THRESH_BINARY
        )
    else:
        sure_fg = np.zeros_like(gray)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(opening, sure_fg)

    # 5. 连通前景种子，生成 watershed 标记。
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # 背景标为 1，前景种子从 2 开始。
    markers[unknown == 255] = 0  # 0 表示待分配的未知区域。

    # 6. Watershed 以标记为种子扩张；-1 表示区域边界。
    markers = cv2.watershed(plate.copy(), markers)
    boundary_view = plate.copy()
    boundary_view[markers == -1] = (0, 0, 255)

    distance_view = cv2.normalize(
        distance, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    row1 = cv2.hconcat([
        panel(plate, "1. Rectified plate"),
        panel(binary, "2. Otsu binary"),
        panel(opening, "3. Morphological opening"),
    ])
    row2 = cv2.hconcat([
        panel(distance_view, "4. Distance transform"),
        panel(sure_fg, "5. Sure foreground seeds"),
        panel(boundary_view, "6. Watershed boundaries"),
    ])
    display = cv2.vconcat([row1, row2])
    cv2.putText(display, label, (4, display.shape[0] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 128, 0), 1, cv2.LINE_AA)
    return display


def main() -> None:
    frontal = watershed_stages("Frontal", "0216KZP", "Frontal 0216KZP")
    lateral = watershed_stages("Lateral", "0182GLK", "Lateral 0182GLK")
    cv2.imshow("Exercise 3a - Frontal", frontal)
    cv2.imshow("Exercise 3a - Lateral", lateral)
    print("按任意键关闭窗口。红色线条是 Watershed 找到的区域边界。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
