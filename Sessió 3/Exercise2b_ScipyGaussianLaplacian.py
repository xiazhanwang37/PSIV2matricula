# -*- coding: utf-8 -*-
"""练习二 b：用 SciPy gaussian_laplace 生成字符候选区域。

运行：
    python Exercise2b_ScipyGaussianLaplacian.py

脚本会读取正面和侧面示例车牌，用练习一的字符候选尺寸估算纵向/横向
sigma，并显示原图、LoG 响应、阈值区域和后处理候选框。只显示，不保存。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "cropped_real_plates"
sys.path.insert(0, str(ROOT / "license_plate"))
from LicensePlateCharacterSegmentation import detectCharacterCandidates  # noqa: E402


def read_and_segment(view: str, image_id: str):
    """读取、透视校正车牌，并运行练习一取得字符候选 mask。"""
    folder = DATA_ROOT / view
    data = np.load(folder / "PlateRegions.npz", allow_pickle=True)
    ids = list(data["imID"])
    if image_id not in ids:
        raise ValueError(f"在 {view} 数据中找不到车牌编号 {image_id}")
    index = ids.index(image_id)
    image_files = list(folder.glob(f"{image_id}*_MLPlate0.png"))
    if not image_files:
        raise FileNotFoundError(f"找不到 {view} 车牌图片 {image_id}")

    with Image.open(image_files[0]) as pil_image:
        image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    plate, _, candidates = detectCharacterCandidates(
        image_bgr, data["regionsImCropped"][index], SHOW=0
    )
    return plate, candidates


def estimate_sigmas(candidate_mask: np.ndarray) -> tuple[float, float, float, float]:
    """从练习一候选框尺寸估算 LoG 的纵向和横向 sigma。"""
    plate_h, plate_w = candidate_mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(candidate_mask)
    heights, widths = [], []
    for x, y, w, h, area in stats[1:count]:
        if area > 0 and 0.45 * plate_h <= h <= 0.95 * plate_h:
            heights.append(h)
            widths.append(w)

    if heights:
        char_h = float(np.median(heights))
        char_w = float(np.median(widths))
        # 高斯尺度取字符框尺寸的一部分，避免把整行车牌过度平滑。
        sigma_y = max(1.0, char_h / 10.0)
        sigma_x = max(0.8, char_w / 10.0)
    else:
        char_h = 0.65 * plate_h
        char_w = 0.12 * plate_h
        sigma_y = max(1.0, char_h / 10.0)
        sigma_x = max(0.8, char_w / 10.0)
    return char_h, char_w, sigma_y, sigma_x


def find_candidates(plate: np.ndarray, candidate_mask: np.ndarray,
                    percentile: float):
    """计算各向异性 LoG 响应并将高响应像素整理成候选区域。"""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # 暗字符取反成亮结构；sigma 顺序是 (行方向 y, 列方向 x)。
    bright_characters = 1.0 - gray
    char_h, char_w, sigma_y, sigma_x = estimate_sigmas(candidate_mask)
    response = -ndi.gaussian_laplace(
        bright_characters, sigma=(sigma_y, sigma_x)
    )
    response = np.maximum(response, 0)

    positive = response[response > 0]
    threshold = float(np.percentile(positive, percentile)) if positive.size else 0.0
    binary = np.uint8(response >= threshold) * 255

    # 把邻近的 LoG 响应连起来，再按练习一估算的字符高度筛选。
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    grouped = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, connect_kernel, iterations=1)
    grouped = cv2.dilate(grouped, connect_kernel, iterations=1)

    # LoG peaks often mark separate strokes, not an entire glyph. Use the
    # character-sized regions found in Exercise 1 as proposals, and retain a
    # proposal only when the LoG threshold image has a response inside it.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask)
    candidates = np.zeros_like(candidate_mask)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (area > 0 and 0.45 * char_h <= h <= 1.25 * char_h
                and w <= 0.30 * candidate_mask.shape[1]):
            response_region = binary[y:y + h, x:x + w]
            # A single strong LoG pixel is enough to keep this character-sized proposal.
            if np.any(response_region):
                candidates[labels == i] = 255
                boxes.append((x, y, w, h))
    return response, binary, candidates, boxes, (char_h, char_w, sigma_y, sigma_x)


def make_display(plate: np.ndarray, response: np.ndarray, binary: np.ndarray,
                 boxes: list[tuple[int, int, int, int]], label: str) -> np.ndarray:
    """拼接各阶段结果；在原图上用红框标出字符候选。"""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    response_view = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    original_view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = original_view.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 1)

    panels = [
        (original_view, "Original"),
        (cv2.cvtColor(response_view, cv2.COLOR_GRAY2BGR), "LoG response"),
        (cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "Threshold"),
        (overlay, "Character candidates"),
    ]
    titled = []
    for image, title in panels:
        cv2.putText(image, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 255), 1, cv2.LINE_AA)
        titled.append(image)
    result = cv2.hconcat(titled)
    cv2.putText(result, label, (4, result.shape[0] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 128, 0), 1, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="练习二 b：SciPy Gaussian-Laplacian")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="LoG 响应阈值百分位，默认 95；调低会保留更多响应")
    args = parser.parse_args()
    if not 50 <= args.percentile < 100:
        parser.error("percentile 应在 50 到 100 之间")

    examples = (("Frontal", "0216KZP", "Frontal 0216KZP"),
                ("Lateral", "0182GLK", "Lateral 0182GLK"))
    displays = []
    for view, image_id, label in examples:
        plate, exercise1_mask = read_and_segment(view, image_id)
        response, binary, candidates, boxes, sizes = find_candidates(
            plate, exercise1_mask, args.percentile
        )
        char_h, char_w, sigma_y, sigma_x = sizes
        print(f"{label}: 练习一候选字符框中位数约为 高={char_h:.1f}px、宽={char_w:.1f}px")
        print(f"  gaussian_laplace sigma=(y={sigma_y:.2f}, x={sigma_x:.2f}); 后处理保留 {len(boxes)} 个候选框")
        displays.append(make_display(plate, response, binary, boxes, label))

    cv2.imshow("Exercise 2b - Frontal", displays[0])
    cv2.imshow("Exercise 2b - Lateral", displays[1])
    print("按任意键关闭结果窗口。红框是练习一提供的字符大小候选区，且框内至少有一个强 LoG 响应。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
