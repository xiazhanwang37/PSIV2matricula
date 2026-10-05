# -*- coding: utf-8 -*-
"""练习一：用自适应阈值生成车牌字符候选区域。

在命令行中进入本文件所在文件夹后运行：
    python Exercise1_ThresholdSegmentation.py

默认显示一张正面车牌。也可以指定正面/侧面以及车牌编号：
    python Exercise1_ThresholdSegmentation.py --view Lateral --id 0182GLK

脚本只显示图像，不会保存或修改任何数据文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as p
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "cropped_real_plates"

# 避免导入 license_plate/__init__.py；直接从模块文件夹导入函数。
sys.path.insert(0, str(ROOT / "license_plate"))
from LicensePlateCharacterSegmentation import detectCharacterCandidates  # noqa: E402


def load_plate(view: str, image_id: str) -> tuple[np.ndarray, np.ndarray]:
    """读取车牌图像和对应的 regionsImCropped 四角坐标。"""
    view_folder = DATA_ROOT / view
    regions_file = view_folder / "PlateRegions.npz"
    if not regions_file.exists():
        raise FileNotFoundError(f"找不到标注文件：{regions_file}")

    data = np.load(regions_file, allow_pickle=True)
    # 数据文件中字段名实际为 imID（小写 i），以及 regionsImCropped。
    if image_id not in data["imID"]:
        available = ", ".join(map(str, data["imID"][:8]))
        raise ValueError(f"{view} 中没有编号 {image_id}。部分可用编号：{available}")
    index = list(data["imID"]).index(image_id)
    corners = data["regionsImCropped"][index]

    matches = list(view_folder.glob(f"{image_id}*_MLPlate0.png"))
    if not matches:
        raise FileNotFoundError(f"找不到车牌图片：{view_folder / (image_id + '*_MLPlate0.png')}")

    # PIL 可正确读取包含重音符号的文件路径；之后转为 OpenCV 的 BGR 顺序。
    with Image.open(matches[0]) as pil_image:
        image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, corners


def show_experiment(view: str, image_id: str) -> None:
    image, corners = load_plate(view, image_id)

    # 原函数返回：校正后的车牌、形态学处理后的阈值图、字符候选 mask。
    plate, threshold_after_morph, candidates = detectCharacterCandidates(
        image, corners, SHOW=0
    )

    # 复现原函数中的自适应阈值步骤，以展示形态学处理之前的结果。
    value = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)[:, :, 2]
    threshold_before_morph = cv2.adaptiveThreshold(
        value,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        15,
    )

    print(f"视角：{view}；车牌编号：{image_id}")
    print("阈值参数：Gaussian 自适应阈值，反向二值化，blockSize=51，C=15。")
    print("后处理筛选：排除碰边区域；面积 > 300；高度为车牌高度的 50% 到 90%。")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"练习一：{view} 车牌 {image_id} 的字符候选区域", fontsize=14)

    panels = [
        (cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), "1. 透视校正后的车牌", None),
        (threshold_before_morph, "2. 自适应阈值（形态学前）", "gray"),
        (threshold_after_morph, "3. 形态学处理后", "gray"),
        (candidates, "4. 筛选后的字符候选 mask", "gray"),
    ]
    for axis, (pixels, title, cmap) in zip(axes, panels):
        axis.imshow(pixels, cmap=cmap)
        axis.set_title(title, fontsize=10)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Session 4 练习一：车牌字符候选分割")
    parser.add_argument("--view", choices=("Frontal", "Lateral"), default="Frontal",
                        help="车牌视角，默认 Frontal")
    parser.add_argument("--id", dest="image_id", default=None,
                        help="车牌编号；默认正面 0216KZP 或侧面 0182GLK")
    args = parser.parse_args()
    image_id = args.image_id or ("0216KZP" if args.view == "Frontal" else "0182GLK")
    show_experiment(args.view, image_id)


if __name__ == "__main__":
    main()
