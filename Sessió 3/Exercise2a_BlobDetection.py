# -*- coding: utf-8 -*-
"""练习二 a：在正面和侧面车牌上运行 scikit-image 的 LoG blob detector。

运行：
    python Exercise2a_BlobDetection.py

可调整参数：
    python Exercise2a_BlobDetection.py --min-sigma 1 --max-sigma 5 --threshold 0.03

脚本只弹出结果图，不会保存或修改数据文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.feature import blob_log


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "cropped_real_plates"
sys.path.insert(0, str(ROOT / "license_plate"))
from LicensePlateCharacterSegmentation import detectCharacterCandidates  # noqa: E402


def get_rectified_plate(view: str, image_id: str) -> np.ndarray:
    """按标注坐标读取并校正车牌，返回 OpenCV BGR 图像。"""
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
    plate, _, _ = detectCharacterCandidates(
        image_bgr, data["regionsImCropped"][index], SHOW=0
    )
    return plate


def main() -> None:
    parser = argparse.ArgumentParser(description="练习二 a：LoG 斑点检测")
    parser.add_argument("--min-sigma", type=float, default=1.0,
                        help="检测斑点的最小尺度，默认 1.0")
    parser.add_argument("--max-sigma", type=float, default=5.0,
                        help="检测斑点的最大尺度，默认 5.0")
    parser.add_argument("--num-sigma", type=int, default=10,
                        help="在尺度范围内尝试的尺度数量，默认 10")
    parser.add_argument("--threshold", type=float, default=0.03,
                        help="斑点响应阈值，默认 0.03")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="斑点之间允许的重叠比例，默认 0.5")
    args = parser.parse_args()
    if args.min_sigma <= 0 or args.max_sigma <= args.min_sigma:
        parser.error("需要满足 0 < min-sigma < max-sigma")

    examples = (("Frontal", "0216KZP", "正面车牌"),
                ("Lateral", "0182GLK", "侧面车牌"))
    display_images = []

    for row, (view, image_id, label) in enumerate(examples):
        plate_bgr = get_rectified_plate(view, image_id)
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # 字符比车牌底色暗；取反后让暗字符成为亮斑，交给 blob_log 检测。
        detection_image = 1.0 - gray
        blobs = blob_log(
            detection_image,
            min_sigma=args.min_sigma,
            max_sigma=args.max_sigma,
            num_sigma=args.num_sigma,
            threshold=args.threshold,
            overlap=args.overlap,
        )
        print(f"{label} {image_id}：检测到 {len(blobs)} 个斑点")
        print("返回值每一行依次是：行坐标 y、列坐标 x、尺度 sigma")

        original = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        overlay = original.copy()
        for y, x, sigma in blobs:
            radius = max(1, int(round(np.sqrt(2) * sigma)))
            center = (int(round(x)), int(round(y)))
            cv2.circle(overlay, center, radius, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(overlay, center, (0, 255, 255),
                           cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)
        cv2.putText(original, f"{label}: original", (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"{label}: LoG blobs ({len(blobs)})", (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
        display_images.append(cv2.hconcat([original, overlay]))

    # 正面和侧面分别显示；按任意键切换/关闭窗口。
    cv2.imshow("Exercise 2a - Frontal plate", display_images[0])
    cv2.imshow("Exercise 2a - Lateral plate", display_images[1])
    print("按任意键关闭结果窗口。红圈表示斑点的位置和估计大小。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
