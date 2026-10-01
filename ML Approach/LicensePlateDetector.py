# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:45:11 2025

@author: debora
"""
# import the necessary packages
#from collections import namedtuple
# from skimage.filters import threshold_local
# from skimage import segmentation
# from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import random
import shutil

SHOW=1
minPlateW=100
minPlateH=30

def detectPlates(image):
        originalHeight, originalWidth = image.shape[:2]

        # if the width is greater than 640 pixels, then resize the image
        if image.shape[1] > 640:
            image = imutils.resize(image, width=640)
        processedHeight, processedWidth = image.shape[:2]
        scaleX = originalWidth / processedWidth
        scaleY = originalHeight / processedHeight
            
        # initialize the rectangular and square kernels to be applied to the image,
        # then initialize the list of license plate regions

        
        # Structuring Element first rectangular shape (width 15 hight 5)  second square shape
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  
        # list of potential regions with a license plate
        regions = []
        

        # convert the image to grayscale, and apply the blackhat operation to emphasize narrow regions with dark gray level
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, squareKernel,iterations =3) #rectKernel
        
        if (SHOW):
            plt.figure()
            plt.imshow(blackhat,cmap='gray')
            plt.title("Black Top Hat ")

        # numbers have vertical changes in gradient
        gradX = cv2.Sobel(blackhat,ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        if (SHOW):
            plt.figure()
            plt.imshow(gradX,cmap='gray')
            plt.title("Gradient X")
        
        # gaussian blur with a 5 x 5 kernel to smooth detail and noise
        gradX = cv2.GaussianBlur(gradX, (7, 7), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel,iterations = 2)
        if (SHOW):
            plt.figure()
            plt.imshow(gradX,cmap='gray')
            plt.title("Gausian Gx")
            
        
        # el valor de corte se fija como el 40% del màximo
        ThrValue= (0.40)*np.max(gradX)
        ThrGradX = cv2.threshold(gradX, ThrValue, 255, cv2.THRESH_BINARY)[1]
        if (SHOW):
            plt.figure()
            plt.imshow(ThrGradX,cmap='gray')
            plt.title("Threshold Gx")
        

        # some morphological operation to join parts first opening to remove small spots second to grow the area of license plate
        thresh = cv2.morphologyEx(ThrGradX, cv2.MORPH_OPEN, squareKernel,iterations = 4 )
        thresh = cv2.dilate(thresh, rectKernel, iterations=2)
        if(SHOW):
            plt.figure()
            plt.imshow(thresh,cmap='gray')
            plt.title("Possible license plates")
                  
        # find contours in the thresholded image
        (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # grab the bounding box associated with the contour and compute the area and
            # aspect ratio
            (x,y,w, h) = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspectRatio = w / float(h)
            if (SHOW):
                print("BLOB ANALYSIS ->",x,y,w,h,aspectRatio,area)
            
            # condition of not touching the border of the image
            NotouchBorder = (
                x != 0 and y != 0
                and x + w != processedWidth
                and y + h != processedHeight
            )

            
                
            # dimension conditions of a license plate
            keepArea = area > 3400 and area < 8000
            keepWidth = w > minPlateW and w <= 250
            keepHeight = h > minPlateH and h <= 60
            keepAspectRatio = 2.5<w/h<7

            # ensure the aspect ratio, width, and height of the bounding box fall within
            # tolerable limits, then update the list of license plate regions
            if all((NotouchBorder,keepAspectRatio,keepWidth,keepHeight,keepArea)):
                # compute the rotated bounding box of the region: 
                
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)

                # Convert points from the processed image back to original-image coordinates.
                box[:, 0] *= scaleX
                box[:, 1] *= scaleY
                regions.append(box)
                if (SHOW):
                    print("REGION BOX ACCEPTED->",box)
        return regions


if __name__ == "__main__":
        # Build a YOLO dataset from ML detector pseudo-labels.
        SHOW = 0  # Disable per-image debug plots during batch processing.
        projectDir = Path(__file__).resolve().parent.parent
        sourceDir = projectDir / "real_plates" / "real_plates"
        outputDir = projectDir / "DL Approach" / "Detection_YOLO" / "plate_dataset"
        views = ["Frontal", "Lateral"]
        rng = random.Random(42)

        imageLists = {
            view: sorted((sourceDir / view).glob("*.jpg"))
            for view in views
        }
        if any(not imageLists[view] for view in views):
            raise FileNotFoundError(f"Expected JPG images under {sourceDir / '<Frontal or Lateral>'}")

        assignments = {split: [] for split in ("train", "val", "test")}
        for view, paths in imageLists.items():
            paths = paths.copy()
            rng.shuffle(paths)
            trainEnd = int(0.70 * len(paths))
            valEnd = trainEnd + int(0.15 * len(paths))
            assignments["train"].extend((view, path) for path in paths[:trainEnd])
            assignments["val"].extend((view, path) for path in paths[trainEnd:valEnd])
            assignments["test"].extend((view, path) for path in paths[valEnd:])

        counts = {split: 0 for split in assignments}
        for split, items in assignments.items():
            for view, imagePath in items:
                imageBytes = np.fromfile(str(imagePath), dtype=np.uint8)
                image = cv2.imdecode(imageBytes, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Skipping unreadable image: {imagePath}")
                    continue

                height, width = image.shape[:2]
                regions = detectPlates(image)
                labelLines = []
                for box in regions:
                    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
                    x1 = float(np.clip(points[:, 0].min(), 0, width))
                    y1 = float(np.clip(points[:, 1].min(), 0, height))
                    x2 = float(np.clip(points[:, 0].max(), 0, width))
                    y2 = float(np.clip(points[:, 1].max(), 0, height))
                    boxWidth, boxHeight = x2 - x1, y2 - y1
                    if boxWidth <= 0 or boxHeight <= 0:
                        continue
                    xCenter = ((x1 + x2) / 2) / width
                    yCenter = ((y1 + y2) / 2) / height
                    labelLines.append(
                        f"0 {xCenter:.6f} {yCenter:.6f} "
                        f"{boxWidth / width:.6f} {boxHeight / height:.6f}"
                    )

                imageOut = outputDir / "images" / split / view / imagePath.name
                labelOut = outputDir / "labels" / split / view / f"{imagePath.stem}.txt"
                imageOut.parent.mkdir(parents=True, exist_ok=True)
                labelOut.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(imagePath, imageOut)
                labelOut.write_text("\n".join(labelLines), encoding="utf-8")
                counts[split] += 1

        yamlText = (
            f"path: '{outputDir.as_posix()}'\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "nc: 1\n"
            "names: [license_plate]\n"
        )
        (outputDir / "data.yaml").write_text(yamlText, encoding="utf-8")
        print(f"YOLO pseudo-label dataset created at: {outputDir}")
        print(f"Images per split: {counts}")
        print("Labels are ML-generated pseudo-labels; images with no detections have empty label files.")


    
