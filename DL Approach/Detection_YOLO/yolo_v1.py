# import the necessary packages
import numpy as np
import cv2
import glob
import os
from imutils import perspective
from matplotlib import pyplot as plt

from ultralytics import YOLO

#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
DataDir=r'C:\Users\Usuario\UAB\4t\MAPSIV\Sessió 2\LicensePlateDetector\real_plates\real_plates'
Views=['Frontal','Lateral']



# Load YOLO model
model = YOLO("yolov8n.pt") # Trained on COCO DataSet
modelclasses=np.array(list(model.names.values()))
model.device # By default model is in GPU device: model=model.to('cpu') for execution in CPU


#### COMPUTE PROPERTIES FOR EACH VIEW
yoloConf={}
yoloObj={}
for View in Views:
    
    ImageFiles=sorted(glob.glob(os.path.join(DataDir,View,'*.jpg')))
    yoloConf[View]=[]
    yoloObj[View]=[]

    # loop over the images
    for imagePath in ImageFiles:
        # load the image
        image = cv2.imread(imagePath)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(img_rgb) 
        obj=results[0].boxes.cls.cpu().numpy().astype(int)
        yoloObj[View].append(obj)
        yoloConf[View].append(results[0].boxes.conf.cpu().numpy())
        # Show results
        # results[0].show()

####  EXPLORE OBJECT DISTRIBUTION FOR EACH VIEW USING HISTOGRAMS AND BOXPLOTS
# Histograms for each view
for View in Views:
    # Combine detections from all images in this view
    confidences = np.concatenate(yoloConf[View]) if yoloConf[View] else np.array([])
    classes = np.concatenate(yoloObj[View]) if yoloObj[View] else np.array([], dtype=int)

    # Confidence score distribution
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, edgecolor='black')
    plt.xlabel('Confidence score')
    plt.ylabel('Number of detections')
    plt.title(f'YOLO confidence scores - {View}')
    plt.xlim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # Number of detections per object class
    if len(classes) > 0:
        class_ids, counts = np.unique(classes, return_counts=True)
        class_names = [model.names[int(class_id)] for class_id in class_ids]

        plt.figure(figsize=(10, 5))
        plt.bar(class_names, counts, edgecolor='black')
        plt.xlabel('Object class')
        plt.ylabel('Number of detections')
        plt.title(f'YOLO detections per class - {View}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print(f'No objects detected for {View}')