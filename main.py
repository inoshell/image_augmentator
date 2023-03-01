import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as F
from yolov5.utils.datasets import LoadImagesAndLabels
from yolov5.utils.general import xywh2xyxy, xyxy2xywh


# Define augmentation techniques
def apply_augmentation(img, labels):
    # Convert image to PIL image
    pil_img = Image.fromarray(img)

    # Randomly apply augmentations
    pil_img = F.adjust_brightness(pil_img, brightness_factor=np.random.uniform(0.5, 1.5))
    pil_img = F.adjust_contrast(pil_img, contrast_factor=np.random.uniform(0.5, 1.5))
    pil_img = F.adjust_saturation(pil_img, saturation_factor=np.random.uniform(0.5, 1.5))
    pil_img = F.adjust_hue(pil_img, hue_factor=np.random.uniform(-0.1, 0.1))
    pil_img = F.rotate(pil_img, angle=np.random.uniform(-10, 10))
    pil_img = F.gaussian_blur(pil_img, kernel_size=np.random.randint(1, 5))

    # Convert PIL image back to numpy array
    img = np.array(pil_img)

    # Apply same transformations to labels
    for i, label in enumerate(labels):
        bbox = label[1:]
        bbox = xywh2xyxy(bbox)
        bbox = F.adjust_brightness(pil_img, brightness_factor=np.random.uniform(0.5, 1.5))
        bbox = F.adjust_contrast(pil_img, contrast_factor=np.random.uniform(0.5, 1.5))
        bbox = F.adjust_saturation(pil_img, saturation_factor=np.random.uniform(0.5, 1.5))
        bbox = F.adjust_hue(pil_img, hue_factor=np.random.uniform(-0.1, 0.1))
        bbox = F.rotate(pil_img, angle=np.random.uniform(-10, 10))
        bbox = F.gaussian_blur(pil_img, kernel_size=np.random.randint(1, 5))
        bbox = xyxy2xywh(bbox)
        labels[i][1:] = bbox

    return img, labels


# Define paths
img_folder = "path/to/image/folder"
anno_folder = "path/to/annotation/folder"

# Load images and annotations
dataset = LoadImagesAndLabels(img_folder, anno_folder)

# Loop through images and annotations
for img, targets, path, _ in dataset:
    # Apply augmentation
    img, targets = apply_augmentation(img, targets)

    # Save augmented image
    cv2.imwrite(path, img)

    # Save re-annotated labels
    with open(path.replace("jpg", "txt"), "w") as f:
        for target in targets:
            label = target[0]
            bbox = target[1:]
            bbox = xywh2xyxy(bbox)
            bbox = [str(coord) for coord in bbox]
            label_str = "{} {} {} {} {}\n".format(label, bbox[0], bbox[1], bbox[2], bbox[3])
            f.write(label_str)

