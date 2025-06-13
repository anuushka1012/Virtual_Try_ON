# utils/image_utils.py
# image processing

import cv2
import numpy as np
from PIL import Image

def load_and_resize(image_path, size=(256, 192)):
    """Load and resize image using OpenCV"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    return img

def to_tensor(img):
    """Convert OpenCV image (BGR) to normalized PyTorch tensor"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    return transform(img)

def save_image(img_tensor, path):
    """Save PyTorch tensor image to disk"""
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = ((img + 1) / 2.0 * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
