# segmentation/segmentation.py

# the person's mask (body/silhouette),

# and optionally garment regions (for future extension).

# This is essential to isolate the person from the background and guide proper garment placement during warping.

# person_mask.png: binary mask of person.

# person_segmented.png: person with background removed.

# This script performs person segmentation — it separates the person from the background in an image. The output is:

# A binary mask (white = person, black = background)
# person_segmented.png: The person shown, background removed
# A masked image that shows only the person (background removed)


import torch
import torchvision.transforms as T
import torchvision.models.segmentation as models
import cv2
import numpy as np
from PIL import Image
import os

class PersonSegmenter:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = models.deeplabv3_resnet50(pretrained=True).to(self.device).eval()

        self.transform = T.Compose([
            T.Resize((256, 192)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def segment_person(self, image_path):
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)


        # Passes the image through the model to get pixel-wise class scores.
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # Class 15 is 'person' in COCO dataset
        #         This line creates a binary mask where:

        # 255 = person

        # 0 = background
        person_mask = (output_predictions == 15).astype(np.uint8) * 255
        return np.array(img), person_mask

    def apply_mask(self, image, mask):
        """Applies the binary mask to the image"""
        masked = cv2.bitwise_and(image, image, mask=mask)
        return masked

    # It applies the mask. Only the pixels where the mask = 255 are kept — everything else is made black.
    def save_outputs(self, image, mask, out_image_path, out_mask_path):
        cv2.imwrite(out_image_path, image)
        cv2.imwrite(out_mask_path, mask)


# Loads a pretrained DeepLabV3 model with a ResNet-50 backbone.

# This model is trained on the COCO dataset to detect 21 different classes.

# Puts the model in evaluation mode.

# Q2. Why did you use DeepLabV3?
# A: It’s a state-of-the-art semantic segmentation model. It provides pixel-level accuracy and is pretrained on COCO, which includes the “person” class.