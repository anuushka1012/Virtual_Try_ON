# pose_estimation/yolo_pose.py

# Saves an annotated image with skeleton overlay.

# Returns a list of (x, y, confidence) keypoints for warping stage.

# Works with GPU (cuda) or CPU (cpu).

from ultralytics import YOLO
import cv2
import os

class YoloPoseEstimator:
    def __init__(self, model_path='yolov11-pose.pt', device='cpu'):
        self.model = YOLO(model_path)
        self.device = device

    def estimate_pose(self, image_path):
        """
        Runs pose estimation on the given image and returns keypoints.

        Returns:
            - image with keypoints drawn
            - list of keypoints (x, y, confidence)
        """
        results = self.model(image_path, device=self.device)
        result = results[0]

        # Extract keypoints
        keypoints = result.keypoints.xy[0].cpu().numpy()  # (num_keypoints, 2)
        scores = result.keypoints.conf[0].cpu().numpy()   # confidence scores

        annotated_img = result.plot()
        return annotated_img, list(zip(keypoints[:, 0], keypoints[:, 1], scores))

    def save_annotated(self, image, output_path):
        cv2.imwrite(output_path, image)


