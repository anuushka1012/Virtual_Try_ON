# main.py

from pose_estimation.yolo_pose import YoloPoseEstimator
from segmentation.segmentation import PersonSegmenter
from warping.stn_warp import warp_garment
from refinement.gan_refine import refine_tryon

import os

def ensure_dirs():
    os.makedirs("data/preprocessed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)

def run_pipeline():
    ensure_dirs()

    # ---- CONFIG ----
    person_img = "data\clothes\Screenshot 2024-09-25 133926.png"
    garment_img = "data\model\1bc8fb28-ae0e-4491-8fbe-5cbd94479e38.jpg"
    yolo_model_path = "models/yolo_weights/yolov11-pose.pt"
    device = "cpu"  # or "cuda" if GPU is available

    # ---- STEP 1: Pose Estimation ----
    print("Running pose estimation...")
    pose_model = YoloPoseEstimator(model_path=yolo_model_path, device=device)
    annotated_img, keypoints = pose_model.estimate_pose(person_img)
    pose_model.save_annotated(annotated_img, "data/preprocessed/pose_annotated.jpg")

    # ---- STEP 2: Person Segmentation ----
    print("Running segmentation...")
    segmenter = PersonSegmenter(device=device)
    original_img, mask = segmenter.segment_person(person_img)
    masked_img = segmenter.apply_mask(original_img, mask)
    segmenter.save_outputs(masked_img, mask,
                           "data/preprocessed/person_segmented.png",
                           "data/preprocessed/person_mask.png")

    # ---- STEP 3: Garment Warping ----
    print("Warping garment...")
    warped_garment_path = warp_garment(garment_img)

    # ---- STEP 4: GAN-Based Refinement ----
    print("Refining final try-on image...")
    final_output_path = refine_tryon(person_img, warped_garment_path,
                                     output_path="data/output/final_tryon.png",
                                     device=device)

    print(f"Virtual try-on complete. Output saved at: {final_output_path}")

if __name__ == "__main__":
    run_pipeline()



# Calls each module in sequence:

# YOLOv11 for keypoints

# DeepLab for mask

# STN to warp the garment

# U-Net GAN for final refinement

# Saves intermediate images in /data/preprocessed/

# Saves the final image in /data/output/