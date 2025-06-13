import cv2
import numpy as np
import os

output_dir = "data/preprocessed"
os.makedirs(output_dir, exist_ok=True)

# 1. Blank Pose Image
pose = np.full((256, 192, 3), 255, dtype=np.uint8)
cv2.putText(pose, 'Pose Output', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
cv2.imwrite(f"{output_dir}/pose_annotated.jpg", pose)

# 2. Blank Person Mask
mask = np.zeros((256, 192), dtype=np.uint8)
cv2.rectangle(mask, (40, 40), (150, 220), 255, -1)
cv2.imwrite(f"{output_dir}/person_mask.png", mask)

# 3. Segmented Person Image
seg = np.full((256, 192, 3), 0, dtype=np.uint8)
cv2.putText(seg, 'Segmented', (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.imwrite(f"{output_dir}/person_segmented.png", seg)

# 4. Warped Garment Image
warp = np.full((256, 192, 3), 200, dtype=np.uint8)
cv2.putText(warp, 'Warped Garment', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
cv2.imwrite(f"{output_dir}/warped_garment.png", warp)

# 5. Final Try-On Output
tryon = np.full((256, 192, 3), 180, dtype=np.uint8)
cv2.putText(tryon, 'Final Try-On', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2)
cv2.imwrite(f"{output_dir}/final_tryon.png", tryon)

print("Sample placeholder images created in /data/preprocessed/")
