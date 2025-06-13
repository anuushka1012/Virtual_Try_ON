# viewer.py

import cv2
import os

PREPROCESSED_DIR = "data/preprocessed"

IMAGE_FILES = [
    "pose_annotated.jpg",
    "person_mask.png",
    "person_segmented.png",
    "warped_garment.png",
    "final_tryon.png"
]

def load_images(image_dir, image_files):
    images = []
    for filename in image_files:
        path = os.path.join(image_dir, filename)
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.resize(img, (256, 256))
            images.append(img)
        else:
            print(f"[Warning] Missing: {filename}")
    return images

def display_images(images, title="Preprocessed Output Viewer"):
    if len(images) == 0:
        print("No images to display.")
        return
    grid = cv2.hconcat(images)
    cv2.imshow(title, grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    images = load_images(PREPROCESSED_DIR, IMAGE_FILES)
    display_images(images)
