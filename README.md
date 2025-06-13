# 🧥 Virtual Try-On System using YOLOv11 & GAN Refinement

This project implements an efficient **virtual try-on pipeline** that lets users digitally try on clothes using pose estimation, garment warping, and GAN-based image refinement.

---

## 📌 Features

- ✅ Real-time pose estimation using **YOLOv11**
- ✅ Garment warping with **Spatial Transformer Network**
- ✅ High-fidelity output using **U-Net-style GAN refinement**
- ✅ Lightweight and modular — runs on **CPU or low-GPU devices**

---

## 🗂️ Folder Structure
virtual_tryon_project/
├── data/
│ ├── input_images/ # Person images
│ ├── garments/ # Garment images
│ ├── preprocessed/ # Intermediate outputs
│ └── output/ # Final try-on results
├── models/
│ ├── yolo_weights/ # yolov11-pose.pt
│ └── viton_hd_pretrained/ # HD-VITON pretrained GAN weights
├── pose_estimation/
├── segmentation/
├── warping/
├── refinement/
├── utils/
├── viewer.py
└── main.py


Install dependencies

pip install -r requirements.txt
Add pretrained weights

yolov11-pose.pt → /models/yolo_weights/

HD-VITON weights (G.pth, etc.) → /models/viton_hd_pretrained/

You can use placeholder files if you just want to test the flow.

