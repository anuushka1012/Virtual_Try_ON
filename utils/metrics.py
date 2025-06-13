# utils/metrics.py

# SSIM and FID

import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models import inception_v3
from skimage.metrics import structural_similarity as ssim
import cv2
from scipy.linalg import sqrtm

def compute_ssim(image1_path, image2_path):
    """Compute SSIM between two images (same size)"""
    img1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
    img1 = cv2.resize(img1, (256, 192))
    img2 = cv2.resize(img2, (256, 192))
    ssim_value = ssim(img1, img2, multichannel=True)
    return ssim_value

def compute_fid(img1_tensor, img2_tensor, device='cpu'):
    """Compute Fréchet Inception Distance between two image tensors"""
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # remove classification head
    model.eval()

    def get_activations(img_tensor):
        img = T.Resize((299, 299))(img_tensor)  # Resize to Inception input
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            act = model(img)[0].cpu().numpy()
        return act

    act1 = get_activations(img1_tensor)
    act2 = get_activations(img2_tensor)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid
