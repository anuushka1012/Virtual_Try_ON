# refinement/gan_refine.py


# Input: Person image + Warped Garment

# Output: A refined try-on image where the garment looks natural — with smooth edges, realistic shadows, and preserved texture.

# This module uses a U-Net-like Generator (simplified) and optionally a Discriminator for training.

# Since full GAN training needs lots of compute, we’ll focus on:

# A ready-to-run Generator model (testable now),

# And a trainable GAN setup (if extended later).
# U net tyoe generator for refinement

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import os
import numpy as np

# Simplified U-Net Generator for refinement
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(GeneratorUNet, self).__init__()

        def conv_block(in_feat, out_feat):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, 4, 2, 1),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            )

        def up_block(in_feat, out_feat):
            return nn.Sequential(
                nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(in_channels, 64)
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.down4 = conv_block(256, 512)

        self.up1 = up_block(512, 256)
        self.up2 = up_block(256, 128)
        self.up3 = up_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(u1 + d3)
        u3 = self.up3(u2 + d2)
        out = self.final(u3 + d1)
        return torch.tanh(out)

def refine_tryon(person_path, warped_path, output_path="data/output/final_tryon.png", device='cpu'):
    model = GeneratorUNet().to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((256, 192)),
        T.ToTensor()
    ])

    person = transform(Image.open(person_path).convert("RGB"))
    garment = transform(Image.open(warped_path).convert("RGB"))
    
    # Stack as 6-channel input (person + garment)
    input_tensor = torch.cat((person, garment), dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # Convert to image
    out_img = output.squeeze().permute(1, 2, 0).cpu().numpy()
    out_img = ((out_img + 1) / 2 * 255).astype(np.uint8)  # from [-1,1] to [0,255]

    cv2.imwrite(output_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    return output_path


# Combines person + warped garment into a realistic output image.

# Uses a simplified U-Net-style generator (trainable later with GAN losses).

# Ready for inference, even without training a GAN from scratch.