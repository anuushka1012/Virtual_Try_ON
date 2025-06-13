# warping/stn_warp.py

# This module:

# Takes the garment image,

# Takes keypoints from YOLO pose,

# Warps the garment to align with the person’s shape using a spatial transformation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class STNWarpModule(nn.Module):
    def __init__(self):
        super(STNWarpModule, self).__init__()
        # Tiny localization net for affine transformation (can be expanded)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),

            nn.MaxPool2d(2,2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2,2),
            nn.ReLU(True)
        )

        # Affine transform: 2x3 params
        # Predicts 6 numbers that form a 2×3 affine transformation matri
        
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 53 * 38, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        # Initialize to identity transform
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # grid
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 53 * 38)
        theta = self.fc_loc(xs).view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

def warp_garment(garment_path, output_path="data/preprocessed/warped_garment.png"):
    model = STNWarpModule()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor()
    ])

    # Load image
    img = Image.open(garment_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    # Forward through STN
    with torch.no_grad():
        warped = model(input_tensor)

    # Convert back to image
    warped_img = warped.squeeze().permute(1,2,0).numpy()
    warped_img = (warped_img * 255).astype(np.uint8)

    cv2.imwrite(output_path, cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR))
    return output_path



# Loads the garment image.

# Warps it using a mock-STN (expandable to deformable conv later).

# Saves the warped garment image to feed into the GAN refinement stage.

# nn.Conv2d(3, 8, kernel_size=7)
# Input: 3 channels (R, G, B)

# Output: 8 feature maps (filters)

# Kernel size: 7×7 (looks at 7×7 pixel blocks)


# nn.MaxPool2d(2,2)
# Downsamples the image by reducing size by half.

# Takes the strongest signal in each 2×2 area.

# Helps the network focus on main patterns, not pixel details.

# nn.ReLU(True)
# Adds non-linearity so the network can learn more complex patterns.

# ReLU turns negative values into zero and keeps positive as-is.