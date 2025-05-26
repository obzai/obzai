# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2.functional as F
from typing import List


def batch_sanity_check(images: Image.Image | List[Image.Image] | torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Ensure the batch of images is in torch.Tensor format with shape (B, C, H, W).
    
    Parameters:
        images: A batch of images, which can be a numpy array, Torch tensor, PIL Image or List of PIL Images.
        
    Returns:
        Processed image as a Torch.Tensor of shape (B, C, H, W).
    """
    if isinstance(images, list):
        images = torch.stack([F.pil_to_tensor(img) for img in images], dim=0)  # (B, C, H, W)
    
    elif isinstance(images, Image.Image):
        images = F.pil_to_tensor(images)

    elif isinstance(images, np.ndarray):
        images = torch.tensor(images)

    elif not isinstance(images, torch.Tensor):
        raise ValueError(f"Unsupported image type: {type(images)}. Must be np.ndarray, torch.Tensor, or List of PIL Images.")

    # Single, grayscale images:
    if images.ndim == 2:  # Single grayscale image (H, W)
        images = images.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    elif images.ndim == 3:
        if images.shape[-1] in [1, 3]:  # Single RGB or grayscale image (H, W, C)
            images = images.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        else:  # (C, H, W) or (B, H, W)
            if images.shape[0] in [1, 3]:
                images = images.unsqueeze(0)  # (1, C, H, W)
            else:
                images = images.unsqueeze(dim=1) # (B, 1, H, W)

    elif images.ndim == 4 and images.shape[-1] in [1, 3]:  # Batched but with channel axis at -1: (B, H, W, C)
        images = images.permute(0, 3, 1, 2)  # (B, C, H, W)

    # Ensure channels are either 1 (grayscale) or 3 (RGB)
    if images.shape[1] not in [1, 3]:
        raise ValueError(f"Invalid channel dimension {images.shape[1]}. Expected 1 (grayscale) or 3 (RGB).")

    return images


def single_image_sanity_check(image: Image.Image | torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Ensure the single image is in torch.Tensor format with shape (C, H, W).
    
    Parameters:
        image: A single image, which can be a numpy array, Torch tensor, or PIL Image.
        
    Returns:
        Processed image as a Torch.Tensor of shape (C, H, W).
    """
    if isinstance(image, Image.Image):
        image = F.pil_to_tensor(image)  # Convert PIL Image to Tensor (C, H, W)
    
    elif isinstance(image, np.ndarray):
        image = torch.tensor(image)

    elif not isinstance(image, torch.Tensor):
        raise ValueError(f"Unsupported image type: {type(image)}. Must be np.ndarray, torch.Tensor, or PIL Image.")
    
    # Handle single grayscale images (H, W)
    if image.ndimension() == 2:  # Grayscale image (H, W)
        image = image.unsqueeze(0)  # (1, H, W)

    # Handle single RGB or grayscale images (H, W, C) or (C, H, W)
    elif image.ndimension() == 3:
        if image.shape[0] in [1, 3]:  # Already (C, H, W)
            pass  # No change needed
        elif image.shape[-1] in [1, 3]:  # (H, W, C)
            image = image.permute(2, 0, 1)  # Convert to (C, H, W)
        else:
            raise ValueError(f"Invalid shape {image.shape}. Expected (H, W, C) or (C, H, W).")
    
    # Ensure channels are either 1 (grayscale) or 3 (RGB)
    if image.shape[0] not in [1, 3]:
        raise ValueError(f"Invalid channel dimension {image.shape[0]}. Expected 1 (grayscale) or 3 (RGB).")
    
    return image
