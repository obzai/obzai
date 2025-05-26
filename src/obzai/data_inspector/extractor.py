# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

from transformers import CLIPVisionModel, CLIPImageProcessor
import torchvision.transforms.v2.functional as F
from scipy.stats import skew, kurtosis
from typing import Sequence, Literal
from abc import ABC, abstractmethod
import torch


class Extractor(ABC):
    def __init__(self):
        self.id: int = None
        self.name: str = None

    def _process_batch(self, 
                       batch: torch.Tensor,
                       image_size: int|Sequence[int] = 224,
                       ensure_grayscale: bool = False,
                       ensure_scale: bool=True) -> torch.Tensor:
        """
        Method accepts a torch.Tensor batch of images and processes it by resizing, optionaly grey scaling.
        Parameters:
            image: Input image as a torch.Tensor of shape (B, C, H, W) or (B, 1, H, W) if grayscale.
        """
        batch = batch.cpu()

        # Ensure proper DataType and scale:
        if ensure_scale:
            batch = F.to_dtype(batch, dtype=torch.float32, scale=True)

        # Resizing the image into a specified size
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        batch = F.resize(batch, size=image_size)
        
        # Convert RGB image to grayscale (if needed)
        if ensure_grayscale:
            F.rgb_to_grayscale(batch)
        
        return batch
    
    @abstractmethod
    def extract(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Method implements a custom loop over batch during features extraction.
        """
        pass


class DummyExtractor(Extractor):
    def __init__(self, feature_size:int=32):
        self.rng = torch.randn
        self.feature_size = feature_size
        self.feature_names = [f"dummy_{i}" for i in range(self.feature_size)]
        self.id = -1
        self.name = "DummyExtractor"
    
    def _compute_features_for_image(self) -> torch.Tensor:
        """Returns a dummy feature vector"""
        return self.rng(self.feature_size)
    
    def extract(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Extracts features during production.
        """
        features = []
        for idx in range(len(image_batch)):
            feat = self._compute_features_for_image()
            features.append(feat)
        return torch.stack(features, dim=0).numpy()
    

class FirstOrderExtractor(Extractor):
    """
    Extracts first-order statistical features from images, inspired by PyRadiomics.
    """
    def __init__(self):        
        # Feature extraction methods
        self.feature_functions = {
            "entropy": self.get_entropy,
            "min": self.get_min_value,
            "max": self.get_max_value,
            "10th_percentile": self.get_10th_percentile,
            "90th_percentile": self.get_90th_percentile,
            "mean": self.mean,
            "median": self.median,
            "interquartile_range": self.get_interquartile_range,
            "range": self.get_range,
            "mean_absolute_deviation": self.get_mean_absolute_deviation,
            "robust_mean_absolute_deviation": self.get_robust_mean_absolute_deviation,
            "root_mean_square": self.get_root_mean_square,
            "skewness": self.get_skewness,
            "kurtosis": self.get_kurtosis,
            "variance": self.get_variance,
            "uniformity": self.get_uniformity
        }
        self.feature_names = list(self.feature_functions.keys())
        self.feat_ids = [i for i in range(1,17)]
        self.id = 1
        self.name = "FirstOrderExtractor"

    def get_entropy(self, image: torch.Tensor) -> torch.Tensor:
        hist = torch.histc(image, bins=256, min=0, max=1)
        hist = hist / hist.sum()
        return -(hist * torch.log2(hist + 1e-6)).sum()
    
    def get_min_value(self, image: torch.Tensor) -> torch.Tensor:
        return image.min()

    def get_max_value(self, image: torch.Tensor) -> torch.Tensor:
        return image.max()

    def get_10th_percentile(self, image: torch.Tensor) -> torch.Tensor:
        return torch.quantile(image.flatten(), 0.1)

    def get_90th_percentile(self, image: torch.Tensor) -> torch.Tensor:
        return torch.quantile(image.flatten(), 0.9)
    
    def mean(self, image: torch.Tensor) -> torch.Tensor:
        return torch.mean(image)
    
    def median(self, image: torch.Tensor) -> torch.Tensor:
        return torch.median(image)

    def get_interquartile_range(self, image: torch.Tensor) -> torch.Tensor:
        q3 = torch.quantile(image.flatten(), 0.75)
        q1 = torch.quantile(image.flatten(), 0.25)
        return q3 - q1

    def get_range(self, image: torch.Tensor) -> torch.Tensor:
        return image.max() - image.min()

    def get_mean_absolute_deviation(self, image: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(image)
        return torch.mean(torch.abs(image - mean))

    def get_robust_mean_absolute_deviation(self, image: torch.Tensor) -> torch.Tensor:
        prcnt10 = self.get_10th_percentile(image)
        prcnt90 = self.get_90th_percentile(image)
        arr_subset = image[(image >= prcnt10) & (image <= prcnt90)]
        return torch.mean(torch.abs(arr_subset - torch.mean(arr_subset)))
    
    def get_root_mean_square(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(image ** 2))

    def get_skewness(self, image: torch.Tensor) -> torch.Tensor:
        res = skew(image.flatten().detach().cpu().numpy())
        return torch.tensor(res)

    def get_kurtosis(self, image: torch.Tensor) -> torch.Tensor:
        res = kurtosis(image.flatten().detach().cpu().numpy())
        return torch.tensor(res)

    def get_variance(self, image: torch.Tensor) -> torch.Tensor:
        return torch.var(image)
    
    def get_uniformity(self, image: torch.Tensor) -> torch.Tensor:
        hist = torch.histc(image, bins=256, min=0, max=1)
        hist = hist / hist.sum()
        return (hist ** 2).sum()

    def _compute_features_for_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute all first-order statistical features for a single image.
        """
        features = [func(image) for func in self.feature_functions.values()]
        return torch.stack(features)

    def extract(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Extracts features during production.
        """
        image_batch = self._process_batch(image_batch, ensure_grayscale=True, ensure_scale=True)
        features = []
        for idx in range(len(image_batch)):
            feat = self._compute_features_for_image(image_batch[idx])
            features.append(feat)
        return torch.stack(features, dim=0).numpy()


class CLIPExtractor(Extractor):
    def __init__(self, patch_size: Literal[16,32] = 32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_processor = CLIPImageProcessor.from_pretrained(f"openai/clip-vit-base-patch{patch_size}")
        self.model = CLIPVisionModel.from_pretrained(f"openai/clip-vit-base-patch{patch_size}", device_map=self.device)
        self.model.to(self.device)

        self.feature_names = [f"clip_{i}" for i in range(768)]
        self.id = 2
        self.name = "CLIPExtractor"

    def extract(self, image_batch: torch.Tensor):
        """
        Extracts features during production.
        """
        image_batch = self.img_processor(images=image_batch, return_tensors="pt", do_rescale=False)
        outputs = self.model(image_batch['pixel_values'].to(self.device))
        return outputs['pooler_output'].detach().cpu().numpy()