# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Sequence, Union, List, Dict, Optional, NamedTuple

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from obz.data_inspector.extractor import Extractor, FirstOrderExtractor, CLIPExtractor
from obz.data_inspector.projector import EmbeddingProjector

logger = logging.getLogger("data_inspector")


#### FEATURE PIPELINE ####

class FeaturePipeline:
    """
    Handles feature extraction and normalization for one or more extractors.
    """
    def __init__(
        self,
        extractors: Union[Extractor, Sequence[Extractor]],
        show_progress: bool = False
    ):
        self.extractors = (
            extractors if isinstance(extractors, Sequence) else [extractors]
        )
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        for ext in self.extractors:
            self.feature_names.extend(ext.feature_names)
        self.show_progress = show_progress

    def fit(self, reference: DataLoader) -> np.ndarray:
        """
        Fit scalers on reference data and return the normalized feature matrix.
        """
        if not isinstance(reference, DataLoader):
            msg = "Reference must be a DataLoader!"
            logger.error(msg)
            raise ValueError(msg)

        # collect per-extractor features
        features_per_ext: Dict[int, List[np.ndarray]] = {
            ext.id: [] for ext in self.extractors
        }
        total_batches = len(reference)

        for ext in self.extractors:
            desc = f"Extracting features with {ext.name}"
            iterator = reference
            if self.show_progress:
                iterator = tqdm(reference, desc=desc, unit='batch', total=total_batches)

            for batch in iterator:
                imgs, _ = batch
                feats = ext.extract(imgs)
                features_per_ext[ext.id].append(feats)

        # stack, scale, and store
        normalized_blocks = []
        for ext in self.extractors:
            all_feats = np.vstack(features_per_ext[ext.id])
            scaler = StandardScaler()
            norm_feats = scaler.fit_transform(all_feats)
            self.scalers[ext.id] = scaler
            normalized_blocks.append(norm_feats)

        # concatenate horizontally
        merged = np.hstack(normalized_blocks)
        return merged

    def transform(self, batch: torch.Tensor) -> np.ndarray:
        """
        Extract and normalize features on-the-fly for inference.
        Returns a merged NumPy array (batch_size x total_features).
        """
        feats_per_ext = {}
        for ext in self.extractors:
            raw = ext.extract(batch)
            scaler = self.scalers.get(ext.id)
            if scaler is None:
                msg = f"Pipeline not fitted: missing scaler for extractor {ext.id}"
                logger.error(msg)
                raise ValueError(msg)
            feats_per_ext[ext.name] = (scaler.transform(raw))
        return feats_per_ext


#### MODELS ####

class ModelResult(NamedTuple):
    outliers: np.ndarray
    scores: np.ndarray

class ThresholdModel(ABC):
    """
    Wraps a generative or reconstruction model plus quantile-based thresholding.
    """
    def __init__(
        self,
        outlier_quantile: float = 0.01,
        higher_scores_normal: bool = True
    ):
        self.outlier_quantile = outlier_quantile
        self.higher_scores_normal = higher_scores_normal
        self.threshold: Optional[float] = None
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> None:
        scores = self.score(data)
        # if higher=normal, lower scores indicate anomaly: use quantile on scores
        # if higher=scores=anomaly, invert by negation
        target = scores if self.higher_scores_normal else -scores
        self.threshold = np.quantile(target, self.outlier_quantile)
        self.is_fitted = True

    def predict(self, data: np.ndarray) -> ModelResult:
        if not self.is_fitted:
            msg = "Model not fitted!"
            logger.error(msg)
            raise ValueError(msg)
        scores = self.score(data)
        target = scores if self.higher_scores_normal else -scores
        outliers = target < self.threshold
        return ModelResult(outliers=outliers, scores=scores)

    @abstractmethod
    def score(self, data: np.ndarray) -> np.ndarray:
        pass


class GMMModel(ThresholdModel):
    def __init__(
        self,
        n_components: int = 16,
        outlier_quantile: float = 0.01
    ):
        super().__init__(outlier_quantile=outlier_quantile, higher_scores_normal=True)
        self.gmm = GaussianMixture(n_components=n_components)

    def score(self, data: np.ndarray) -> np.ndarray:
        # GMM.log_prob (higher is better)
        return self.gmm.score_samples(data)

    def fit(self, data: np.ndarray) -> None:
        self.gmm.fit(data)
        super().fit(data)


class PCAReconstructionModel(ThresholdModel):
    def __init__(
        self,
        n_components: int = 64,
        outlier_quantile: float = 0.01
    ):
        # for reconstruction loss model, higher reconstruction error = more anomalous
        super().__init__(outlier_quantile=outlier_quantile, higher_scores_normal=False)
        self.pca = PCA(n_components=n_components)

    def fit(self, data: np.ndarray) -> None:
        self.pca.fit(data)
        super().fit(data)

    def score(self, data: np.ndarray) -> np.ndarray:
        # reconstruction loss per sample
        transformed = self.pca.transform(data)
        reconstructed = self.pca.inverse_transform(transformed)
        losses = np.mean((data - reconstructed) ** 2, axis=1)
        return losses


#### OUTLIER DETECTOR ####

class DetectionResult(NamedTuple):
    img_features: np.ndarray
    scores: np.ndarray
    outliers: np.ndarray

class PCADetectionResult(NamedTuple):
    img_features: np.ndarray
    scores: np.ndarray
    outliers: np.ndarray
    projector_results: np.ndarray


class OutlierDetector:
    """
    Generic wrapper that composes a FeaturePipeline with a ThresholdModel.
    """
    def __init__(
        self,
        pipeline: FeaturePipeline,
        model: ThresholdModel
    ):
        self.pipeline = pipeline
        self.model = model
        self.is_fitted = None

    def return_reference_features(self) -> pd.DataFrame:
        col_names = self.pipeline.feature_names
        df = pd.DataFrame(self.feats, columns=col_names)
        return df

    def fit(self, reference: DataLoader) -> None:
        self.feats = self.pipeline.fit(reference)
        self.model.fit(self.feats)
        self.is_fitted = True

    def detect(self, batch: torch.Tensor) -> DetectionResult:
        feats = self.pipeline.transform(batch)
        merged_feats = np.hstack([feats[key] for key in feats.keys()])
        outliers, scores = self.model.predict(merged_feats)
        return DetectionResult(img_features=feats, 
                               scores=scores, 
                               outliers=outliers)


# Convenience classes:
class GMMDetector(OutlierDetector):
    def __init__(
        self,
        extractors: FirstOrderExtractor,
        n_components: int = 16,
        outlier_quantile: float = 0.01,
        show_progress: bool = False
    ):
        self.data_inspection_routine_id = 1
        pipeline = FeaturePipeline(extractors, show_progress=show_progress)
        model = GMMModel(n_components=n_components, outlier_quantile=outlier_quantile)
        super().__init__(pipeline, model)


class PCAReconstructionLossDetector(OutlierDetector):
    def __init__(
        self,
        extractor: CLIPExtractor,
        n_components: int = 64,
        outlier_quantile: float = 0.01,
        show_progress: bool = False
    ):
        self.data_inspection_routine_id = 2
        pipeline = FeaturePipeline(extractor, show_progress=show_progress)
        # for visualization
        self.embedding_2D_projector = EmbeddingProjector()
        model = PCAReconstructionModel(n_components=n_components, outlier_quantile=outlier_quantile)
        super().__init__(pipeline, model)
    
    def return_reference_2D_components(self):
        return self.embedding_2D_projector.get_reference_embeddings()

    def fit(self, reference: DataLoader) -> None:
        self.feats = self.pipeline.fit(reference)
        # also fit 2D projector
        self.embedding_2D_projector.fit(self.feats)
        self.model.fit(self.feats)
        self.is_fitted = True

    def detect(self, batch: torch.Tensor) -> PCADetectionResult:
        feats = self.pipeline.transform(batch)
        merged_feats = np.hstack([feats[key] for key in feats.keys()])
        outliers, scores = self.model.predict(merged_feats)
        projector_results = self.embedding_2D_projector.transform(merged_feats)
        return PCADetectionResult(img_features=feats,
                                  scores=scores, 
                                  outliers=outliers, 
                                  projector_results=projector_results)
