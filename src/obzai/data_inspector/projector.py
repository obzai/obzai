# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

from sklearn.decomposition import PCA
from typing import NamedTuple
import pandas as pd
import numpy as np
#import umap


class EmbeddingProjectorResult(NamedTuple):
    pca_coords: np.ndarray
    umap_coords: np.ndarray


class EmbeddingProjector:
    """
    Encodes images using a DNN encoder and projects embeddings to lower-dimensional spaces (PCA/UMAP).
    """
    def __init__(self, pca_components: int = 2, umap_components: int = 2):
        self.pca = PCA(n_components=pca_components)
        self.umap = PCA(n_components=pca_components) #umap.UMAP(n_components=umap_components) if umap else None

        self.pca_embeddings = None
        self.umap_embeddings = None
        self.is_fitted = False

    def fit(self, img_embedding: np.ndarray):
        """
        Fit PCA and UMAP (if available) on embeddings from the reference feature set.
        """
        self.pca_embeddings = pd.DataFrame(self.pca.fit_transform(img_embedding), columns=["x_coor", "y_coor"])
        self.umap_embeddings = pd.DataFrame(self.umap.fit_transform(img_embedding), columns=["x_coor", "y_coor"])
        self.is_fitted = True

    def get_reference_embeddings(self):
        """
        Returns reduced reference embeddings.
        """
        if not self.is_fitted:
            raise ValueError("Projector not fitted.")
        
        return EmbeddingProjectorResult(pca_coords=self.pca_embeddings, umap_coords=self.umap_embeddings)

    def transform(self, img_embedding:np.ndarray):
        """
        Encode images and return their PCA and UMAP projections.
        """
        if not self.is_fitted:
            raise ValueError("Projector not fitted. Call fit() first.")

        pca_coords = self.pca.transform(img_embedding)
        umap_coords = self.umap.transform(img_embedding) if self.umap else None
        return EmbeddingProjectorResult(pca_coords=pca_coords, umap_coords=umap_coords)
