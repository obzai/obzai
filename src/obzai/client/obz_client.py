# Obz AI - Copyright (C) 2025 Alethia XAI Sp. z o.o.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import Any, Dict, List, Optional, Tuple, Literal
import concurrent.futures
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import requests
import logging
import torch
import gzip
import os

from obz.client.logger_config import setup_logging
from obz.client.credentials import read_netrc_key, write_netrc_key, get_api_key_interactively
from obz.client.api_config import APIConfig

from obz.data_inspector.detector import GMMDetector, PCAReconstructionLossDetector
from obz.xai.xai_tool import AttentionTool, GradientTool
from obz.xai.xai_utils import normalize_xai_maps

ML_TASKS = {"binary_classification": 1,
            "multiclass_classification": 2,
            "multilabel_classification": 3,
            "semantic_segmentation": 4}
LOG_CHUNK_SIZE = 16
REF_ROWS_CHUNK_SIZE = 50


logger = logging.getLogger("obz_client")


class ObzClient:
    def __init__(
        self,
        detector=None,
        xai_tools=None,
        max_workers: int = 8,
        verbose: bool = False
    ):
        if verbose:
            setup_logging()
        
        self.detector = detector
        self.xai_tools = xai_tools or []
        self.max_workers = max_workers

        if not detector and not xai_tools:
            logger.info("Neither 'detector' nor 'xai_tools' were provided.\nThis will limit ObzClient functionality to logging only images and predictions.")

        self.user_verified = False
        self.project_id = None
        self.api_key = None

        # HTTP session
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def login(self, api_key: Optional[str] = None, relogin: bool = False):
        api_key = (
            api_key
            or os.getenv("OBZ_API_KEY")
            or (None if relogin else read_netrc_key())
        )
        if not api_key:
            logger.info("Cached credentials not available. Provide your API Key.")
            api_key = get_api_key_interactively()

        try:
            r = self.session.post(APIConfig.get_url("auth"), json={"api_token": api_key}, timeout=5)
            data = r.json()
            if r.status_code == 200 and data.get("success"):
                self.user_verified = True
                self.api_key = api_key
                if not read_netrc_key() or relogin:
                    write_netrc_key(api_key)
                logger.info("Login successful")
            else:
                raise ValueError("Invalid API key")
        except requests.RequestException as e:
            logger.error("Login failed: %s", e)
            raise RuntimeError(f"Login failed: {e}")
        
    def init_project(
        self,
        project_name: str,
        ml_task: Literal[
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
            "semantic_segmentation",
        ],
        logit2name: Optional[Dict[int, str]] = None
    ):
        if not self.user_verified:
            logger.info("Not logged in – attempting to authenticate with cached credentials.")
            try:
                self.login()
            except Exception:
                raise RuntimeError("Authentication required: run `.login(api_key=...)` first.")

        self.ml_task = ml_task
        self.logit2name = logit2name

        payload = {
            "api_token": self.api_key,
            "project_name": project_name,
            "ml_task": ML_TASKS[ml_task],
            "logit2name": logit2name,
            "data_inspection_routine_id": self.detector.data_inspection_routine_id
        }
        r = self.session.post(APIConfig.get_url("init_project"), json=payload, timeout=5)
        data = r.json()
        if r.status_code == 200 and data.get("success"):
            self.project_id = data["project_id"]
            logger.info("Project initialized (ID=%s).", self.project_id)
        else:
            msg = data.get("message", "Unknown error")
            logger.error("Failed to initialize project: %s", msg)
            raise RuntimeError(f"Project init failed: {msg}")

    def _create_ref_entry(self, reference_name:str):
        ref_entry = {
            "project_id": self.project_id,
            "reference_name": reference_name
        }
        try:
            response = self.session.post(
                APIConfig.get_url("create_ref_entry"),
                json=ref_entry,
                timeout=5,
                allow_redirects=False
                )
            response.raise_for_status()
            logger.info("Succesfully created ref entry.")
            return response.json()["id"]
        except requests.RequestException  as e:
            logger.error(f"Failed to create reference entry: {e}")

    def _upload_reference_features(self, ref_id:int, ref_features:pd.DataFrame, pca_or_umap:Optional[Literal["pca", "umap"]]=None):
        """
        Uploads reference features into database.
        """
        CHUNK_SIZE = 50

        # Split the DataFrame into chunks
        num_chunks = (len(ref_features) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for i, chunk in enumerate(range(0, len(ref_features), CHUNK_SIZE)):
            df_chunk = ref_features.iloc[chunk:chunk + CHUNK_SIZE]

            # Compress the chunk
            buf = BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="w") as gz:
                df_chunk.to_csv(gz, index=False)
            compressed_data = buf.getvalue()

            payload = {
                "ref_id": str(ref_id),
                "reducer_type": pca_or_umap if pca_or_umap else "none"
            }

            files = {
                "reference_file": (f"reference_features_chunk_{i + 1}.csv.gz", compressed_data, "application/gzip"),
            }

            # Upload the chunk
            try:
                response = self.session.post(
                    APIConfig.get_url("upload_ref_features"),
                    data=payload,
                    files=files,
                    timeout=30,
                )
                response.raise_for_status()
                logger.info("Uploaded chunk %d/%d successfully.", i + 1, num_chunks)
            except requests.RequestException as e:
                logger.error("Failed to upload chunk %d/%d: %s", i + 1, num_chunks, e)
                return None

        logger.info("All chunks uploaded successfully.")
        return {"status": "success", "chunks_uploaded": num_chunks}
    

    def log_reference(self, reference_name:str):
        """
        Method automatically log all reference to the backend.
        """
        if not self.project_id:
            logger.error("Project not initialized; call .init_project() first.")
            return None
        
        if self.detector:
            if not self.detector.is_fitted:
                logger.error("Please, firstly fit your outlier detector instance!")
                return None   
        else:
            logger.error("You haven't provide outlier detector instance, so there is no reference to log.")
            return None

        # Create a reference entry in the database and retrieve ref_id
        ref_id = self._create_ref_entry(reference_name)

        # Log features itself
        if isinstance(self.detector, GMMDetector):
            ref_features = self.detector.return_reference_features()
            self._upload_reference_features(ref_id, ref_features, pca_or_umap=None)
        elif isinstance(self.detector, PCAReconstructionLossDetector):
            ref_reduced_features = self.detector.return_reference_2D_components()
            self._upload_reference_features(ref_id, ref_reduced_features.pca_coords, pca_or_umap="pca")
            self._upload_reference_features(ref_id, ref_reduced_features.umap_coords, pca_or_umap="umap")
        else:
            msg = f"Not recognized outlier detector object of name: {self.detector.__class__.__name__}"
            logger.error(msg)
            raise ValueError(msg)


    def _encode_numpy(self, arr: np.ndarray) -> bytes:
        buf = BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb') as gz:
            np.save(gz, arr)
        return buf.getvalue()


    def _encode_image(self, img: torch.Tensor, quality: int = 75) -> bytes:
        arr = img.cpu().permute(1, 2, 0).numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255).astype('uint8')
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


    def _upload_parallel(self, items: List[Tuple[bytes, str]]) -> List[Optional[str]]:
        """Uploads binaries in parallel, returns list of keys or None on failure."""
        def upload(data_ext):
            data, ext = data_ext
            files = {'file': (f"upload.{ext}", data)}
            try:
                r = self.session.post(APIConfig.get_url("upload_image"), files=files, timeout=30)
                r.raise_for_status()
                return r.json().get('key')
            except Exception as e:
                logger.error("Upload error: %s", e)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(upload, items))

    def _compute_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.ml_task == "binary_classification":
            return torch.sigmoid(logits)
        elif self.ml_task == "multiclass_classification":
            return torch.softmax(logits, dim=1)
        else:
            raise NotImplementedError

    def _run_xai_tools(
        self,
        image_batch: torch.Tensor,
        logits: torch.Tensor,
        transform,
    ) -> Dict[int, List[np.ndarray]]:
        if self.ml_task == "binary_classification":
            target_idx = 0
        elif self.ml_task == "multiclass_classification":
            #TODO In future, we would like to compute XAI Maps for top-K classes.
            target_idx = torch.argmax(logits, dim=1).tolist()
        else:
            raise NotImplementedError(f"Currently {self.ml_task} scenario is not served.")

        tool_maps: Dict[int, List[np.ndarray]] = {}
        for tool in self.xai_tools:
            if isinstance(tool, AttentionTool):
                raw_maps = tool.explain(transform(image_batch))
            elif isinstance(tool, GradientTool):
                raw_maps = tool.explain(transform(image_batch), target_idx)

            tool_maps[tool.id] = normalize_xai_maps(raw_maps)
        return tool_maps

    def run_and_log(
        self,
        model: torch.nn.Module,
        image_batch: torch.Tensor,
        transform,
    ) -> Dict[str, Any]:
        
        if not self.project_id:
            logger.error("Project not initialized; call .init_project() first.")
            return None

        # 1) Inference
        with torch.no_grad():
            inputs = transform(image_batch)
            logits = model(inputs).cpu()
            probs = self._compute_probs(logits)

        # 2) Outlier detection
        outliers = None
        if self.detector:
            detection_results = self.detector.detect(image_batch)
            outliers = detection_results.outliers.tolist()
            outlier_scores = detection_results.scores.tolist()
            if isinstance(self.detector, GMMDetector):
                features = detection_results.img_features
                pca_coords, umap_coords = None, None
            elif isinstance(self.detector, PCAReconstructionLossDetector):
                features = None
                pca_coords = detection_results.projector_results.pca_coords.tolist()
                umap_coords = detection_results.projector_results.umap_coords.tolist()         

        # 3) XAI
        xai_maps = None
        if self.xai_tools:
            xai_maps = self._run_xai_tools(image_batch, logits, transform)

        # 4) Encode binaries
        raw_bytes = [self._encode_image(img) for img in image_batch]
        xai_bytes: Dict[int, List[bytes]] = {}
        if xai_maps:
            for tool_id, maps in xai_maps.items():
                xai_bytes[tool_id] = [self._encode_numpy(m) for m in maps]

        # 5) Parallel upload
        raw_keys = self._upload_parallel([(b, 'jpeg') for b in raw_bytes])
        xai_keys: Dict[int, List[Optional[str]]] = {}
        for tool_id, bytes_list in xai_bytes.items():
            keys = self._upload_parallel([(b, 'npy.gz') for b in bytes_list])
            xai_keys[tool_id] = keys
        
        #6) Chunked log metadata
        results = []
        N = len(raw_keys)
        for i in range(0, N, LOG_CHUNK_SIZE):
            # Prepare chunk of data to log
            probabilities_chunk = probs.tolist()[i:i+LOG_CHUNK_SIZE]
            raw_keys_chunk = raw_keys[i:i+LOG_CHUNK_SIZE]
            xai_map_keys_chunk = {tool_id: keys[i:i+LOG_CHUNK_SIZE] for tool_id, keys in xai_keys.items()} or None
            xai_eval_chunk = None
            outliers_chunk = outliers[i:i+LOG_CHUNK_SIZE] if outliers else None
            outlier_scores_chunk = outlier_scores[i:i+LOG_CHUNK_SIZE]
            img_features_chunk = {feat_name: features[feat_name][i:i+LOG_CHUNK_SIZE].tolist() for feat_name in features.keys()} if features else None

            if pca_coords and umap_coords:
                pca_coords = pca_coords[i:i+LOG_CHUNK_SIZE]
                umap_coords = umap_coords[i:i+LOG_CHUNK_SIZE]
            else:
                pca_coords, umap_coords = None, None

            batch = {
                'project_id': self.project_id,
                'probabilities': probabilities_chunk,
                'raw_keys': raw_keys_chunk,
                'xai_map_keys': xai_map_keys_chunk,
                'xai_eval': xai_eval_chunk,
                'outliers': outliers_chunk,
                'outlier_scores': outlier_scores_chunk,
                'img_features': img_features_chunk,
                'pca_coords': pca_coords,
                'umap_coords': umap_coords
            }
            resp = self.session.post(APIConfig.get_url("log"), json=batch, timeout=50)

            try:
                resp.raise_for_status()
                results.append(resp.json())
            except Exception as e:
                logger.error("Logging batch failed: %s", e)
                results.append({'status': 'error', 'detail': str(e)})
        
        logger.info(f"Uploaded batch of data from {len(image_batch)} images.")

        return {
            'logits': logits,
            'probs': probs,
            'outliers': outliers,
            'outlier_scores': outlier_scores,
            'features': features,
            'xai_maps': xai_maps,
            'log_responses': results
        }