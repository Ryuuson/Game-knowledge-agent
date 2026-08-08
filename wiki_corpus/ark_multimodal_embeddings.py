"""Dense text embeddings from Ark's multimodal embedding endpoint."""

import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import requests


def extract_dense_embedding(payload: Mapping[str, Any]) -> np.ndarray:
    """Read the dense vector returned as ``data.embedding`` by Ark multimodal API."""
    data = payload.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("embedding"), list):
        raise ValueError("Ark multimodal response does not contain data.embedding")
    vector = np.asarray(data["embedding"], dtype=np.float32)
    if vector.ndim != 1 or not len(vector):
        raise ValueError("Ark multimodal response contains an invalid dense embedding")
    return vector


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    """Normalize a dense vector so SQLite search uses cosine similarity consistently."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or norm == 0:
        raise ValueError("embedding must be a non-zero one-dimensional vector")
    return vector / norm


class ArkMultimodalTextEmbedder:
    """Request one dense text embedding at a time from a deployed Ark endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.url = f"{base_url.rstrip('/')}/embeddings/multimodal"
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def encode(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("text for embedding cannot be empty")

        payload = {
            "model": self.model,
            "input": [{"type": "text", "text": text}],
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.url, headers=headers, json=payload, timeout=self.timeout
                )
                if response.ok:
                    return normalize_embedding(extract_dense_embedding(response.json()))
                retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt == self.max_retries:
                    raise RuntimeError(f"Ark embedding request failed: HTTP {response.status_code}")
            except requests.RequestException as error:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Ark embedding request failed: {error}") from error
            time.sleep(2**attempt)

        raise RuntimeError("Ark embedding request exhausted retries")
