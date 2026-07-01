"""
Backend API Client for Streamlit frontend.
"""
from typing import Any
import sys
import httpx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import API_URL, API_KEY


class APIClient:
    """Client for centralizing HTTP requests to the FastAPI backend."""

    def __init__(self) -> None:
        self.base_url = API_URL
        self.headers = {"X-API-Key": API_KEY} if API_KEY else {}

    def _get(self, path: str) -> dict[str, Any]:
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=10.0) as client:
            response = client.get(path)
            response.raise_for_status()
            return response.json()

    def _post(self, path: str, json_data: Any) -> Any:
        with httpx.Client(base_url=self.base_url, headers=self.headers, timeout=10.0) as client:
            response = client.post(path, json=json_data)
            response.raise_for_status()
            return response.json()

    def get_liveness(self) -> dict[str, Any]:
        return self._get("/health/liveness")

    def get_readiness(self) -> dict[str, Any]:
        return self._get("/health/readiness")

    def get_model_info(self) -> dict[str, Any]:
        return self._get("/model")

    def get_model_features(self) -> dict[str, Any]:
        return self._get("/model/features")

    def get_metadata(self) -> dict[str, Any]:
        return self._get("/metadata")

    def get_version(self) -> dict[str, Any]:
        return self._get("/version")

    def predict_single(self, transaction: dict[str, Any]) -> dict[str, Any]:
        return self._post("/predict", transaction)

    def predict_batch(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._post("/predict/batch", transactions)


api_client = APIClient()
