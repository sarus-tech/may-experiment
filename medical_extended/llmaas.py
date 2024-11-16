from typing import Any
import os
import httpx
from pathlib import Path

class LLMaaS:
    """A LLMaaS client"""
    def __init__(self, client: httpx.Client):
        self.client = client
        self.curr_path = Path(__file__).parent

    def upload(self, name: str, exist_ok=True) -> str:
        path = os.path.join(self.curr_path, name)
        with open(path, "rb") as training_file:
            files = {"file":(name, training_file,)}
            dataset_id = name.replace('.', '_')
            try:
                response = self.client.post(
                    f"http://localhost:8000/v1/dataset/{dataset_id}",
                    files=files,
                )
                assert response.status_code==200 and response.json()["dataset_id"]==dataset_id
            except Exception:
                if not exist_ok:
                    raise RuntimeError("The file could not be uploaded. Maybe the file already exists")
        return dataset_id

    def finetune(self, params: Any):
        response = self.client.post(
            "http://localhost:8000/v1/new/fine-tune", json=params,
        )
        assert response.status_code==200
        return  response.json()["task_id"]

    def sample(self, params: Any):
        response = self.client.post(
            "http://localhost:8000/v1/new/sample", json=params,
        )
        assert response.status_code==200
        return  response.json()["task_id"]