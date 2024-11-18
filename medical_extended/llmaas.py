from typing import Any
import os
from io import BytesIO, TextIOWrapper
import gzip
import tarfile
import csv
import json
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
    
    def config(self, task_id: str):
        response = self.client.get(
            f"http://localhost:8000/v1/task/{task_id}/config",
        )
        assert response.status_code==200
        return  response.json()
    
    def status(self, task_id: str):
        response = self.client.get(
            f"http://localhost:8000/v1/task/{task_id}/status",
        )
        assert response.status_code==200
        return  response.json()['status']
    
    def download_sammple(self, task_id: str):
        response = self.client.get(
            f"http://localhost:8000/v1/task/{task_id}/download",
        )
        assert response.status_code==200
        compressed_buffer = BytesIO(response.content)
        with gzip.open(compressed_buffer, 'rb') as tar_file:
            tar_buffer = BytesIO(tar_file.read())
        with tarfile.open(fileobj=tar_buffer) as archive_file:
            archive_member = archive_file.getmember('output.csv')
            extracted_file = archive_file.extractfile(archive_member)
        for line in csv.reader(TextIOWrapper(extracted_file, encoding='utf-8')):
            try:
                yield line[0]
            except Exception as e:
                print(e)


if __name__ == "__main__":
    # A quick demo
    with httpx.Client() as client:
        llmaas = LLMaaS(client)
        for task_id in ['42b12dcf8823b8c3b57d4a71edb677e0']:
            print(f"Task: {task_id}")
            print(f"  Status: {llmaas.status(task_id)}")
            print(f"  Config: {llmaas.config(task_id)}")
        samples = llmaas.download_sammple(task_id)
        for _ in range(3):
            print(next(samples))
