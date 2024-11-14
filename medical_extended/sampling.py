import httpx
import os
from pathlib import Path
curr_path=Path(__file__).parent

def upload_file(client, doc_path, doc_name):
    with open(doc_path, "rb") as training_file:
        files = {"file":(doc_name, training_file,)}
        response = client.post(
            f"http://localhost:8000/v1/dataset/{doc_name.replace('.', '_')}",
            files=files,
        )
    assert response.status_code==200
    return  response.json()["dataset_id"]

def sample(client, params):
    response = client.post(
            f"http://localhost:8000/v1/new/sample", json=params,
        )
    print(response.text)
    assert response.status_code==200
    return  response.json()["task_id"]
    
    

with httpx.Client() as client:
    sample_file=os.path.join(curr_path, 'sample.jsonl')
    # sample_uuid=upload_file(client, doc_path=sample_file, doc_name='sample.jsonl')
    sample_uuid="sample_jsonl"
    finetuning_id="827b4b999b27364100e502f53b97ec16"
    hyperparameters={
        "max_length": 300,
        "batch_size": 100,
        "temperature": 1.0,
    }
    params={
        "saving_filename": "output.jsonl",
        "hyperparameters": hyperparameters,
        "finetuning_id": finetuning_id,
        "sample_dataset_id": sample_uuid,
    }
    task_uuid=sample(client, params=params)
    print(task_uuid)
    
    