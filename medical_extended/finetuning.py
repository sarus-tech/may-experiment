import httpx
import os
import json
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

def finetune(client, params):
    response = client.post(
                    "http://localhost:8000/v1/new/fine-tune", json=params,
                )
    assert response.status_code==200
    return  response.json()["task_id"]
    


with httpx.Client() as client:
    experiments_file=os.path.join(curr_path, 'experiments.txt')
    train_file=os.path.join(curr_path, 'train_ds.jsonl')
    test_file=os.path.join(curr_path, 'test_ds.jsonl')
    try:
        train_ds_uuid=upload_file(client, doc_path=train_file, doc_name='train_ds.jsonl')
        test_ds_uuid=upload_file(client, doc_path=test_file, doc_name='test_ds.jsonl')
    except Exception:
        print("Files are already uploaded")
        train_ds_uuid='train_ds_jsonl'
        test_ds_uuid='test_ds_jsonl'
    for params in experiments():
        task_uuid=finetune(client,params=params)
        with open(experiments_file, 'a') as f:
            json_params = json.dumps(params, indent=2)
            f.write(f"Experiment: {task_uuid}\n{json_params}\n")
    
    