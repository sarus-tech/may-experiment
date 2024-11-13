import httpx
import os
from pathlib import Path
curr_path=Path(__file__).parent

def upload_file(client,doc_path,doc_name):
    with open(doc_path, "rb") as training_file:
                files = {"file":(doc_name, training_file,)}
                response = client.post(
                    f"http://localhost:8000/v1/dataset",
                    files=files,
                )
    assert response.status_code==200
    return  response.json()["dataset_id"]

def sample(client,params):
    response = client.post(
                    f"http://localhost:8000/v1/new/sample", json=params,
                )
    assert response.status_code==200
    return  response.json()["task_id"]
    
    

with httpx.Client() as client:

    sample_file=os.path.join(curr_path,'sample.jsonl')
    sample_uuid=upload_file(client,doc_path=sample_file,doc_name='sample_ds.jsonl')
    
    finetuning_id="43a494320d5c6be2a9f801027c816832"
    hyperparameters= { "batch_size": 100,
            "max_length": 300,
            "temperature": 0.1,}

    params={ 
            "hyperparameters": hyperparameters,
            "sample_dataset_id":sample_uuid,
            "finetuning_id": finetuning_id
}
    
    task_uuid=sample(client,params=params)
    print(task_uuid)
    
    