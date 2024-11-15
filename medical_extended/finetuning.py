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
                    f"http://localhost:8000/v1/new/fine-tune", json=params,
                )
    assert response.status_code==200
    return  response.json()["task_id"]
    
def experiments():
    for noise_multiplier in [0.05, 1.0, 0.1, 0.2, 0.5, 2.0]:
        hyperparameters = {
            "is_dp": True,
            'noise_multiplier': noise_multiplier,
            'l2_norm_clip': 0.01,
            "gradient_accumulation_steps": 32,
            "physical_batch_size": 16,
            "learning_rate": 3e-4,
            "epochs": 15,
            "use_lora": True,
            "quantize": True,
            "apply_lora_to_output": True,
            "apply_lora_to_mlp": True,
            "lora_attn_modules": ["q_proj","v_proj",'k_proj'],
            "save_every_n_grad_steps": 50,
            "eval_every_n_grad_steps": 10,
        }
        params = {
            "sample_type": "instruct",
            "foundation_model_name": "open_mistral_7b",
            "train_dataset_id": train_ds_uuid,
            "test_dataset_id": test_ds_uuid,
            "hyperparameters": hyperparameters,
        }
        yield params

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
    
    