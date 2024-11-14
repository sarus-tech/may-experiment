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

def finetune(client, params):
    response = client.post(
                    f"http://localhost:8000/v1/new/fine-tune", json=params,
                )
    assert response.status_code==200
    return  response.json()["task_id"]
    
    

with httpx.Client() as client:
    train_file=os.path.join(curr_path, 'train_ds.jsonl')
    test_file=os.path.join(curr_path, 'test_ds.jsonl')
    train_ds_uuid=upload_file(client, doc_path=train_file, doc_name='train_ds.jsonl')
    test_ds_uuid=upload_file(client, doc_path=test_file, doc_name='test_ds.jsonl')
    hyperparameters= {
        "is_dp": True,
        'noise_multiplier':0.05,
        'l2_norm_clip':0.01,
        "gradient_accumulation_steps": 32,
        "physical_batch_size": 16,
        "learning_rate": 3e-4,
        "epochs": 15,
        "use_lora": True,
        "quantize": True,
        "apply_lora_to_output":True,
        "apply_lora_to_mlp":True,
        "lora_attn_modules": ["q_proj","v_proj",'k_proj'],
        "save_every_n_grad_steps":50,
        "eval_every_n_grad_steps":10,
    }
    params={
        "sample_type": "instruct",
        "foundation_model_name": "open_mistral_7b",
        "train_dataset_id": train_ds_uuid,
        "test_dataset_id": test_ds_uuid,
        "hyperparameters": hyperparameters,
    }
    task_uuid=finetune(client,params=params)
    print(task_uuid)
    
    