from typing import Any
from llmaas import LLMaaS

class Experiments:
    def __init__(self, llmaas: LLMaaS):
        self.llmaas = llmaas
        self.train_file_id = self.llmaas.upload('train_ds.jsonl')
        self.test_file_id = self.llmaas.upload('test_ds.jsonl')
        self.noise_multipliers = [0.05, 1.0, 0.1] #, 0.2, 0.5, 2.0]

    def prepare_dataset(ds, save_dir):
        with open(save_dir, "w") as outfile:
            for example in ds:
                messages = {
                    "messages": [
                        {"role": "user", "content": example["question"]},
                        {"role": "assistant", "content": example["answer"]},
                    ]
                }
                outfile.write(json.dumps(messages) + "\n")
        return
    
    def finetuning_params(self, is_dp: bool=True, noise_multiplier: float=1.0, l2_norm_clip: float=0.01, learning_rate: float=3e-4, epochs: int=15) -> Any:
        hyperparameters = {
                "is_dp": is_dp,
                "noise_multiplier": noise_multiplier,
                "l2_norm_clip": l2_norm_clip,
                "gradient_accumulation_steps": 32,
                "physical_batch_size": 16,
                "learning_rate": learning_rate,
                "epochs": 15,
                "use_lora": True,
                "quantize": True,
                "apply_lora_to_output": True,
                "apply_lora_to_mlp": True,
                "lora_attn_modules": ["q_proj","v_proj","k_proj"],
                "save_every_n_grad_steps": 50,
                "eval_every_n_grad_steps": 10,
            }
        return {
            "sample_type": "instruct",
            "foundation_model_name": "open_mistral_7b",
            "train_dataset_id": self.train_ds_uuid,
            "test_dataset_id": self.test_ds_uuid,
            "hyperparameters": hyperparameters,
        }
    
    def sampling_params(self, finetuning_id: str, sample_dataset_id: str) -> Any:
        hyperparameters = {
            "max_length": 300,
            "batch_size": 100,
            "temperature": 1.0,
        }
        return {
            "saving_filename": "output.jsonl",
            "hyperparameters": hyperparameters,
            "finetuning_id": finetuning_id,
            "sample_dataset_id": sample_dataset_id,
        }
        
    def params(self):
        for noise_multiplier in self.noise_multipliers:
            
        yield params

# with LLMaaS() as llmaas:
    
#     experiments_file=os.path.join(curr_path, 'experiments.txt')
    
#     for params in experiments():
#         task_uuid=finetune(client,params=params)
#         with open(experiments_file, 'a') as f:
#             json_params = json.dumps(params, indent=2)
#             f.write(f"Experiment: {task_uuid}\n{json_params}\n")
    
    