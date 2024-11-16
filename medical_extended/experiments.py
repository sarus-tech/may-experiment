from typing import Any
import os
import httpx
from datasets import load_dataset
import json
from llmaas import LLMaaS
from pathlib import Path

class Experiments:
    def __init__(self, llmaas: LLMaaS, seed=10):
        self.llmaas = llmaas
        self.seed = seed
        self.curr_path = Path(__file__).parent
        # Prepare data
        ds = load_dataset('sarus-tech/medical_extended',token=os.getenv('HF_TOKEN'))['train']
        ds = ds.train_test_split(train_size=9000, seed=self.seed)
        self.prepare_dataset(ds['train'], 'train_ds.jsonl')
        self.prepare_dataset(ds['test'], 'test_ds.jsonl')
        self.prepare_sample(ds['test'], 'sample.jsonl')
        # Upload data
        self.train_file_id = self.llmaas.upload('train_ds.jsonl')
        self.test_file_id = self.llmaas.upload('test_ds.jsonl')
        self.sample_file_id = self.llmaas.upload('sample.jsonl')
        # Ground truth
        with open(self.curr_path / 'ground_truth.jsonl', 'w') as f:
            for example in ds['test']:
                f.write(json.dumps({'drug': example['Drug'], 'disease': example['Disease']}) + "\n")
        # Parameter grid
        self.noise_multipliers = [0.05, 1.0, 0.1] #, 0.2, 0.5, 2.0]

    def prepare_dataset(self, ds, save_file):
        """Pull the dataset and dump as jsonl"""
        with open(self.curr_path / save_file, 'w') as f:
            for example in ds:
                messages = {
                    "messages": [
                        {"role": "user", "content": example["question"]},
                        {"role": "assistant", "content": example["answer"]},
                    ]
                }
                f.write(json.dumps(messages) + "\n")
    
    def prepare_sample(self, ds, save_file):
        """Pull the dataset and dump as jsonl"""
        with open(self.curr_path / save_file, "w") as f:
            for example in ds:
                messages = {
                    "messages": [
                        {"role": "user", "content": example["question"]},
                    ]
                }
                f.write(json.dumps(messages) + "\n")
    
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
            "train_dataset_id": self.train_file_id,
            "test_dataset_id": self.test_file_id,
            "hyperparameters": hyperparameters,
        }
    
    def sampling_params(self, finetuning_id: str) -> Any:
        hyperparameters = {
            "max_length": 300,
            "batch_size": 100,
            "temperature": 1.0,
        }
        return {
            "saving_filename": "output.jsonl",
            "hyperparameters": hyperparameters,
            "finetuning_id": finetuning_id,
            "sample_dataset_id": self.sample_file_id,
        }
        
    def finetune(self) -> list[str]:
        finetuning_ids = []
        for noise_multiplier in self.noise_multipliers:
            finetuning_params = self.finetuning_params(noise_multiplier=noise_multiplier)
            finetuning_id = self.llmaas.finetune(params=finetuning_params)
            finetuning_ids.append(finetuning_id)
            print(f"Finetuning task: {finetuning_id}")
            with open(self.curr_path / 'experiments.txt', 'a') as f:
                json_params = json.dumps(finetuning_params, indent=2)
                f.write(f"Finetuning task: {finetuning_id}\n{json_params}\n")
        return finetuning_ids
    
    def sample(self):
        finetuning_ids = self.finetune()
        for finetuning_id in finetuning_ids:
            sampling_params = self.sampling_params(finetuning_id)
            sampling_id = self.llmaas.sample(sampling_params)
            print(f"Sampling task: {sampling_id}")
            with open(self.curr_path / 'experiments.txt', 'a') as f:
                json_params = json.dumps(sampling_params, indent=2)
                f.write(f"Finetuning task: {finetuning_id}\n{json_params}\n")


with httpx.Client() as client:
    llmaas = LLMaaS(client)
    experiments = Experiments(llmaas)
    # experiments.finetune()
