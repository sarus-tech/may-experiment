from typing import Any
import os
import time
from math import exp
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher
import httpx
import json
from datasets import load_dataset
from Levenshtein import distance
from llmaas import LLMaaS


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
        self.prepare_privacy_test(ds['train'], 'privacy_test.jsonl')
        # Upload data
        self.train_file_id = self.llmaas.upload('train_ds.jsonl')
        self.test_file_id = self.llmaas.upload('test_ds.jsonl')
        self.sample_file_id = self.llmaas.upload('sample.jsonl')
        self.privacy_test_file_id = self.llmaas.upload('privacy_test.jsonl')
        # Ground truth
        with open(self.curr_path / 'ground_truth.jsonl', 'w') as f:
            for example in ds['test']:
                f.write(json.dumps({'drug': example['Drug'], 'disease': example['Disease']}) + "\n")
        # Parameter grid
        # self.noise_multipliers = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        self.noise_multipliers = [0.1, 0.2, 0.3, 0.4]
        

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
        """To test if the model can predict well the output"""
        with open(self.curr_path / save_file, "w") as f:
            for example in ds:
                messages = {
                    "messages": [
                        {"role": "user", "content": example["question"]},
                    ]
                }
                f.write(json.dumps(messages) + "\n")
        
    def prepare_privacy_test(self, ds, save_file):
        """To test if the model outputs some of the training set"""
        with open(self.curr_path / save_file, "w") as f:
            for example in ds:
                messages = {
                    "messages": [
                        {"role": "user", "content": example["question"]},
                    ]
                }
                f.write(json.dumps(messages) + "\n")
    
    def log(self, task_id: str, task_type: str, data: Any):
        """Log experiments"""
        logs = dict()
        try:
            with open(self.curr_path / 'experiments.json', 'r') as f:
                logs = json.load(f)
        except Exception:
            print("Could not read the experiment logs")
        logs[task_id] = {'type': task_type, 'data': data}
        try:
            with open(self.curr_path / 'experiments.json', 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception:
            print("Could not update the experiment logs")
    
    def finetuning_params(self, is_dp: bool=True, noise_multiplier: float=1.0, l2_norm_clip: float=0.01, learning_rate: float=5e-4, epochs: int=25) -> Any:
        hyperparameters = {
                "is_dp": is_dp,
                "noise_multiplier": noise_multiplier,
                "l2_norm_clip": l2_norm_clip,
                "gradient_accumulation_steps": 128,
                "physical_batch_size": 16,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "use_lora": True,
                "quantize": True,
                "apply_lora_to_output": True,
                "apply_lora_to_mlp": True,
                "lora_attn_modules": ["q_proj","v_proj","k_proj"],
                "save_every_n_grad_steps": 20,
                "eval_every_n_grad_steps": 5,
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
            "temperature": 0.1,
        }
        return {
            "saving_filename": "output.csv",
            "hyperparameters": hyperparameters,
            "finetuning_id": finetuning_id,
            "sample_dataset_id": self.sample_file_id,
        }
    
    def privacy_test_params(self, finetuning_id: str) -> Any:
        hyperparameters = {
            "max_length": 300,
            "batch_size": 100,
            "temperature": 0.1,
        }
        return {
            "saving_filename": "output.csv",
            "hyperparameters": hyperparameters,
            "finetuning_id": finetuning_id,
            "sample_dataset_id": self.privacy_test_file_id,
        }
        
    def finetune(self) -> list[str]:
        finetuning_ids = []
        for noise_multiplier in self.noise_multipliers:
            finetuning_params = self.finetuning_params(noise_multiplier=noise_multiplier)
            finetuning_id = self.llmaas.finetune(params=finetuning_params)
            finetuning_ids.append(finetuning_id)
            print(f"Finetuning task: {finetuning_id}")
            self.log(finetuning_id, 'finetuning', finetuning_params)
        return finetuning_ids
    
    def sample(self):
        finetuning_ids = self.finetune()
        sampling_ids = []
        privacy_test_ids = []
        for finetuning_id in finetuning_ids:
            while not self.llmaas.status(finetuning_id) == 'SUCCESS':
                print(f"Status of finetuning {finetuning_id} is {self.llmaas.status(finetuning_id)}")
                time.sleep(60)
            if self.llmaas.status(finetuning_id) == 'SUCCESS':
                # Simple sampling
                sampling_params = self.sampling_params(finetuning_id)
                sampling_id = self.llmaas.sample(sampling_params)
                sampling_ids.append(sampling_id)
                print(f"Sampling task: {sampling_id}")
                self.log(sampling_id, 'sampling', sampling_params)
                # Prepare privacy test
                privacy_test_params = self.privacy_test_params(finetuning_id)
                privacy_test_id = self.llmaas.sample(privacy_test_params)
                privacy_test_ids.append(privacy_test_id)
                print(f"Privacy test task: {privacy_test_id}")
                self.log(privacy_test_id, 'privacy_test', privacy_test_params)
        return zip(finetuning_ids, sampling_ids, privacy_test_ids)
    
    @staticmethod
    def privacy_risk_levenshtein(a: str, b: str) -> float:
        long = max(len(a), len(b))
        short = min(len(a), len(b))
        return 1-(1+distance(a, b)-(long-short))/(1+short)
    
    @staticmethod
    def privacy_risk(a: str, b: str) -> float:
        short = min(len(a), len(b))
        match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
        return 1 if match.size > short//10 else 0
    
    # TODO Add frequent disease
    def evaluate(self):
        finetuning_sample_privacy_test_ids = self.sample()
        evaluations = Counter()
        for finetuning_id, sample_id, privacy_test_id in finetuning_sample_privacy_test_ids:
            while not self.llmaas.status(finetuning_id) == 'SUCCESS':
                print(f"Status of finetuning {finetuning_id} is {self.llmaas.status(finetuning_id)}")
                time.sleep(60)
            while not self.llmaas.status(sample_id) == 'SUCCESS':
                print(f"Status of sampling {sample_id} is {self.llmaas.status(sample_id)}")
                time.sleep(60)
            while not self.llmaas.status(privacy_test_id) == 'SUCCESS':
                print(f"Status of privacy test {privacy_test_id} is {self.llmaas.status(privacy_test_id)}")
                time.sleep(60)
            if self.llmaas.status(finetuning_id) == 'SUCCESS' and self.llmaas.status(sample_id) == 'SUCCESS' and self.llmaas.status(privacy_test_id) == 'SUCCESS':
                # Simple sampling
                sampling_params = self.sampling_params(finetuning_id)
                sampling_id = self.llmaas.sample(sampling_params)
                with open(self.curr_path / 'ground_truth.jsonl') as f:
                    for sample, truth in zip(self.llmaas.download_sample(sampling_id), f):
                        truth = json.loads(truth)
                        evaluations[(finetuning_id, 'disease')] += 1
                        if truth['disease'].lower() in sample.lower():
                            evaluations[(finetuning_id, 'disease_ok')] += 1
                        evaluations[(finetuning_id, 'drug')] += 1
                        if truth['drug'].lower() in sample.lower():
                            evaluations[(finetuning_id, 'drug_ok')] += 1
                # Privacy evaluation
                privacy_test_params = self.privacy_test_params(finetuning_id)
                privacy_test_id = self.llmaas.sample(privacy_test_params)
                with open(self.curr_path / 'train_ds.jsonl') as f:
                    for sample, truth in zip(self.llmaas.download_sample(privacy_test_id), f):
                        evaluations[(finetuning_id, 'privacy')] += 1
                        truth_content = json.loads(truth)['messages'][-1]['content']
                        evaluations[(finetuning_id, 'privacy_ok')] += 1-self.privacy_risk(truth_content, sample)
        return evaluations

if __name__ == "__main__":
    with httpx.Client() as client:
        llmaas = LLMaaS(client)
        experiments = Experiments(llmaas)
        evaluations = experiments.evaluate()
        for finetuning_id in experiments.finetune():
            print(f"""Model {finetuning_id}:
  Disease accuracy: {(1+evaluations[(finetuning_id, 'disease_ok')])/(1+evaluations[(finetuning_id, 'disease')])}
  Drug accuracy: {(1+evaluations[(finetuning_id, 'drug_ok')])/(1+evaluations[(finetuning_id, 'drug')])}
  Privacy protection: {(1+evaluations[(finetuning_id, 'privacy_ok')])/(1+evaluations[(finetuning_id, 'privacy')])}
""")
        
