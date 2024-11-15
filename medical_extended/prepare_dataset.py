from datasets import load_dataset
import os
import json

temp_dir= os.getcwd()
train_data_dir = os.path.join(temp_dir, "train_ds.jsonl")
test_data_dir = os.path.join(temp_dir, "test_ds.jsonl")

os.makedirs(temp_dir, exist_ok=True)


def prepare_json(ds, save_dir):
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

ds=load_dataset('sarus-tech/medical_extended',token=os.getenv('HF_TOKEN'))['train']
ds=ds.train_test_split(train_size=9000,seed=10)
prepare_json(ds['train'], train_data_dir)
prepare_json(ds['test'], test_data_dir)