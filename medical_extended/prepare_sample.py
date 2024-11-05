from datasets import load_dataset
import os
import json

temp_dir = os.getcwd()
sample_data_dir = os.path.join(temp_dir, "sample.jsonl")
def prepare_json(ds, save_dir):
    with open(save_dir, "w") as outfile:
        for example in ds:
            messages = {
                "messages": [
                    {"role": "user", "content": example["question"]}
                ]
            }
            outfile.write(json.dumps(messages) + "\n")
    return

ds=load_dataset('sarus-tech/medical_extended',token=os.getenv('HF_TOKEN'))['train']
ds=ds.train_test_split(train_size=9000,seed=10)
prepare_json(ds['test'],sample_data_dir)
