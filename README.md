## Setup the env
You need:
- a running image of llmaas
- use the makefile to lock and create a virtual env to launch the jobs.

## Launch experiments (DEPRECATED)
- Each folder corresponds to one dataset.
- the scripts prepare_sample/prepare_dataset download the datasets from huggingface and create jsonl files to be uploaded to the service
- the script finetuning.py/sampling.py launch a finetuning/sampling, the config has to be filled in these scripts.
- the file config.txt sums up the ones I used to run the experiments, there is also the hash of the task that it is supposed to be retrieved.

## Launch experiments
- Each folder corresponds to one dataset.
- the script experiments.py download the datasets from huggingface, create files, upload them, rn the finetuning and sampling jobs
- the file experiments.txt sums up the experiments run

