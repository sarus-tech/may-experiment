import json
import numpy as np
import matplotlib.pyplot as plt

with open('results/evaluation_on_37628_events.json', 'r') as f:
    packed_evaluation = json.load(f)

# Unpack the evaluation
evaluation = {}
for packed_key in packed_evaluation:
    evaluation[tuple(json.loads(packed_key))] = packed_evaluation[packed_key]


print(evaluation)