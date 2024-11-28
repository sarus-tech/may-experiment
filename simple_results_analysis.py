import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint

with open('results/evaluation_on_37628_events.json', 'r') as f:
    packed_evaluation = json.load(f)

# Unpack the evaluation
evaluation = {}
for packed_key in packed_evaluation:
    evaluation[tuple(json.loads(packed_key))] = packed_evaluation[packed_key]

cprint(evaluation, "dark_grey")

experiment_noise = sorted({(key[0], key[1]) for key in evaluation if key[0] != '*'}, key = lambda en: en[1])
cprint(experiment_noise[0][0], "red")

for exp, nois in experiment_noise:
    cprint(f'Experiment {exp} with epsilon={nois}', 'blue')
    disease_accuracy = evaluation[(exp, nois, 'accuracy', '*', 'disease')] / evaluation[(exp, nois, 'accuracy', '*')]
    drug_accuracy = evaluation[(exp, nois, 'accuracy', '*', 'drug')] / evaluation[(exp, nois, 'accuracy', '*')]
    cprint(f'Accuracy for experiment {exp}', 'green')
    print(f" - accuracy for disease prediction = {round(100*disease_accuracy, 1)}%")
    print(f" - accuracy for drug prediction = {round(100*drug_accuracy, 1)}%")
    privacy_risk = evaluation[(exp, nois, 'privacy', 'risk')] / evaluation[(exp, nois, 'privacy')]
    privacy_breach = evaluation[(exp, nois, 'privacy', 'breach')] / evaluation[(exp, nois, 'privacy')]
    cprint(f'Privacy for experiment {exp}', 'red')
    print(f" - privacy risk = {round(100*privacy_risk, 1)}%")
    print(f" - privacy breach = {round(100*privacy_breach, 1)}%")
    print()

# Count by disease frequency
# Compute disease frequency
disease_count = [(key[3], evaluation[key]) for key in evaluation if key[0] == experiment_noise[0][0] and key[2] == 'accuracy' and len(key)==4]
cprint(disease_count, "blue")

evaluation_by_disease_count = Counter()

for exp, nois in experiment_noise:
    cprint(f'Experiment {exp} with epsilon={nois}', 'blue')
    for disease, count in disease_count:
        cprint(f'Disease {disease} with count={count}', 'light_blue')
        evaluation_by_disease_count[(exp, nois, 'accuracy', count)] += evaluation[((exp, nois, 'accuracy', disease))]
        evaluation_by_disease_count[(exp, nois, 'accuracy', count, 'disease')] += evaluation[((exp, nois, 'accuracy', disease, 'disease'))]
        evaluation_by_disease_count[(exp, nois, 'accuracy', count, 'drug')] += evaluation[((exp, nois, 'accuracy', disease, 'drug'))]

cprint(evaluation_by_disease_count, "yellow")