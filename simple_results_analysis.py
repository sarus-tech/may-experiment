import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint
from bisect import bisect_left

with open('results/evaluation_on_56442_events.json', 'r') as f:
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
with open('train_counts.json', 'r') as f:
    train_counts = json.load(f)
    disease_count = [(disease.lower(), train_counts[disease]) for disease in train_counts]

evaluation_by_disease_count = Counter()
count_buckets = [10, 20, 50, 100, 200]
count_bucket_labels = ['[0 10[', '[10 20[', '[20 50[', '[50 100[', '[100 200[', '≥200']
for exp, nois in experiment_noise:
    cprint(f'Experiment {exp} with epsilon={nois}', 'blue')
    for disease, count in disease_count:
        # cprint(f'Disease {disease} with count={count}', 'light_blue')
        if (exp, nois, 'accuracy', disease) in evaluation:
            bucket_index = bisect_left(count_buckets, count)
            evaluation_by_disease_count[(exp, nois, 'accuracy', bucket_index)] += evaluation[((exp, nois, 'accuracy', disease))]
            evaluation_by_disease_count[(exp, nois, 'accuracy', bucket_index, 'disease')] += evaluation[((exp, nois, 'accuracy', disease, 'disease'))]
            evaluation_by_disease_count[(exp, nois, 'accuracy', bucket_index, 'drug')] += evaluation[((exp, nois, 'accuracy', disease, 'drug'))]

disease_accuracy_per_count = []
drug_accuracy_per_count = []
for exp, nois in experiment_noise:
        cprint(f'Experiment {exp} with noise multiplier={nois}', 'blue')
        for bucket_index in range(len(count_bucket_labels)):
            # cprint(f'Count = {i}', 'light_blue')
            if (exp, nois, 'accuracy', bucket_index) in evaluation_by_disease_count:
                disease_accuracy = evaluation_by_disease_count[(exp, nois, 'accuracy', bucket_index, 'disease')] / evaluation_by_disease_count[(exp, nois, 'accuracy', bucket_index)]
                drug_accuracy = evaluation_by_disease_count[(exp, nois, 'accuracy', bucket_index, 'drug')] / evaluation_by_disease_count[(exp, nois, 'accuracy', bucket_index)]
                cprint(f'Accuracy for bucket {bucket_index}', 'green')
                print(f" - accuracy for disease prediction = {round(100*disease_accuracy, 1)}%")
                print(f" - accuracy for drug prediction = {round(100*drug_accuracy, 1)}%")
                disease_accuracy_per_count.append((exp, bucket_index, disease_accuracy))
                drug_accuracy_per_count.append((exp, bucket_index, drug_accuracy))

for k, (exp, nois) in enumerate(experiment_noise):
    buckets, acc = zip(*[(count_bucket_labels[i], acc) for e,i,acc in disease_accuracy_per_count if e==exp])
    print(buckets)
    print(acc)
    plt.plot(buckets, acc, linewidth=2, marker='o')
plt.show()