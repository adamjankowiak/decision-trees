import os
import pandas as pd
import numpy as np
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

file_name = "example_data_structure"
directory_name = "data"

os.makedirs("Results\\",exist_ok=True)
os.makedirs(f'Results\\{directory_name}',exist_ok=True)

df_train = pd.read_csv(f'{directory_name}//{file_name}.csv')
df_test = pd.read_csv(f'{directory_name}//{file_name}.csv')

metrics_per_class = {}
accuracy_scores = []

for i in range(1,11):
    r = random.randint(1,42)
    sample_size = int(0.1 * len(df_train))
    sampled_indices_index = df_train.sample(n=sample_size, random_state=r).index
    train_samples = df_train.loc[sampled_indices_index]
    test_samples = df_test.loc[sampled_indices_index]
    remaining_samples_index = df_train.index.difference(sampled_indices_index)
    train_remaining_samples = df_train.loc[remaining_samples_index]

    X_train = train_remaining_samples.iloc[:, 1:-4]
    Y_train = train_remaining_samples.iloc[:, -1]

    X_test = test_samples.iloc[:, 1:-4]
    Y_test = test_samples.iloc[:, -1]

    clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0, ccp_alpha=0.02)
    clf_entropy.fit(X_train, Y_train)
    y_pred_entropy = clf_entropy.predict(X_test)

    report = classification_report(Y_test, y_pred_entropy, output_dict=True, zero_division=0)

    accuracy_scores.append(report['accuracy'])

    for class_label, metrics in report.items():
        if class_label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        if class_label not in metrics_per_class:
            metrics_per_class[class_label] = {
                'precision': [],
                'recall': [],
                'f1-score': []
            }
        metrics_per_class[class_label]['precision'].append(metrics['precision'])
        metrics_per_class[class_label]['recall'].append(metrics['recall'])
        metrics_per_class[class_label]['f1-score'].append(metrics['f1-score'])

average_accuracy = np.mean(accuracy_scores)

final_metrics_per_class = {
    class_label: {metric: np.mean(values) for metric, values in metrics.items()}
    for class_label, metrics in metrics_per_class.items()
}

print("Average results after 10 iterations:")
for class_label, metrics in final_metrics_per_class.items():
    print(f"\nClass {class_label}:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

print(f"\nAverage accuracy after 10 iterations: {average_accuracy:.4f}")

with open(f'Results\\{directory_name}\\{file_name}_report.txt', 'w') as f:
    pass
with open(f'Results\\{directory_name}\\{file_name}_report.txt', 'a', encoding='UTF-8') as file:
    file.write("Average results after 10 iterations:")
    for class_label, metrics in final_metrics_per_class.items():
        file.write(f'\nClass {class_label}:')
        for metric, value in metrics.items():
            file.write(f"  {metric.capitalize()}: {value:.4f}")
    file.write(f'\nAverage accuracy after 10 iterations: {average_accuracy:.4f}')