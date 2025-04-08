import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import classification_report
from sklearn import tree

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (prob: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules

file_name = "example_data_structure"
directory_name = "data"

os.makedirs("Results\\",exist_ok=True)
os.makedirs(f'Results\\Classifier',exist_ok=True)

df = pd.read_csv(f'{directory_name}//{file_name}.csv')

X_train = df.iloc[:, 1:-4]
Y_train = df.iloc[:, -1]

X_test = df.iloc[:, 1:-4]
Y_test = df.iloc[:, -1]

clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0, ccp_alpha=0.02)
clf_entropy.fit(X_train, Y_train)
y_pred_entropy = clf_entropy.predict(X_test)

plt.figure(figsize=(20, 16))
tree.plot_tree(clf_entropy, feature_names=X_test.columns,label='all')
plt.savefig(f'Results\\Classifier\\Classifier_tree.png')
# plt.show()

rules = get_rules(clf_entropy, X_test.columns, ["1-2","3-4"])

with open(f'Results\\Classifier\\Classifier_rules.txt', 'w') as f:
    pass
with open(f'Results\\Classifier\\Classifier_rules.txt', 'a', encoding='UTF-8') as file:
    for r in rules:
        file.write(f'{r}\n')

with open(f'Results\\Classifier\\Classifier_report.txt', 'w') as f:
    pass
with open(f'Results\\Classifier\\Classifier_report.txt', 'a', encoding='UTF-8') as file:
    file.write(classification_report(Y_test, y_pred_entropy))