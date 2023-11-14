import random
import matplotlib.pyplot as plt
import laspy
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import display

t1 = time.time()
rand1 = random.randint(0, 3)
rand2 = random.randint(0, 4)

labels = ["z", "intensity", "number of returns", "red", "green", "blue"]

raw_training_data = laspy.read("training.laz")
training_data = np.append(np.reshape(raw_training_data.z, (-1, 1)), np.reshape(raw_training_data.intensity, (-1, 1)), 1)
training_data = np.append(training_data, np.reshape(raw_training_data.number_of_returns, (-1, 1)), 1)
training_data = np.append(training_data, np.reshape(raw_training_data.red, (-1, 1)), 1)
training_data = np.append(training_data, np.reshape(raw_training_data.green, (-1, 1)), 1)
training_data = np.append(training_data, np.reshape(raw_training_data.blue, (-1, 1)), 1)
training_features = np.array_split(training_data, 4)[rand1]
training_targets = np.array_split(raw_training_data.classification, 4)[rand1]

raw_test_data = laspy.read("test.laz")
test_data = np.append(np.reshape(raw_test_data.z, (-1, 1)), np.reshape(raw_test_data.intensity, (-1, 1)), 1)
test_data = np.append(test_data, np.reshape(raw_test_data.number_of_returns, (-1, 1)), 1)
test_data = np.append(test_data, np.reshape(raw_test_data.red, (-1, 1)), 1)
test_data = np.append(test_data, np.reshape(raw_test_data.green, (-1, 1)), 1)
test_data = np.append(test_data, np.reshape(raw_test_data.blue, (-1, 1)), 1)
test_features = np.array_split(test_data, 5)[rand2]
test_targets = np.array_split(raw_test_data.classification, 5)[rand2]

print(time.time()-t1)

random_forest = RandomForestClassifier()
random_forest.fit(training_features, training_targets)
predictions = random_forest.predict(test_features)

accuracy = accuracy_score(test_targets, predictions)
print("Accuracy:", accuracy)
precision = precision_score(test_targets, predictions, average="binary")
print("Precision: ", precision)
recall = recall_score(test_targets, predictions, average="binary")
print("Recall: ", recall)
f1 = f1_score(test_targets, predictions, average="binary")
print("F1: ", f1)


print(time.time()-t1)
cm = confusion_matrix(test_targets, predictions, labels=None)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
disp.plot()
plt.show()


for i in range(3):
    tree = random_forest.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=labels,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)

