import random
import matplotlib.pyplot as plt
import laspy
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import display

t1 = time.time()

labels = ["z", "number_of_returns", "return_number"]

raw_data = laspy.read("CApoints.laz")
data = raw_data.points[(raw_data.classification != 0) & (raw_data.classification != 1)]

features = np.reshape(data.z, (-1, 1))
for x in range(1, len(labels)):
    features = np.hstack((features, np.reshape(data[labels[x]], (-1, 1))))

gradient_calc = np.gradient(raw_data.xyz, axis=0)
gradient_calc = np.reshape(gradient_calc[:, 2], (-1, 1))
gradient_calc = np.hstack((gradient_calc, np.reshape(raw_data.classification, (-1, 1))))
gradient = gradient_calc[:, 0][(gradient_calc[:, 1] != 0) & (gradient_calc[:, 1] != 1)]
labels.append("z_gradient")
features = np.hstack((features, np.reshape(gradient, (-1,1))))


z_dif_max = np.max(features[:, 0])-features[:, 0]
# z_dif_min = features[:, 0]-np.min(features[:, 0])
# z_dif_average = features[:, 0]-np.mean(features[:, 0])

features = np.hstack((features, np.reshape(z_dif_max, (-1, 1))))
# features = np.hstack((features, np.reshape(z_dif_min, (-1, 1))))
# features = np.hstack((features, np.reshape(z_dif_average, (-1, 1))))

targets = data.classification
training_features, testing_features, training_targets, testing_targets = train_test_split(features, targets, test_size=0.3)

print(time.time()-t1)


random_forest = RandomForestClassifier(n_estimators=200, n_jobs=6)
random_forest.fit(training_features, training_targets)
predictions = random_forest.predict(testing_features)

accuracy = accuracy_score(testing_targets, predictions)
print("Accuracy:", accuracy)
precision = precision_score(testing_targets, predictions, average="micro")
print("Precision: ", precision)
recall = recall_score(testing_targets, predictions, average="micro")
print("Recall: ", recall)
f1 = f1_score(testing_targets, predictions, average="micro")
print("F1: ", f1)


print(time.time()-t1)
disp = ConfusionMatrixDisplay.from_predictions(testing_targets, predictions)
plt.show()

labels.append("dif from max z")
# labels.append("dif from min z")
# labels.append("dif from average z")
# labels.append("gradient")
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

