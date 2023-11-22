import random
import matplotlib.pyplot as plt
import laspy
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay


t1 = time.time()
rand1 = random.randint(0, 3)
rand2 = random.randint(0, 4)

raw_training_data = laspy.read("training.laz")
training_features = np.array_split(raw_training_data.xyz, 4)[rand1]
training_targets = np.array_split(raw_training_data.classification, 4)[rand1]

raw_test_data = laspy.read("test.laz")
test_features = np.array_split(raw_test_data.xyz, 5)[rand2]
test_targets = np.array_split(raw_test_data.classification, 5)[rand2]

print(time.time()-t1)

random_forest = RandomForestClassifier(n_jobs=6)
random_forest.fit(training_features, training_targets)
predictions = random_forest.predict(test_features)

accuracy = accuracy_score(test_targets, predictions)
print("Accuracy:", accuracy)
precision = precision_score(test_targets, predictions)
print("Precision: ", precision)
recall = recall_score(test_targets, predictions)
print("Recall: ", recall)

cm = confusion_matrix(test_targets, predictions, labels=None)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
disp.plot()
plt.show()
print(time.time()-t1)


#if __name__ == '__main__':
# a

