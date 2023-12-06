import math
import matplotlib.pyplot as plt
import laspy
import numpy as np
import time
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import binned_statistic_2d
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import display


def main(filename):
    # t1 = time.time()
    # Read in las/laz file and remove unclassified points from the dataset
    raw_data = laspy.read(filename)
    data = raw_data.points[(raw_data.classification != 0) & (raw_data.classification != 1)]

    # Get the z values and distance in x and y directions; assumes that xy is in UTM (universal transverse mercator)
    z_values = data.z
    x_bins = data.X / 100
    y_bins = data.Y / 100
    x_distance = math.ceil(np.max(x_bins)-np.min(x_bins))
    y_distance = math.ceil(np.max(y_bins)-np.min(y_bins))

    # take statistics using z values over 1 meter by 1 meter squares based on xy position
    mean_bins = binned_statistic_2d(x_bins, y_bins, z_values, statistic="mean", bins=(x_distance, y_distance), expand_binnumbers=True)
    max_bins = binned_statistic_2d(x_bins, y_bins, z_values, statistic="max", bins=(x_distance, y_distance), expand_binnumbers=True)
    min_bins = binned_statistic_2d(x_bins, y_bins, z_values, statistic="min", bins=(x_distance, y_distance), expand_binnumbers=True)
    std_bins = binned_statistic_2d(x_bins, y_bins, z_values, statistic="std", bins=(x_distance, y_distance), expand_binnumbers=True)

    # unused statistics for 1x1
    # rough_bins = binned_statistic_2d(x_bins, y_bins, z_values, statistic=roughness, bins=(x_distance, y_distance), expand_binnumbers=True)
    # mm_bins = binned_statistic_2d(x_bins, y_bins, z_values, statistic=maxmin_ratio, bins=(x_distance, y_distance), expand_binnumbers=True)
    # m2m_bins = binned_statistic_2d(x_bins, y_bins, z_values, statistic=min2max_ratio, bins=(x_distance, y_distance), expand_binnumbers=True)

    # combine bin statistics into single array
    combined_bins = np.dstack((mean_bins.statistic, max_bins.statistic))
    combined_bins = np.dstack((combined_bins, min_bins.statistic))
    combined_bins = np.dstack((combined_bins, std_bins.statistic))
    # combined_bins = np.dstack((combined_bins, rough_bins.statistic))
    # combined_bins = np.dstack((combined_bins, mm_bins.statistic))
    # combined_bins = np.dstack((combined_bins, m2m_bins.statistic))

    # binned_statistic_2d binnumber gives bin coordinates in xy starting from 1
    bin_coords = mean_bins.binnumber-1

    # create feature vector from combined bins
    features_from_bins = [combined_bins[bin_coords[0][i]][bin_coords[1][i]] for i in range(len(z_values))]
    features_from_bins = np.array(features_from_bins)

    # repeat for 3 meter by 3 meter bins
    x_distance_3x3 = math.ceil(x_distance / 3)
    y_distance_3x3 = math.ceil(y_distance / 3)
    mean_bins_3x3 = binned_statistic_2d(x_bins, y_bins, z_values, statistic="mean", bins=(x_distance_3x3, y_distance_3x3), expand_binnumbers=True)
    max_bins_3x3 = binned_statistic_2d(x_bins, y_bins, z_values, statistic="max", bins=(x_distance_3x3, y_distance_3x3), expand_binnumbers=True)
    min_bins_3x3 = binned_statistic_2d(x_bins, y_bins, z_values, statistic="min", bins=(x_distance_3x3, y_distance_3x3), expand_binnumbers=True)
    std_bins_3x3 = binned_statistic_2d(x_bins, y_bins, z_values, statistic="std", bins=(x_distance_3x3, y_distance_3x3), expand_binnumbers=True)
    rough_bins_3x3 = binned_statistic_2d(x_bins, y_bins, z_values, statistic=min2max_ratio, bins=(x_distance_3x3, y_distance_3x3), expand_binnumbers=True)
    combined_bins_3x3 = np.dstack((mean_bins_3x3.statistic, max_bins_3x3.statistic))
    combined_bins_3x3 = np.dstack((combined_bins_3x3, min_bins_3x3.statistic))
    combined_bins_3x3 = np.dstack((combined_bins_3x3, std_bins_3x3.statistic))
    combined_bins_3x3 = np.dstack((combined_bins_3x3, rough_bins_3x3.statistic))

    bin_coords_3x3 = mean_bins_3x3.binnumber-1
    features_from_bins_3x3 = [combined_bins_3x3[bin_coords_3x3[0][i]][bin_coords_3x3[1][i]] for i in range(len(z_values))]
    features_from_bins_3x3 = np.array(features_from_bins_3x3)

    # create feature vector of z_values and bin values
    features = np.reshape(z_values, (-1, 1))
    features = np.hstack((features, features_from_bins))
    features = np.hstack((features, features_from_bins_3x3))
    # feature_labels should include all features in order
    feature_labels = ["z", "mean", "max", "min", "std", "3x3 mean", "3x3 max", "3x3 min", "3x3 std", "3x3 roughness"]

    # add difference from the absolute maximum z value as a feature
    z_dif_max = np.max(features[:, 0])-features[:, 0]
    features = np.hstack((features, np.reshape(z_dif_max, (-1, 1))))
    feature_labels.append("dif from max z")

    # Get classification of points to train on
    targets = data.classification
    # Split features and targets into training and test sets
    training_features, testing_features, training_targets, testing_targets = train_test_split(features, targets, test_size=0.2)

    # print(time.time()-t1) # data processing time

    # Run a random forest classifier; n_estimators should be 200-250; adjust n_jobs based on cores/threads available
    random_forest = RandomForestClassifier(n_estimators=250, n_jobs=6)
    random_forest.fit(training_features, training_targets)
    predictions = random_forest.predict(testing_features)

    # print(time.time() - t1) # total run time

    # Print accuracy, precision, recall and f1 scores; values will be between 0 and 1, higher values are better
    accuracy = accuracy_score(testing_targets, predictions)
    print("Accuracy:", accuracy)
    precision = precision_score(testing_targets, predictions, average="micro")
    print("Precision: ", precision)
    recall = recall_score(testing_targets, predictions, average="micro")
    print("Recall: ", recall)
    f1 = f1_score(testing_targets, predictions, average="micro")
    print("F1: ", f1)

    # Create and display a confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(testing_targets, predictions)
    plt.show()

    # Calculate and show feature importance
    importances = random_forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_labels)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    fig.tight_layout()
    plt.show()

    # Display part of the first three decision trees in the random forest
    for i in range(3):
        tree = random_forest.estimators_[i]
        dot_data = export_graphviz(tree,
                                   feature_names=feature_labels,
                                   filled=True,
                                   max_depth=3,
                                   impurity=False,
                                   proportion=True)
        graph = graphviz.Source(dot_data)
        display(graph)


def roughness(bin):
    return np.max(bin) - np.min(bin)


def maxmin_ratio(bin):
    return np.max(bin) / np.min(bin)


def min2max_ratio(bin):
    return (np.min(bin) ** 2) / np.max(bin)


# def local_from_max(bin):
#     return np.max(bin) - bin
if __name__ == '__main__':
    main("points.laz") # Switch filename here