import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


#A1
#to calculate entropy of the dataset
def calculate_entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    H = 0

    for p in probs:
        H -= p * np.log2(p)

    return H


#A2
#to calculate GINI index of the dataset
def calculate_gini(labels):
    values, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    gini = 1 - np.sum(probs ** 2)
    return gini


#A4
#equal width binning for converting continuous features
#divides the feature range into equal sized bins
def equal_width_binning(data, bins=4):
    min_val = np.min(data)
    max_val = np.max(data)
    width = (max_val - min_val) / bins
    binned = []

    for value in data:
        idx = int((value - min_val) / width)
        if idx == bins:
            idx -= 1
        binned.append(idx)

    return np.array(binned)


#A4
#equal frequency binning
#each bin contains approximately the same number of samples
def equal_frequency_binning(data, bins=4):
    sorted_data = np.sort(data)
    bin_size = len(data) // bins
    binned = np.zeros(len(data))

    for i in range(bins):
        start = i * bin_size
        if i == bins - 1:
            end = len(data)
        else:
            end = (i + 1) * bin_size
        values = sorted_data[start:end]

        for v in values:
            binned[data == v] = i

    return binned


#A3 
#function to compute Information Gain
#information gain is used to choose the best feature for splitting
def information_gain(feature, labels):
    total_entropy = calculate_entropy(labels)
    values = np.unique(feature)
    weighted_entropy = 0

    for v in values:
        subset = labels[feature == v]
        weighted_entropy += (len(subset) / len(labels)) * calculate_entropy(subset)
    gain = total_entropy - weighted_entropy
    return gain


#A3
#function to identify the best root feature using information gain
def find_root_feature(X, y):
    best_feature = -1
    best_gain = -1

    for i in range(X.shape[1]):
        gain = information_gain(X[:, i], y)
        if gain > best_gain:
            best_gain = gain
            best_feature = i
    return best_feature


#A5
#build a decision tree model 
def build_decision_tree(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model


#A6
#visualize the trained decision tree
def visualize_tree(model):
    plt.figure(figsize=(10,6))
    plot_tree(model, filled=True)
    plt.title("Decision Tree")
    plt.show()


#A7
#plot decision boundary using two features
def plot_decision_boundary(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()

    return model