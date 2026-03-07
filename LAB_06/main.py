from pre import build_dataset
from lab6 import *


dataset_path = "Dataset"
X, y = build_dataset(dataset_path)

print("Dataset shape:", X.shape)

#A1
#calculate entropy of the dataset
entropy_value = calculate_entropy(y)
print("Entropy of dataset:", entropy_value)

#A2
#calculate GINI index of the dataset
gini_value = calculate_gini(y)
print("Gini index of dataset:", gini_value)

#A3
#function to identify the best root feature using information gain
root_feature = find_root_feature(X, y)
print("Best root feature using Information Gain:", root_feature)

#A5
#build a decision tree model
tree_model = build_decision_tree(X, y)
print("Decision tree model constructed.")

#A6
#visualize the decision tree
print("Visualizing Decision Tree")
visualize_tree(tree_model)

#A7
#plot decision boundary using first two features
print("Plotting Decision Boundary using first two features")
plot_decision_boundary(X[:, :2], y)