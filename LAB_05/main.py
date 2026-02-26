from pre import build_dataset
from lab5 import *
from sklearn.model_selection import train_test_split

dataset_path = "Dataset"

X, y = build_dataset(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#A1
feature_index = 0 #selecting one numerical atribute
single_model = train_single_attribute_lr(X_train, feature_index)

#predicting on train and test data
train_true, train_pred = predict_single_attribute_lr(single_model, X_train, feature_index)
test_true, test_pred = predict_single_attribute_lr(single_model, X_test, feature_index)

#A2
#calculating regression evaluation metrics
train_metrics = regression_scores(train_true, train_pred)
test_metrics = regression_scores(test_true, test_pred)

print("A2 Train Metrics:", train_metrics)
print("A2 Test Metrics:", test_metrics)

#A3
#training regression model using multiple attributes
multi_model = train_multi_attribute_lr(X_train, feature_index)
multi_true, multi_pred = predict_multi_attribute_lr(multi_model, X_test, feature_index)
multi_metrics = regression_scores(multi_true, multi_pred)

print("A3 Metrics:", multi_metrics)

#a4
#k=2, cluster labels and cluster centers
kmeans_model = perform_kmeans(X_train, 2)
labels, centers = get_cluster_details(kmeans_model)

print("Cluster Centers:")
print(centers)

#A5
#calculating clustering evaluation scores
sil, ch, db = clustering_metrics(X_train, labels)

print("Silhouette Score:", sil)
print("Calinski Harabasz Score:", ch)
print("Davies Bouldin Index:", db)

#A6
#evaluating clustering performance for diff K values
k_values = range(2, 10)
sil_scores, ch_scores, db_scores = evaluate_multiple_k(X_train, k_values)

plot_k_analysis(k_values, sil_scores, ch_scores, db_scores)

#A7
#determining optimal K using elbow method
k_range = range(2, 20)
distortions = elbow_method(X_train, k_range)

plot_elbow(k_range, distortions)