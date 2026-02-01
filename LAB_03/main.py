from lab3 import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

root = r"C:\Users\jeevi\Downloads\ML_LAB3\Lab3_Dataset"

X, y = build_dataset(root)

#A1

A = X[0]
B = X[1]

print("Manual Dot:", dot_product(A,B))
print("NumPy Dot:", np.dot(A,B))

print("Manual Norm:", euclid_norm(A))
print("NumPy Norm:", np.linalg.norm(A))


#A2

c0 = X[y==0]
c1 = X[y==1]

cent0 = compute_mean(c0)
cent1 = compute_mean(c1)

print("Inter-class distance:",
      np.linalg.norm(cent0-cent1))

print("Avg Spread Class0:",
      np.mean(compute_std(c0)))

print("Avg Spread Class1:",
      np.mean(compute_std(c1)))


#A3

feature = X[:,0]

hist_vals, bins = compute_histogram(feature)

print("Histogram counts:", hist_vals)
print("Bins:", bins)

print("Mean:", feature_mean(feature))
print("Variance:", feature_variance(feature))

plt.hist(feature, bins=10)
plt.title("A3 Histogram")
plt.show()


#A4


p_vals = list(range(1,11))
dists = []

for p in p_vals:
    d = minkowski_distance(A,B,p)
    dists.append(d)
    print(f"p={p}, distance={d}")

plt.plot(p_vals, dists, marker='o')
plt.title("A4 Minkowski Plot")
plt.show()


#A5

my_m, sp_m = compare_minkowski(A,B,3)

print("My Minkowski:", my_m)
print("SciPy Minkowski:", sp_m)


#A6

X_train, X_test, y_train, y_test = split_two_classes(X,y)

print("Train size:", len(X_train))
print("Test size:", len(X_test))


#A7

knn_model = train_knn(X_train, y_train, 3)
print("kNN trained (k=3)")


#A8

print("Test Accuracy:",
      test_knn_accuracy(knn_model,X_test,y_test))


#A9

y_pred = predict_labels(knn_model,X_test)

print("Predictions:", y_pred[:10])
print("Actual:", y_test[:10])


#A10

my_preds = my_knn_batch(X_train,y_train,X_test,3)

print("Own kNN Accuracy:",np.mean(my_preds==y_test))


#A11

k_vals = list(range(1,12))

accs = compute_knn_accuracies(X_train,y_train,X_test,y_test,k_vals)

print("k=1 accuracy:", accs[0])
print("k=3 accuracy:", accs[2])

plt.plot(k_vals,accs,marker='o')
plt.title("A11 Accuracy vs k")
plt.show()


#A12

cm = confusion_matrix(y_test,y_pred)

plt.imshow(cm)
plt.title("A12 Confusion Matrix")
plt.colorbar()
plt.show()


#A13

p = compute_precision(cm)
r = compute_recall(cm)

print("Accuracy:", compute_accuracy(cm))
print("Precision:", p)
print("Recall:", r)
print("F1:", compute_fbeta(p,r))


#A14

W = train_matrix_inversion(X_train,y_train)

mi_preds = predict_matrix_inversion(X_test,W)

print("Matrix Inversion Accuracy:", np.mean(mi_preds==y_test))

print("kNN Accuracy:", test_knn_accuracy(knn_model,X_test,y_test))


#O1 

d = generate_normal_data()

plt.hist(d,bins=30,density=True)
plt.title("O1 Normal Distribution")
plt.show()


#O2

print(test_knn_metrics(X_train,y_train, X_test,y_test))


#O3

auc_val = plot_auroc(knn_model, X_test, y_test)

print("AUC:", auc_val)
