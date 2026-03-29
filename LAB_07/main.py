from pre import build_dataset
from lab7 import *


dataset_path = "Dataset"
X, y = build_dataset(dataset_path)


print("Dataset shape:", X.shape)
X_train, X_test, y_train, y_test = split_data(X, y)


models = get_models()
print("\nModel Comparison Results:\n")

#outputs comparing models
for name, model in models.items():

    train_acc, test_acc, precision, recall, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)

    print(name)
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("-" * 40)


print("\nTuning SVM using RandomizedSearchCV")

best_svm = tune_svm(X_train, y_train)

train_acc, test_acc, precision, recall, f1 = evaluate_model(best_svm, X_train, X_test, y_train, y_test)

print("Best SVM Results")
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#optional shap
print("\nRunning SHAP analysis on Random Forest")
rf_model = models["Random Forest"]
rf_model.fit(X_train, y_train)
shap_analysis(rf_model, X_train)