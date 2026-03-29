import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#split dataset
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


#evaluate model on classification metrics
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    return train_acc, test_acc, precision, recall, f1


#get all models
def get_models():

    models = {}

    models["SVM"] = SVC()
    models["Decision Tree"] = DecisionTreeClassifier()
    models["Random Forest"] = RandomForestClassifier()
    models["AdaBoost"] = AdaBoostClassifier()
    models["Naive Bayes"] = GaussianNB()
    models["MLP"] = MLPClassifier(max_iter=300)

    return models


#A2 tuning
def tune_svm(X_train, y_train):

    param_dist = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }

    search = RandomizedSearchCV(SVC(), param_dist, n_iter=4, cv=3)
    search.fit(X_train, y_train)
    return search.best_estimator_


#optional SHAP analysis 
def shap_analysis(model, X_train):

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train)

    except:
        print("SHAP not installed or error occurred")