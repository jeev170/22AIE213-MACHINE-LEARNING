import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error, r2_score)


#image prepocessing functions

#function for splitting images into horizontal strips
def split_horizontal(image, n_strips):
    h, w, c = image.shape
    sh = h // n_strips
    strips = []

    for i in range(n_strips):
        part = image[i*sh:(i+1)*sh, :]
        strips.append(part)

    return strips


#here, dividing strips into 15 blocks for rgb
def split_blocks(strip):
    h, w, c = strip.shape
    blocks = []

    bh = h // 3
    bw = w // 5

    for r in range(3):
        for col in range(5):
            block = strip[r*bh:(r+1)*bh, col*bw:(col+1)*bw]
            blocks.append(block)

    return blocks


#calculating mean and variance for rgb blocks
def rgb_stats(block):
    m = np.mean(block, axis=(0,1))
    v = np.var(block, axis=(0,1))
    return np.concatenate((m,v)) #combining the values so that each block has 6 features


#each image will give 90 features
def extract_image_features(path):
    img = cv2.imread(path)
    strips = split_horizontal(img,5) #5 strips
    features = []

    for s in strips:
        blocks = split_blocks(s)
        vec = []
        for b in blocks:
            f = rgb_stats(b)
            vec.extend(f) #adding rcb stats to feature vector
        features.append(vec)

    return np.array(features)



#building the dataset
def build_dataset(folder):
    X = [] #feature matrix
    y = [] #class labels
    label = 0

    for person in os.listdir(folder):
        ppath = os.path.join(folder,person)

        for imgname in os.listdir(ppath):
            ipath = os.path.join(ppath,imgname)
            feats = extract_image_features(ipath)

            #storing features and labels
            for f in feats:
                X.append(f)
                y.append(label)

        label += 1

    return np.array(X), np.array(y)

#A1
#evaluating classsification for confusion matrix
def evaluate_classification(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro') #precision
    recall = recall_score(y_true, y_pred, average='macro') #recall
    f1 = f1_score(y_true, y_pred, average='macro') #f1 score
    return cm, precision, recall, f1


#A2
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred) #mean squared error
    rmse = np.sqrt(mse) #root mean squared error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 #mean absolute percentage error
    r2 = r2_score(y_true, y_pred) #r-squared score
    return mse, rmse, mape, r2

#A3
def generate_training_points():
    X = np.random.uniform(1, 10, (20, 2)) #generate random points
    y = np.array([0 if x[0] + x[1] < 10 else 1 for x in X]) #asiigning class labels based on feature sum
    return X, y

#A4
def generate_test_grid():
    xs = np.arange(0, 10, 0.1) #generate test grid points
    grid = np.array([[x, y] for x in xs for y in xs]) #cartesian product
    return grid

#A5
def classify_grid(X_train, y_train, grid, k):
    model = KNeighborsClassifier(n_neighbors=k) #initializing knn model
    model.fit(X_train, y_train)
    return model.predict(grid) #predicting class labels 

#A6
def select_two_features_two_classes(X, y, f1=0, f2=1, c1=0, c2=1):
    mask = (y == c1) | (y == c2) #only 2 classes
    X_sel = X[mask][:, [f1, f2]] #selecting two features
    y_sel = y[mask] #selecting corresponding labels
    return X_sel, y_sel

#A7
def tune_k(X_train, y_train):
    param_grid = {'n_neighbors': list(range(1, 16))} #using diff k values
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5) #5 fold cross validation
    grid.fit(X_train, y_train)
    return grid.best_params_['n_neighbors'] #return best k value


