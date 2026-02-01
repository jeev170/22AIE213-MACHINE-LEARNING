import numpy as np
#for imgage processing
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.spatial.distance import minkowski as scipy_minkowski



#image prepocessing functions

#function for splitting images into vertical strips
def split_vertical(image, n_strips):
    h, w, c = image.shape #height, width, channels of image
    sw = w // n_strips
    strips = []

    for i in range(n_strips):
        part = image[:, i*sw:(i+1)*sw]
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
            block = strip[r*bh:(r+1)*bh,
                          col*bw:(col+1)*bw]
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
    strips = split_vertical(img,5) #5 strips
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
#dot product
def dot_product(a,b):
    s = 0
    for i in range(len(a)):
        s += a[i]*b[i] #multiply matching index elements
    return s

#euclidean norm
def euclid_norm(a):
    s = 0
    for i in range(len(a)):
        s += a[i]*a[i]
    return np.sqrt(s)


#A2

#mean
def compute_mean(data):
    return np.sum(data, axis=0) / len(data)

#variance
def compute_variance(data):
    m = compute_mean(data)
    return np.sum((data - m)**2, axis=0) / len(data)

#standard deviation
def compute_std(data):
    return np.sqrt(compute_variance(data))


#A3
#ihstogram of feature
def compute_histogram(feature, bins=10):
    hist_vals, bin_edges = np.histogram(feature, bins=bins)
    return hist_vals, bin_edges

#mean of feature
def feature_mean(feature):
    return np.mean(feature)

#variance of feature
def feature_variance(feature):
    return np.var(feature)


#A4
#minkowski distance from p=1 to p=10
def minkowski_distance(a, b, p):
    s = 0
    for i in range(len(a)):
        s += abs(a[i] - b[i]) ** p
    return s ** (1/p)

#A5
#compare minkowski distance
def compare_minkowski(a, b, p):
    my_val = minkowski_distance(a, b, p)
    scipy_val = scipy_minkowski(a, b, p)
    return my_val, scipy_val

#A6
def split_two_classes(X, y, class_a=0, class_b=1, test_size=0.3):
    
    #masks for getting only two classes
    mask = (y == class_a) | (y == class_b)
    
    X_two = X[mask]
    y_two = y[mask]
    
    #splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_two, y_two, test_size=test_size, random_state=42)    
    return X_train, X_test, y_train, y_test

#A7
#training knn classifier
def train_knn(X_train, y_train, k=3):
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    return model

#A8
#testing knn model accuracy
def test_knn_accuracy(model, X_test, y_test):
    
    acc = model.score(X_test, y_test)
    
    return acc

#A9
#studying prediction behaviour of classifier for test vector
def predict_labels(model, X_test):
    
    preds = model.predict(X_test)
    
    return preds

#now, for a given vector
def predict_single(model, test_vector):
    
    pred = model.predict([test_vector])
    
    return pred[0]

#A10
#own implementation of kNN
def euclidean_distance(a, b):
    s = 0
    for i in range(len(a)):
        s += (a[i] - b[i])**2
    return np.sqrt(s)

def my_knn_predict(X_train, y_train, test_vec, k=3):
    
    distances = []
    
    for x, label in zip(X_train, y_train):
        d = euclidean_distance(test_vec, x)
        distances.append((d, label))
    
    distances.sort(key=lambda x: x[0])
    
    k_labels = [label for _, label in distances[:k]]
    
    return Counter(k_labels).most_common(1)[0][0]

def my_knn_batch(X_train, y_train, X_test, k=3):
    
    preds = []
    
    for test_vec in X_test:
        p = my_knn_predict(X_train, y_train, test_vec, k)
        preds.append(p)
    
    return np.array(preds)

#A11
#k=1 and k=3 accuracies
def compute_knn_accuracies(X_train, y_train, X_test, y_test, k_values):
    
    accs = []
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        accs.append(acc)
    
    return accs

#A12
#confusion matrix plotting
def plot_confusion_matrix_image(y_true, y_pred, title="Confusion Matrix"):
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    return cm
#computing precission, recall, f1 score for train and test
def compute_metrics(cm):
    
    precision_list = []
    recall_list = []
    
    for i in range(len(cm)):
        
        tp = cm[i,i]
        fp = sum(cm[:,i]) - tp
        fn = sum(cm[i,:]) - tp
        
        p = tp/(tp+fp+1e-9)
        r = tp/(tp+fn+1e-9)
        
        precision_list.append(p)
        recall_list.append(r)
    
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = 2*precision*recall/(precision+recall+1e-9)
    
    return precision, recall, f1
#based on outputs ours is regular fit model

#A13

#computing a12 metrics on our own 
def compute_accuracy(cm):
    
    correct = 0
    total = 0
    
    for i in range(len(cm)):
        correct += cm[i,i]
        for j in range(len(cm)):
            total += cm[i,j]
    
    return correct / total

def compute_precision(cm):
    
    precisions = []
    
    for i in range(len(cm)):
        tp = cm[i,i]
        fp = 0
        
        for j in range(len(cm)):
            if j != i:
                fp += cm[j,i]
        
        if tp + fp == 0:
            precisions.append(0)
        else:
            precisions.append(tp / (tp + fp))
    
    return sum(precisions) / len(precisions)


def compute_recall(cm):
    
    recalls = []
    
    for i in range(len(cm)):
        tp = cm[i,i]
        fn = 0
        
        for j in range(len(cm)):
            if j != i:
                fn += cm[i,j]
        
        if tp + fn == 0:
            recalls.append(0)
        else:
            recalls.append(tp / (tp + fn))
    
    return sum(recalls) / len(recalls)

def compute_fbeta(precision, recall, beta=1):
    
    beta_sq = beta * beta
    
    if precision + recall == 0:
        return 0
    
    return (1 + beta_sq) * precision * recall / (
            beta_sq * precision + recall
           )

#A14
#comparing performace of knn classifier with matrix inversion method
def train_matrix_inversion(X_train, y_train):
    
    #pseudo-inverse
    X_pinv = np.linalg.pinv(X_train)
    
    #weight vector
    W = np.dot(X_pinv, y_train)

    
    return W

def predict_matrix_inversion(X_test, W):
    
    preds = np.dot(X_test, W)
    
    #convert to class labels(0 or 1)
    preds = np.round(preds)
    
    return preds


#O1 

def generate_normal_data(mean=0, std=1, size=1000):
    return np.random.normal(mean, std, size)

def plot_normal_and_hist(data):
    
    #histogram
    plt.hist(data, bins=30, density=True, alpha=0.6)
    
    #theoretical plot
    mean = np.mean(data)
    std = np.std(data)
    
    x = np.linspace(min(data), max(data), 100)
    y = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)
    
    plt.plot(x, y)
    plt.title("Normal Distribution vs Histogram")
    plt.show()


#O2 

def test_knn_metrics(X_train, y_train, X_test, y_test):
    #different distance metric for kNN classifier
    metrics = ['euclidean', 'manhattan', 'chebyshev']
    results = {}
    
    for m in metrics:
        model = KNeighborsClassifier(
            n_neighbors=3,
            metric=m
        )
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        results[m] = acc
    
    return results
#all 3 distances gave same classification accuracy 0.8

#O3 
#AUROC plot
from sklearn.metrics import roc_curve, auc

def plot_auroc(model, X_test, y_test):
    
    probs = model.predict_proba(X_test)
    
    #probability of class 1
    probs_class1 = probs[:,1]
    
    fpr, tpr, _ = roc_curve(y_test, probs_class1)
    
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()
    
    return roc_auc
