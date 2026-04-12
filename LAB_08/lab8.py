import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


#PREPROCESSING (ResNet + 5 strips)

model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def split_horizontal(img, n=5):
    h, w = img.shape[:2]
    sh = h // n
    return [img[i*sh:(i+1)*sh, :] for i in range(n)]


def extract_feature(path):
    img = np.array(Image.open(path).convert("RGB"))
    strips = split_horizontal(img, 5)

    features = []

    for s in strips:
        s = Image.fromarray(s)
        s = transform(s).unsqueeze(0)

        with torch.no_grad():
            f = model(s)

        features.extend(f.view(-1).numpy())

    return np.array(features)


def build_dataset(folder):
    X, y = [], []
    label = 0

    for person in os.listdir(folder):
        ppath = os.path.join(folder, person)

        for img in os.listdir(ppath):
            try:
                feat = extract_feature(os.path.join(ppath, img))
                X.append(feat)
                y.append(label)
            except:
                continue

        label += 1

    return np.array(X), np.array(y)


#A1 FUNCTIONS

def step(x): return 1 if x >= 0 else 0
def bipolar(x): return 1 if x >= 0 else -1
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_deriv(x): return x*(1-x)
def relu(x): return max(0,x)

#PERCEPTRON

def train_perceptron(X, y, activation, lr=0.05, epochs=1000):

    w = np.array([0.2, -0.75])
    b = 10

    errors = []

    for epoch in range(epochs):

        total_error = 0

        for i in range(len(X)):

            net = np.dot(X[i], w) + b
            y_pred = activation(net)

            e = y[i] - y_pred

            w += lr * e * X[i]
            b += lr * e

            total_error += e**2

        errors.append(total_error)

        if total_error <= 0.002:
            break

    return w, b, errors



#A2 AND GATE

X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

w,b,err = train_perceptron(X_and, y_and, step)

plt.plot(err)
plt.title("A2 AND Error")
plt.show()

#A3 DIFFERENT ACTIVATIONS

for func in [bipolar, sigmoid, relu]:
    _,_,err = train_perceptron(X_and, y_and, func)
    print(func.__name__, "epochs:", len(err))


#A4 LEARNING RATE

lrs = [0.1,0.2,0.3,0.4,0.5]
iters = []

for lr in lrs:
    _,_,err = train_perceptron(X_and, y_and, step, lr)
    iters.append(len(err))

plt.plot(lrs, iters)
plt.xlabel("Learning Rate")
plt.ylabel("Iterations")
plt.title("A4 Learning Rate")
plt.show()


#A5 XOR (Perceptron FAIL)

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

_,_,err = train_perceptron(X_xor, y_xor, step)

print("A5 XOR final error:", err[-1])


#A6 DATASET PERCEPTRON


X_data = np.array([
    [20,6,2,386],
    [16,3,6,289],
    [27,6,2,393],
    [19,1,2,110],
    [24,4,2,280]
])

y_data = np.array([1,1,1,0,1])

train_perceptron(X_data, y_data, sigmoid)


#A9 BACKPROP (XOR)

def train_mlp(X, y, lr=0.05, epochs=10000):

    np.random.seed(0)

    W1 = np.random.uniform(-0.5,0.5,(2,2))
    W2 = np.random.uniform(-0.5,0.5,(2,1))

    for epoch in range(epochs):

        for i in range(len(X)):

            x = X[i].reshape(1,-1)
            target = y[i]

            h = sigmoid(np.dot(x, W1))
            o = sigmoid(np.dot(h, W2))

            error = target - o

            d_out = error * sigmoid_deriv(o)
            d_hidden = d_out.dot(W2.T) * sigmoid_deriv(h)

            W2 += lr * h.T.dot(d_out)
            W1 += lr * x.T.dot(d_hidden)

    return W1, W2


y_xor2 = np.array([[0],[1],[1],[0]])
train_mlp(X_xor, y_xor2)


#A10 TWO OUTPUT

y_encoded = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0]
])

#reuse same MLP structure if needed


#A11 sklearn MLP

mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)

mlp.fit(X_and, y_and)
print("A11 AND:", mlp.predict(X_and))

mlp.fit(X_xor, y_xor)
print("A11 XOR:", mlp.predict(X_xor))


#A12 PROJECT DATASET (ResNet features)

X, y = build_dataset("Dataset")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)

mlp.fit(X_train, y_train)

print("A12 Train Accuracy:", mlp.score(X_train, y_train))
print("A12 Test Accuracy:", mlp.score(X_test, y_test))