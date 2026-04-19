import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


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

        if not os.path.isdir(ppath):
            continue

        for img in os.listdir(ppath):

            try:
                feat = extract_feature(os.path.join(ppath, img))
                X.append(feat)
                y.append(label)
            except:
                continue

        label += 1

    return np.array(X), np.array(y)


#A1 — STACKING CLASSIFIER

def build_stacking():

    base_models = [
        ("svm", SVC(probability=True)),
        ("dt", DecisionTreeClassifier()),
        ("rf", RandomForestClassifier()),
        ("nb", GaussianNB())
    ]

    final_model = LogisticRegression()

    model = StackingClassifier(estimators=base_models,final_estimator=final_model)

    return model


#A2 — PIPELINE

def build_pipeline():

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("stack", build_stacking())
    ])

    return pipe


#A3 — LIME

def run_lime(model, X_train, X_test):

    try:
        from lime.lime_tabular import LimeTabularExplainer

        explainer = LimeTabularExplainer( X_train,mode="classification")

        exp = explainer.explain_instance(X_test[0], model.predict_proba)        

        exp.show_in_notebook(show_table=True)

    except:
        print("LIME not installed or error occurred")



#MAIN EXECUTION

dataset_path = "Dataset"

X, y = build_dataset(dataset_path)

print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


stack_model = build_stacking()

stack_model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, stack_model.predict(X_train))
test_acc = accuracy_score(y_test, stack_model.predict(X_test))

print("\nStacking Results")
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)


# A2 — Pipeline
pipeline = build_pipeline()

pipeline.fit(X_train, y_train)

train_acc = accuracy_score(y_train, pipeline.predict(X_train))
test_acc = accuracy_score(y_test, pipeline.predict(X_test))

print("\nPipeline Results")
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)


# A3 — LIME
print("\nRunning LIME Explanation")
run_lime(pipeline, X_train, X_test)
