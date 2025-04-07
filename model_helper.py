import os
from pickle import dump, load
from sklearn.ensemble import RandomForestClassifier

from settings import MODEL_FILE

def get_model(X_train, y_train, save=True) -> RandomForestClassifier:
    """
    save: wether to store the model in a file for future use
    """
    if not os.path.exists(MODEL_FILE):
        clf = train_model(X_train, y_train)
        if save:
            save_model(clf)
        return clf
    else:
        return load_model

def train_model(X_train, y_train) -> RandomForestClassifier:
    print("\ntraining model...")
    return RandomForestClassifier().fit(X_train, y_train)

def load_model() -> RandomForestClassifier:
    print("\nloading the model...")
    with open(MODEL_FILE, "rb") as f:
        clf : RandomForestClassifier = load(f)
    return clf

def save_model(clf : RandomForestClassifier):
    print("\nsaving the model...")
    with open(MODEL_FILE, "wb") as file:
        dump(clf, file, protocol=5)