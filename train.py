from sklearn.tree import DecisionTreeClassifier
from joblib import dump

def train(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    dump(model, "music.joblib", protocol=5)

if __name__ == "__main__":
    train()