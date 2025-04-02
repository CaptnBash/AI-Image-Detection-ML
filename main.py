import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from joblib import  load

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=["genre"])
y = music_data["genre"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = load("music.joblib")

tree.export_graphviz(model, out_file="music.dot", feature_names=["age", "gender"], 
                     class_names=sorted(y.unique()), label="all", rounded=True, filled=True)

predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
print(score)