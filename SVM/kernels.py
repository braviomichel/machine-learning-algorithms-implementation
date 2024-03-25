

from sklearn.metrics import accuracy_score
import numpy as np
# Generate a random n-class classification problem.
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.model_selection import train_test_split

classes = 4
X, t = make_classification(100, 5, n_classes=classes,
                           random_state=40, n_informative=2, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=0.50)
print("#########linear######")
model = svm.SVC(kernel='linear', random_state=0, C=1.0)
model.fit(X_train, y_train)
y = model.predict(X_test)
print("predict y", y)
y2 = model.predict(X_train)
score = accuracy_score(y, y_test)
print("score :", score)
score2 = accuracy_score(y2, y_train)
print("score y2", score2)

res = model.predict([[0.2, 2.4, 3, 2, 1]])
print("predict :", res)

print("#########RBF#######")
model = svm.SVC(kernel='rbf', random_state=0, C=1.0)
model.fit(X_train, y_train)
y = model.predict(X_test)
print("predict y", y)
y2 = model.predict(X_train)
score = accuracy_score(y, y_test)
print("score :", score)
score2 = accuracy_score(y2, y_train)
print("score y2", score2)

res = model.predict([[0.2, 2.4, 3, 2, 1]])
print("predict :", res)
print("#########POLY#######")
model = svm.SVC(kernel='poly', random_state=0, C=1.0)
model.fit(X_train, y_train)
y = model.predict(X_test)
print("predict y", y)
y2 = model.predict(X_train)
score = accuracy_score(y, y_test)
print("score :", score)
score2 = accuracy_score(y2, y_train)
print("score y2", score2)

res = model.predict([[0.2, 2.4, 3, 2, 1]])
print("predict :", res)
print("#########SIGMOID#######")
model = svm.SVC(kernel='sigmoid', random_state=0, C=1.0)
model.fit(X_train, y_train)
y = model.predict(X_test)
print("predict y", y)
y2 = model.predict(X_train)
score = accuracy_score(y, y_test)
print("score :", score)
score2 = accuracy_score(y2, y_train)
print("score y2", score2)

res = model.predict([[0.2, 2.4, 3, 2, 1]])
print("predict :", res)
