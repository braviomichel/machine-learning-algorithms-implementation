

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D
# Sklearn modules & classes
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC  # support vecto Classification
# split the dataset to two blobs
from sklearn.model_selection import train_test_split
# scale the dataset so that the features are all in the same range
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics  # calculate accuracy ..etc


from sklearn import svm

# dataset which is non linearly seperable
X, Y = make_circles(n_samples=700, noise=0.02)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.7, random_state=1, stratify=Y)
sc = StandardScaler()
sc.fit(X_train)
X_train_t = sc.transform(X_train)
X_test_t = sc.transform(X_test)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.')
plt.show()


X1 = X_train[:, 0].reshape((-1, 1))
X2 = X_train[:, 1].reshape((-1, 1))
X3 = (X1**2 + X2**2)
# stack horizontally
X_train = np.hstack((X_train, X3))
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
axes.scatter(X1, X2, X1**2 + X2**2, c=y_train, depthshade=True)
plt.show()

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
w = svc.coef_
b = svc.intercept_

x1 = X_train[:, 0].reshape((-1, 1))
x2 = X_train[:, 1].reshape((-1, 1))
# reate a rectangular grid out of two given one-dimensional arrays
x1, x2 = np.meshgrid(x1, x2)
x3 = -(w[0][0]*x1 + w[0][1]*x2 + b) / w[0][2]
fig = plt.figure()
axes2 = fig.add_subplot(111, projection='3d')
axes2.scatter(X1, X2, X1**2 + X2**2, c=y_train, depthshade=True)
axes1 = fig.gca(projection='3d')
axes1.plot_surface(x1, x2, x3, alpha=0.01)
plt.show()


svc = SVC(C=1.0, random_state=1, kernel='linear')

svc.fit(X_train_t, y_train)

y_predict = svc.predict(X_test_t)

print("Accuracy score %.3f" % metrics.accuracy_score(y_test, y_predict))

report = metrics.classification_report(y_test, y_predict)

print("Classification report \n ", report)
