

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
# Generate a random n-class classification problem.
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.model_selection import train_test_split

classes = 4
X, t = make_classification(100, 5, n_classes=classes,
                           random_state=40, n_informative=2, n_clusters_per_class=1)

print(X, t)
X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=0.50)
model = svm.SVC(kernel='linear', random_state=0, C=1.0)
model.fit(X_train, y_train)

print(X_train)
y = model.predict(X_test)
print(y)
y2 = model.predict(X_train)
score = accuracy_score(y, y_test)
print(score)
score2 = accuracy_score(y2, y_train)
print(score2)

color = ['black' if c == 0 else 'lightgrey' for c in y]
# color = ['black' if c == 0 else 'blue' if c ==
#        1 else 'red' if c == 2 else 'yellow' for c in y]
plt.scatter(X_train[:, 0], X_train[:, 1], c=color)

# Create the hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plot the hyperplane
plt.plot(xx, yy)
plt.axis("on"), plt.show()
res = model.predict([[0.2, 2.4, 3, 2, 1]])
print(res)
