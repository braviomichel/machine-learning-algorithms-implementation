
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(21)

N = 1000


def makeData(x):
    r = [a/10 for a in x]
    # create arrays filled with random samples which are from a uniform distribution
    y = np.sin(x)+np.random.uniform(-.5, .2, len(x))
    return np.array(y+r)


x = [i/100 for i in range(N)]
y = makeData(x)
x = np.array(x).reshape(-1, 1)

plt.scatter(x, y, s=5, color="blue")
plt.show()
svr = SVR().fit(x, y)
print(svr)

yfit = svr.predict(x)

plt.scatter(x, y, s=5, color="blue", label="original")
plt.plot(x, yfit, lw=2, color="red", label="fitted")
plt.legend()
plt.show()
# r squared statistical measure of fit that indicates how much variation of a
# dependent variable is explained by the independent variable(s) in a regression model.
score = svr.score(x, y)
print("R-squared:", score)
# erreur quadratique moyenne.  the square root of the mean square
print("MSE:", mean_squared_error(y, yfit))
