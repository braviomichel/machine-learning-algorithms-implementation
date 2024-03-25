from DecisionTree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []  #where to keep all the trees

    def fit(self, X, y):   #creating a forest of trees
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)   #trained on different sets
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True) #true here bc we drop some of the samples and use some again
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # [[1,0,0,0,1],[0,1,0,1,1],[samples of each tree],....]
        tree_preds = np.swapaxes(predictions, 0, 1)
        # [[1 from 1st tree, 0 from 2nd tree,...],[0 from 1st tree, 1 from 2nd tree,...]]
        #all predictions from the same sample of different trees put in the same inner list 
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions