import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):  
        self.feature = feature    # which feature this was devided with
        self.threshold = threshold  #which treshold was used to devide
        self.left = left   # left node we pointing to
        self.right = right  # right node we pointing to
        self.value = value  # class aka value of this node (a leaf node meaning pure node) else it's gonna be NONE
        
    def is_leaf_node(self):   #test if it's a leaf node or not, leaf -> returns value(class), else None
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None): #can change
        #stopping criteria :
        self.min_samples_split=min_samples_split  
        self.max_depth=max_depth    #depth of the tree
        self.n_features=n_features   #number of features to use when splitting (random)
        self.root=None  #root of this node

    def fit(self, X, y):
        # we check if the features do not exceed the real number of features we have
        # X.shape gives number of samples [0] and number of features [1] -> shape[1]gives number of features
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)  #returns the root of the tree

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            #n_labels == 1 means it's a leaf node
            leaf_value = self._most_common_label(y) # (107) function below that finds the most common label to asign it to this node
            return Node(value=leaf_value)

        #(JP to 47)
        #BBBBBBBB
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        #random to get a random number of features to use everytime
        

        # If we don't get caught in the stopping criteria : find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs) # (42)(56)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right) #(116)


    def _best_split(self, X, y, feat_idxs):
        #CCCCCCCC
        best_gain = -1
        split_idx, split_threshold = None, None #what we will return

        for feat_idx in feat_idxs: 
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)  #All the things I can split with

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr) #(77)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y) #(103)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold) #(97)splits with column and threshold we have now

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0 #IG = 0 in this case (pure node)
        
        # calculate the weighted avg. entropy of children
        n = len(y) #how many samples in y
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        #EEEEEEEE
        left_idxs = np.argwhere(X_column <= split_thresh).flatten() #which indexes will go tp the left
        right_idxs = np.argwhere(X_column > split_thresh).flatten() # which indx will go to the right
        return left_idxs, right_idxs #(84)

    def _entropy(self, y): 
        #KKKKKKKK 
        hist = np.bincount(y)  #gives a lil histogram
        ps = hist / len(y)   #probabilité = number of occurences/total number of values
        return -np.sum([p * np.log(p) for p in ps if p>0]) #(81)


    def _most_common_label(self, y):
        #00000000
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value  #(37)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X]) 
        #array of values for each entry x in our set X
        


    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value   #will all end up here in a leaf node

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        

