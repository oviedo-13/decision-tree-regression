# ---------------------------------------------------------
# Author: Antonio Oviedo Paredes
# Date: 28 / 08 / 2023
# Implementation of a decision tree for regresion.
# ---------------------------------------------------------

class Node:
    '''Class to represent a node o a decision tree.'''
    def __init__(self, feature=None, threshold=None, left=None, right=None, avg=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.avg = avg

    def is_leaf(self):
        return self.avg is not None


class DecisionTree:
    '''Class that implements a decision tree for regression.'''

    def __init__(self, min_samples_split=20, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.n_rndm_feats = None

    def train(self, X, y, n_rndm_feats=None):
        '''
        Function to train the model.

        Parameters:
            X: pandas.DataFrame -> Input variables.
            y: pandas.Series -> Output variable.
            n_rndm_feats: int -> Number of X variables used to find a new best feature to split the data.
        '''
        if n_rndm_feats:
            self.n_rndm_feats = n_rndm_feats
        else:
            self.n_rndm_feats = X.shape[1]

        self.root = self._grow_tree(X, y)

    def predict(self, X):
        '''
        Function to predict new values of y from a trained tree.

        Parameters:
            X: pandas.Series -> Input variables to which you want to predict an output value.

        Returns:
            numpy.ndarray -> Array with the predicted values.
        '''

        return np.array([self._traverse_tree(X.loc[x], self.root) for x in X.index])
    
    def show(self):
        '''
        Function to print the structure of the trained tree.

        Returns:
            Dict[] -> Trained tree representation
        '''

        repr = self._build_representation(self.root)
        print(repr)
        return repr

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]

        # Stop tree growth
        if (depth >= self.max_depth or self.min_samples_split > n_samples):
            avg = y.mean()
            # Leaf node
            return Node(avg=avg)

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)

        # Stop growth if there is no split
        if (not best_feature or not best_threshold):
            avg = y.mean()
            return Node(avg=avg)

        # Create child nodes
        left_vals = X.loc[X[best_feature] <= best_threshold]
        right_vals = X.loc[X[best_feature] > best_threshold]
        left = self._grow_tree(left_vals, y.loc[X[best_feature] <= best_threshold], depth+1)
        right = self._grow_tree(right_vals, y.loc[X[best_feature] > best_threshold], depth+1)

        return Node(best_feature, best_threshold, left, right)


    def _find_best_split(self, X, y):
        best_sse = float("inf")
        best_feature = None
        best_threshold = None

        # Choose features to evaluate
        chosen_features = np.random.choice(X.columns, self.n_rndm_feats, replace=False)

        # Find best feature and threshold
        for feature in chosen_features:
            threshold, sse = self._find_best_threshold(X[feature], y)

            if sse < best_sse:
                best_feature = feature
                best_threshold = threshold
                best_sse = sse

        return best_feature, best_threshold


    def _find_best_threshold(self, x, y):
        x_sorted_idxs = x.drop_duplicates().sort_values().index

        best_threshold = None
        sse = float("inf")

        # Loop through every possible threshold
        for i in range(0, x_sorted_idxs.size - 1):
            threshold = (x[x_sorted_idxs[i]] + x[x_sorted_idxs[i + 1]]) / 2

            # Split data by the threshold
            left_vals = y.loc[x <= threshold]
            right_vals = y.loc[x > threshold]

            # Mean for both sides
            left_mean = left_vals.mean()
            right_mean = right_vals.mean()

            # Sum of squared errors for both sides
            left_sse = np.power(left_vals - left_mean, 2).sum()
            right_sse = np.power(right_vals - right_mean, 2).sum()
            
            # Total sum of squared errors
            total_see = left_sse + right_sse

            # Update best threshold
            if (total_see < sse):
                best_threshold = threshold
                sse = total_see
                
        return best_threshold, sse

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.avg
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _build_representation(self, node):

        if node.is_leaf():
            return {"avg": node.avg}
        
        return {"feature": node.feature,
                "thrs": node.threshold,
                "left": self._build_representation(node.right),
                "right": self._build_representation(node.right)}


if __name__ == "__main__":
    # Import libraries 
    from math import sqrt
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import numpy as np

    np.random.seed(131102)

    # *Make sure to have the data set "concrete-data.csv" in the same directory as the python file*
    data = pd.read_csv("./concrete-data.csv")

    # Split data
    train, test = train_test_split(data, test_size=0.2)

    # Training data
    X = train[data.columns[:-1]]
    y = train[data.columns[-1]]

    # Test data
    test_X = test[data.columns[:-1]]
    test_y = test[data.columns[-1]]

    # Decision tree (regression)
    tree = DecisionTree(8, 100)
    tree.train(X, y, 7)

    # Predictions
    pred = tree.predict(test_X)

    # Values
    print("---- Actual and predicted values ----")
    values = test_y.to_frame()
    values["Predicted value"] = pred.tolist()
    values.columns = ["Actual value", "Predicted value"]
    print(values)

    # Root Mean Square Error
    RMSE = sqrt(np.power((test_y - pred), 2).sum() / pred.size)
    print("\n---- Root Mean Square Error ----")
    print(RMSE)

    # Tree representation
    print("\n---- Tree representation ----")
    tree.show()

