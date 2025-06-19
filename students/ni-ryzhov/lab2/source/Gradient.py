import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import time

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CustomGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.init_val = None

    def fit(self, X, y):
        self.models = []
        self.init_val = np.mean(y)
        y_pred = np.full(y.shape, self.init_val)

        for i in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.models.append(tree)
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_val)
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

start_time = time.time()
custom_model = CustomGradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
custom_model.fit(X_train, y_train)
custom_train_time = time.time() - start_time

y_pred_custom = custom_model.predict(X_test)
custom_mse = mean_squared_error(y_test, y_pred_custom)

print(f"Custom GB MSE: {custom_mse:.4f}")
print(f"Custom GB Training Time: {custom_train_time:.4f} seconds")

start_time = time.time()
sklearn_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
sklearn_model.fit(X_train, y_train)
sklearn_train_time = time.time() - start_time

y_pred_sklearn = sklearn_model.predict(X_test)
sklearn_mse = mean_squared_error(y_test, y_pred_sklearn)

print(f"Sklearn GB MSE: {sklearn_mse:.4f}")
print(f"Sklearn GB Training Time: {sklearn_train_time:.4f} seconds")

cv_scores = cross_val_score(sklearn_model, X, y, scoring="neg_mean_squared_error", cv=5)
print(f"Sklearn GB CV MSE: {-np.mean(cv_scores):.4f}")