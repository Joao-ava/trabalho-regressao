import numpy as np


class SingleLinearRegression:
    """Classe para regressão linear com apenas um vetor x"""
    def __init__(self):
        self.b0 = None
        self.b1 = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        self.b1 = np.sum((y - y_mean) * (x - x_mean)) / np.sum((x - x_mean) ** 2)
        self.b0 = y_mean - self.b1 * x_mean
        return self.predict(x)

    def predict(self, x: np.ndarray):
        return self.b0 + self.b1 * x


class MultiLinearRegression:
    """Classe para regressão linear com x sendo uma matriz"""
    def __init__(self):
        self.b = None

    def fit(self, x, y):
        data = np.column_stack((np.ones(len(y)), x))
        self.b = np.linalg.inv(data.T @ data) @ (data.T @ y)
        return self.predict(x)


    def predict(self, x):
        data = np.column_stack((np.ones(len(x)), x))
        return data @ self.b
