import numpy as np


def rss(y, y_pred):
    """
    Soma dos quadrados dos resíduos (RSS)
    Argumentos:
    - y: variável dependente
    - y_pred: predição do modelo
    """
    return np.sum((y - y_pred) ** 2)


def r2(y, y_pred):
    """Coeficiente de Determinação (R^2)"""
    y_mean = np.mean(y)
    return 1 - (np.sum((y - y_pred)** 2) / np.sum((y - y_mean) ** 2))


def r2_adj(y, y_pred, p):
    """Coeficiente de Determinação Ajustado (R^2 adj)"""
    r = r2(y, y_pred)
    n = len(y)
    return 1 - (1 - r) * ((n - 1) / (n - p - 1))


def rmse(y, y_pred):
    sum = np.sum((y - y_pred) ** 2)
    n = len(y)
    return np.sqrt(1/n * sum)


def mae(y, y_pred):
    n = len(y)
    return 1/n * np.sum(np.abs(y - y_pred))
