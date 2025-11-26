import numpy as np

def WMAPE(y, y_hat): return np.sum(np.abs(y - y_hat)) / np.sum(y)

def RMSSE(y, y_hat, insample):
    diff = np.diff(insample)
    denom = np.mean(diff ** 2)
    numer = np.mean((y - y_hat) ** 2)
    return np.sqrt(numer / denom)
