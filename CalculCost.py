import numpy as np


def CalculCost(x, y, theta):
    m = len(y)
    cout = np.sum((x.dot(theta) - y) ** 2)
    return (1 / (2 * m)) * cout
