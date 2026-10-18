import numpy as np

from CalculCost import CalculCost


def DesctGrad(x, y, theta, alpha, iters):
    # Initialiser quelques valeurs utiles
    m = len(y)
    cout = np.zeros(iters)
    theta = theta.copy()
    for i in range(iters):
        error = x.dot(theta) - y
        theta = theta - (alpha / m) * error.dot(x)
        cout[i] = CalculCost(x, y, theta)
    return theta, cout
