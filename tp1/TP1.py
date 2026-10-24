import numpy as np
import os
import matplotlib.pyplot as plt

from tp1.CalculCost import CalculCost
from tp1.DesctGrad import DesctGrad

# ============================================================
# 4.1 Affichage des données
# ============================================================
datafile = os.path.join('ville.txt')
data = np.loadtxt(datafile, delimiter=',')

# Afficher les donnees
X, y = data[:, 0], data[:, 1]
# nombre d'exemples d'apprentissage
m = y.size
plt.figure()
plt.scatter(X, y)
plt.xlabel('Population de la ville en 10000 habitants')
plt.ylabel('Benefice en 10000$')

# ============================================================
# 4.2 Implémentation
# ============================================================
# Ajouter une colonne de uns à X
X = np.stack([np.ones(m), X], axis=1)
# Initialiser les parametres d'ajustements
theta = np.array([0.0, 0.0])

# ============================================================
# 4.2.1 Calculer le coût J(theta)
# ============================================================
# Test de la fonction cout
print('J(theta) =', CalculCost(X, y, theta))
# Q.2 : theta = [-1; 2]
print('J([-1, 2]) =', CalculCost(X, y, np.array([-1.0, 2.0])))

# ============================================================
# 4.2.2 Descente du gradient
# ============================================================
# Test de Descente du gradient
alpha, iters = 0.01, 1500
theta, cout = DesctGrad(X, y, theta, alpha, iters)
print('theta =', theta)
print('J(theta) =', CalculCost(X, y, theta))

# La prédiction (bénéfices pour 40000 et 80000 habitants)
prdct1 = np.array([1, 4]).dot(theta)
prdct2 = np.array([1, 8]).dot(theta)
print('prdct1, prdct2 =', prdct1, prdct2)

# Droite de la régression linéaire
plt.figure()
plt.scatter(X[:, 1], y, label='Donnees d apprentissage')
plt.plot(X[:, 1], X.dot(theta), 'r-', label='Regression Lineaire')
plt.legend()
plt.xlabel('Population de la ville en 10000 habitants')
plt.ylabel('Benefice en 10000$')

# Affichage du coût
plt.figure()
plt.plot(np.arange(iters), cout)
plt.xlabel('Iterations')
plt.ylabel('Cout')

# Q.6 : initialisation aléatoire, à zéro et à un
plt.figure()
np.random.seed(0)
for init, name in [(np.random.rand(2), 'aleatoire'),
                   (np.zeros(2), 'zero'),
                   (np.ones(2), 'un')]:
    th, c = DesctGrad(X, y, init, alpha, iters)
    print('Initialisation', name, '-> theta =', th, ' J =', CalculCost(X, y, th))
    plt.plot(np.arange(iters), c, label=name)
plt.xlabel('Iterations')
plt.ylabel('Cout')
plt.legend()

# ============================================================
# 3 HomeWork : salaires.txt
# ============================================================
data = np.loadtxt('salaires.txt', delimiter=',')
Xs, ys = data[:, 0], data[:, 1]
ms = ys.size
plt.figure()
plt.scatter(Xs, ys)
plt.xlabel("Annee d'experience")
plt.ylabel('Salaire')

Xs = np.stack([np.ones(ms), Xs], axis=1)
thetas, couts = DesctGrad(Xs, ys, np.zeros(2), alpha, iters)
print('theta (salaires) =', thetas)
print('Salaire 4 ans =', np.array([1, 4]).dot(thetas))
print('Salaire 8 ans =', np.array([1, 8]).dot(thetas))

plt.figure()
plt.scatter(Xs[:, 1], ys, label='Donnees d apprentissage')
plt.plot(Xs[:, 1], Xs.dot(thetas), 'r-', label='Regression Lineaire')
plt.legend()
plt.xlabel("Annee d'experience")
plt.ylabel('Salaire')

plt.figure()
plt.plot(np.arange(iters), couts)
plt.xlabel('Iterations')
plt.ylabel('Cout')

# ============================================================
# 2.3 / 4.3 sklearn.linear_model
# ============================================================
from sklearn.linear_model import SGDRegressor

# ville.txt
sgd_reg = SGDRegressor(max_iter=15000, tol=1e-3, penalty=None, eta0=0.01)
sgd_reg.fit(X, y)
print('SGD ville : theta0 =', (sgd_reg.intercept_ + sgd_reg.coef_[0]).item(),
      ' theta1 =', sgd_reg.coef_[1])
y_prdct = sgd_reg.predict(X)
plt.figure()
plt.plot(X[:, 1], y_prdct, "r-")
plt.plot(X[:, 1], y, "b.")
plt.xlabel('Population de la ville en 10000 habitants')
plt.ylabel('Benefice en 10000$')

# salaires.txt
sgd_reg = SGDRegressor(max_iter=15000, tol=1e-3, penalty=None, eta0=0.01)
sgd_reg.fit(Xs, ys)
print('SGD salaires : theta0 =', (sgd_reg.intercept_ + sgd_reg.coef_[0]).item(),
      ' theta1 =', sgd_reg.coef_[1])
plt.figure()
plt.plot(Xs[:, 1], sgd_reg.predict(Xs), "r-")
plt.plot(Xs[:, 1], ys, "b.")
plt.xlabel("Annee d'experience")
plt.ylabel('Salaire')

plt.show()
