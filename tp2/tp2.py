import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import svm
np.random.seed(0)

def Noyaugaussien(x1, x2, sigma):
    x1 = np.asarray(x1); x2 = np.asarray(x2)
    return np.exp(-np.dot(x1 - x2, x1 - x2) / (2 * sigma**2))

# ---------- 2.1 SVM linéaire ----------
d = loadmat('TP2data1.mat'); X, y = d['X'], d['y'][:, 0]
svm.traceData(X, y); plt.title('Données 1'); plt.show()

for C in [1, 10, 100, 1000]:
    m = svm.svmModel(X, y, C, svm.Noyaulineare, 1e-3, 20)
    plt.figure()
    svm.VFL(X, y, m)
    plt.title(f"C={C}, nb SV={len(m['alphas'])}")
    print(f"C={C}  nSV={len(m['alphas'])}")
plt.show()

# ---------- 2.2 Noyau gaussien ----------
# Q.3 : test du noyau avec sigma = 2
print(Noyaugaussien([1,1,1,0],[1,1,1,1],2), Noyaugaussien([1,-1,4,0],[0,5,2,1],2))

# Q.5 : même test avec d'autres sigma
for s in [0.5, 1, 5, 20]:
    print("sigma =", s,
          Noyaugaussien([1,1,1,0],[1,1,1,1], s),
          Noyaugaussien([1,-1,4,0],[0,5,2,1], s))

d = loadmat('TP2data2.mat'); X, y = d['X'], d['y'][:, 0]

# Q.2 : SVM linéaire
m = svm.svmModel(X, y, 1, svm.Noyaulineare, 1e-3, 20)
svm.VFL(X, y, m); plt.title('Linéaire sur data2'); plt.show()

# Q.6 et Q.7 : SVM gaussien avec plusieurs sigma
for s in [0.1, 0.5, 1, 10]:
    m = svm.svmModel(X, y, 1, Noyaugaussien, args=(s,))
    acc = np.mean(svm.svmPredict(m, X) == y) * 100
    print(f"C=1  sigma={s}  nSV={len(m['alphas'])}  précision={acc:.1f}%")
    plt.figure()
    svm.VFNL(X, y, m)
    plt.title(f"Gaussien C=1, sigma={s}, précision={acc:.1f}%")
plt.show()

# ---------- 2.3 Validation ----------
def ParamsTP2data3(X, y, Xval, yval):
    vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    min_error, C, sigma = 1, None, None
    for Cv in vals:
        for sv in vals:
            model = svm.svmModel(X, y, Cv, Noyaugaussien, args=(sv,))
            err = np.mean(svm.svmPredict(model, Xval) != yval)
            if err < min_error:
                min_error, C, sigma = err, Cv, sv
    return C, sigma

d = loadmat('TP2data3.mat')
X, y, Xval, yval = d['X'], d['y'][:, 0], d['Xval'], d['yval'][:, 0]
C, sigma = ParamsTP2data3(X, y, Xval, yval)
model = svm.svmModel(X, y, C, Noyaugaussien, args=(sigma,))
svm.VFNL(X, y, model); plt.title('data3'); plt.show()
precision = np.mean(svm.svmPredict(model, Xval) == yval) * 100
print(f"C optimal = {C}, sigma optimal = {sigma}, précision = {precision:.1f} %")