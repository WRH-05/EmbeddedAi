import os
import matplotlib
matplotlib.use('Agg')  # Must be set BEFORE importing pyplot or svm

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import svm

np.random.seed(0)

OUTPUT_DIR = "tp2/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(filename):
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def Noyaugaussien(x1, x2, sigma):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.exp(-np.dot(x1 - x2, x1 - x2) / (2 * sigma**2))


# ---------- 2.1 SVM linéaire ----------
d = loadmat("tp2/TP2data1.mat")
X, y = d['X'], d['y'][:, 0]

plt.figure()
svm.traceData(X, y)
plt.title('Données 1')
save_fig("2_1_Q1_data1.png")

# Q.2 : repérer le(s) point(s) aberrant(s) (ex: point positif isolé tout à gauche)
outliers = X[(y == 1) & (X[:, 0] < 1)]
print("2.1 Q.2 - Point(s) aberrant(s) :", outliers)

# Q.3, Q.5, Q.6 : variation de C, vecteurs de support et équation de l'hyperplan
for C in [1, 10, 100, 1000]:
    m = svm.svmModel(X, y, C, svm.Noyaulineare, 1e-3, 20)
    w, b = m['w'], m['b']
    print(f"C={C:4d} | nSV={len(m['alphas']):2d} | "
          f"Équation: {w[0]:.4f}*x1 + ({w[1]:.4f})*x2 + ({b:.4f}) = 0")
    plt.figure()
    svm.VFL(X, y, m)
    plt.title(f"Data1 - C={C}, nb SV={len(m['alphas'])}")
    save_fig(f"2_1_Q3_data1_C{C}.png")


# ---------- 2.2 Noyau gaussien ----------
# Q.3 : test du noyau avec sigma = 2
print("\n2.2 Q.3 (sigma=2):",
      Noyaugaussien([1, 1, 1, 0], [1, 1, 1, 1], 2),
      Noyaugaussien([1, -1, 4, 0], [0, 5, 2, 1], 2))

# Q.5 : même test avec d'autres sigma
print("2.2 Q.5:")
for s in [0.5, 5, 20]:
    print(f"  sigma = {s:4} ->",
          Noyaugaussien([1, 1, 1, 0], [1, 1, 1, 1], s),
          Noyaugaussien([1, -1, 4, 0], [0, 5, 2, 1], s))

d = loadmat("tp2/TP2data2.mat")
X, y = d['X'], d['y'][:, 0]

# Q.1 : tracer les données 2D
plt.figure()
svm.traceData(X, y)
plt.title('Données 2')
save_fig("2_2_Q1_data2.png")

# Q.2 : SVM linéaire sur data2
m = svm.svmModel(X, y, 1, svm.Noyaulineare, 1e-3, 20)
plt.figure()
svm.VFL(X, y, m)
plt.title('Linéaire sur data2')
save_fig("2_2_Q2_data2_linear.png")

# Q.6 et Q.7 : effet de sigma (C=1 fixé)
print("\n2.2 Q.6-Q.7 (variation de sigma, C=1):")
for s in [0.1, 0.5, 1, 10]:
    m = svm.svmModel(X, y, 1, Noyaugaussien, args=(s,))
    acc = np.mean(svm.svmPredict(m, X) == y) * 100
    print(f"  C=1   sigma={s:<4} nSV={len(m['alphas']):3d} précision={acc:.1f}%")
    plt.figure()
    svm.VFNL(X, y, m)
    plt.title(f"Gaussien C=1, sigma={s}, SV={len(m['alphas'])}, acc={acc:.1f}%")
    save_fig(f"2_2_Q7_data2_C1_sigma{s}.png")

# Q.7 : effet de C (sigma=0.1 fixé)
print("\n2.2 Q.7 (variation de C, sigma=0.1):")
for C_val in [0.1, 1, 10, 100]:
    m = svm.svmModel(X, y, C_val, Noyaugaussien, args=(0.1,))
    acc = np.mean(svm.svmPredict(m, X) == y) * 100
    print(f"  C={C_val:<4} sigma=0.1  nSV={len(m['alphas']):3d} précision={acc:.1f}%")
    plt.figure()
    svm.VFNL(X, y, m)
    plt.title(f"Gaussien C={C_val}, sigma=0.1, SV={len(m['alphas'])}, acc={acc:.1f}%")
    save_fig(f"2_2_Q7_data2_C{C_val}_sigma0.1.png")


# ---------- 2.3 Validation croisée ----------
def ParamsTP2data3(X, y, Xval, yval):
    vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    min_error, C, sigma = 1.0, None, None
    for Cv in vals:
        for sv in vals:
            model = svm.svmModel(X, y, Cv, Noyaugaussien, args=(sv,))
            err = np.mean(svm.svmPredict(model, Xval) != yval)
            if err < min_error:
                min_error, C, sigma = err, Cv, sv
    return C, sigma

d = loadmat("tp2/TP2data3.mat")
X, y, Xval, yval = d['X'], d['y'][:, 0], d['Xval'], d['yval'][:, 0]

plt.figure()
svm.traceData(X, y)
plt.title('Données 3')
save_fig("2_3_data3_raw.png")

C, sigma = ParamsTP2data3(X, y, Xval, yval)
model = svm.svmModel(X, y, C, Noyaugaussien, args=(sigma,))
plt.figure()
svm.VFNL(X, y, model)
plt.title(f'data3 (C={C}, sigma={sigma})')
save_fig("2_3_data3_optimal.png")

precision = np.mean(svm.svmPredict(model, Xval) == yval) * 100
print(f"\n2.3 data3 -> C optimal = {C}, sigma optimal = {sigma}, précision = {precision:.1f} %")


# ---------- 3. HomeWork (TP2HW.mat) ----------
d_hw = loadmat("tp2/TP2HW.mat")
X_hw, y_hw = d_hw['X'], d_hw['y'].ravel()

# Q.1 & Q.2 : SVM linéaire avec C = 10
m_hw = svm.svmModel(X_hw, y_hw, 10, svm.Noyaulineare, 1e-3, 20)
w_hw, b_hw = m_hw['w'], m_hw['b']
print(f"\n3 Q.1-Q.2 HW (C=10) | nSV={len(m_hw['alphas'])} | "
      f"Équation: {w_hw[0]:.4f}*x1 + ({w_hw[1]:.4f})*x2 + ({b_hw:.4f}) = 0")

plt.figure()
svm.VFL(X_hw, y_hw, m_hw)
plt.title('HomeWork - Linéaire C=10')
save_fig("3_Q1_HW_linear_C10.png")

# Q.3 : Validation croisée (si Xval et yval existent dans TP2HW.mat)
if 'Xval' in d_hw and 'yval' in d_hw:
    Xval_hw, yval_hw = d_hw['Xval'], d_hw['yval'].ravel()
    C_hw, sigma_hw = ParamsTP2data3(X_hw, y_hw, Xval_hw, yval_hw)
    m_hw_nl = svm.svmModel(X_hw, y_hw, C_hw, Noyaugaussien, args=(sigma_hw,))
    prec_hw = np.mean(svm.svmPredict(m_hw_nl, Xval_hw) == yval_hw) * 100
    print(f"3 Q.3 HW -> C optimal = {C_hw}, sigma optimal = {sigma_hw}, précision = {prec_hw:.1f} %")

    plt.figure()
    svm.VFNL(X_hw, y_hw, m_hw_nl)
    plt.title(f'HomeWork - Gaussien (C={C_hw}, sigma={sigma_hw})')
    save_fig("3_Q3_HW_optimal.png")
else:
    print("3 Q.3 HW -> Pas de Xval/yval dans TP2HW.mat : utiliser un split ou k-fold sur (X_hw, y_hw).")