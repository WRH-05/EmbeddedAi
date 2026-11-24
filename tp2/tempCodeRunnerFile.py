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