
# Q5.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import linear_model

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
features = diabetes.feature_names

plt.cla()   # Clear axis
plt.clf()


print("Computing regularization path using the LARS ...")
_, _, coefs = linear_model.lars_path(X, y, method="lasso", verbose=True)


xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]


plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle="dashed")
plt.legend(labels=features, loc=2, bbox_to_anchor=(0.95, 1))
plt.xlabel("|coef| / max|coef|")
plt.ylabel("Coefficients")
plt.title("LASSO Path")
plt.axis("tight")

plt.savefig('figs/Q5_Lasso')



