
# Q4.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
#import Q1_hw4_diabetes as q1

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

#X, y = q1.X, q1.y

plt.cla()   # Clear axis
plt.clf() 

pcr = make_pipeline(StandardScaler(), PCA(), LinearRegression())

nc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tuned_parameters = [{"pca__n_components": nc}]
n_folds = 5

clf = GridSearchCV(pcr, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
#train_scores = clf.cv_results_['mean_train_score']
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

#print('Train ' + str(train_scores))
print('Test ' + str(scores))

plt.plot(nc, scores)

std_error = scores_std / np.sqrt(n_folds)

plt.plot(nc, scores + std_error, "b--")
plt.plot(nc, scores - std_error, "b--")

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(nc, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel("CV score +/- std error")
plt.xlabel("Component count")
plt.axhline(np.max(scores), linestyle="--", color=".5")
plt.xlim([nc[0], nc[-1]])
plt.title('Cross Validation Score by Component Count')

plt.savefig('figs/Q4_CV_by_Component_Count')


