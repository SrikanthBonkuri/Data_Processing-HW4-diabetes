
# Q2.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_diabetes

plt.cla()   # Clear axis
plt.clf() 

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
columns = diabetes.feature_names

print(columns)

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

linear_regression = LinearRegression()

l = len(columns)

store = []
flag = []
for i in range(l):
    store.append(['a', 0])
    flag.append(False)

k = 0

for i in range(l-1):
    sfs = SequentialFeatureSelector(linear_regression, n_features_to_select=i+1)
    sfs.fit(X, y)
    arr = sfs.get_support()
    #print(arr)
    for j in range(l):
        if arr[j]==True:
            if flag[j]==False:
                X_curr = X[:,j]
                X_curr = X_curr.reshape(-1,1)
                model = linear_regression.fit(X_curr, y)
                store[k][0] = columns[j]
                store[k][1] = model.score(X_curr,y)
                flag[j] = True
                k += 1
                break

print(arr)
for j in range(l):
    if arr[j]!=True:
        X_curr = X[:,j]
        X_curr = X_curr.reshape(-1,1)
        model = linear_regression.fit(X_curr, y)
        store[k][0] = columns[j]
        store[k][1] = model.score(X_curr,y)
        k+=1
        break


print('Rankwise Features Order:')
for i in range(l):
    #store[i][1] = i+1
    print(store[i][0],end=',')

sns.set(style='whitegrid')
data = pd.DataFrame(store, columns = ['Feature', 'Squared relation']) 
reg = sns.barplot(data=data, x='Feature', y='Squared relation')

fig = reg.get_figure()
fig.subplots_adjust(top=0.90)
fig.suptitle('Ranks ordered plot')
fig.savefig('figs/Q2_Rank ordered features')



