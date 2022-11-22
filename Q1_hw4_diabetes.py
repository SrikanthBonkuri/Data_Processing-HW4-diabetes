
# Q1.

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

linear_regression = LinearRegression()

l = len(columns)

plot_data = [None] * l

for i in range(l):
    X_curr = X[:,i]
    X_curr = X_curr.reshape(-1,1)
    model = linear_regression.fit(X_curr, y)
    print(columns[i], model.score(X_curr,y))
    plot_data[i] = [columns[i], model.score(X_curr,y)]
    
#bar_plot_data.sort(key=lambda x: x[1], reverse=True)
sorted_data = sorted(plot_data, key=lambda x: x[1], reverse=True)

sns.set(style='whitegrid')
data = pd.DataFrame(sorted_data, columns = ['Feature', 'Squared Correlation']) 
reg = sns.barplot(data=data, x='Feature', y='Squared Correlation')


fig = reg.get_figure()
fig.subplots_adjust(top=0.9)
fig.suptitle('Squared Correlation by Feature')
fig.savefig('figs/Q1_Squared_Correlations')



