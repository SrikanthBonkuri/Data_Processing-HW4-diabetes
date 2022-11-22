# hw-diabetes

## Goals:

* Model-selection techniques using the diabetes dataset
* Comparison of feature ranking with $R^2$, forward selection and Lasso
* PCR (principal component regression) with cross validation to assess feature dimension

## Assignment

Use the diabetes dataset from [sklearn.datasets.load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) to answer the questions below.

## Question 1

Rank the features according to their squared correlation with the target. 
Note that squared correlation is the same as $R^2$ for univariate regression. 
Visualize the ordered scores with a bar chart.

### Visualization:
![image](https://user-images.githubusercontent.com/45035308/202631882-f7517733-b07c-48af-9f73-484caf4f1da1.png)


## Question 2

Rank the features according to the order that they're added in the [forward sequential feature selection](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) algorithm. Use this ranking to reorder the bar chart in question 1.

### Visualization:
![Q2_Rank ordered features](https://user-images.githubusercontent.com/45035308/203200839-b5b80e82-1bb1-4d1b-ac8c-6fadc7c5fcb3.png)

## Question 3

Compare the bar charts in Questions 1 & 2.
Briefly discuss differences between the two charts and possible causes.
Add a figure to prove your point.

### Visualization:

Observing the bar charts of Feature vs Squared relation from Q1 and Q2, where chart of Q1 is ordered with respect to the value of squared relation, and chart of Q2 been ranked as per the order according to forward sequential feature selection. It is found that the features {bmi, s5, bp} follows the same ranks being the top 3 features in both approaches.
The features s4 and s6 which had 4th and 6th highest squared relation has been shifted to the rank of 8 and 10 in forward selection respectively.
Likewise, the features s2 and sex had better ranks in forward selection by difference of 2 and 5 positions compared to their respective positions with squared relation order.
The positions of features s3, s1 and age has differed by one place.

![image](https://user-images.githubusercontent.com/45035308/202631882-f7517733-b07c-48af-9f73-484caf4f1da1.png)
![Q2_Rank ordered features](https://user-images.githubusercontent.com/45035308/203200887-8a5586c1-6381-45cb-a397-2e0b9f4c9e39.png)


## Question 4

Plot cross-validation scores versus the number of components used in Principal Component Regression (PCR). 
(Recall the [PCR vs PLS](https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html)
demo mentioned in class.)
Include both training and test scores.
Comment on the dimensionality of the dataset and the degree of overfitting.
Hint: The [CV-diabetes demo](https://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html),
which uses cross-validation to determine the best `alpha`, may be helpful in answering this question.

### Visualization:
![Q4_CV_by_Component_Count](https://user-images.githubusercontent.com/45035308/203201050-cf0d3ee6-8291-4735-b4c5-2ff905716dcf.png)



## Question 5

The [lasso lars demo](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html) computes and plots the coefficients with lasso. Add a legend to the plot so that you can relate colors to feature names. Briefly compare the lasso ordering to your answers above.

### Visualization:
![Q5_Lasso](https://user-images.githubusercontent.com/45035308/203201081-c0401c69-4357-4b51-aca6-301e12d7ac89.png)


