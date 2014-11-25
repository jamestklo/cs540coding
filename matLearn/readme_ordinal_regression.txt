Lo, Teng Kin
46867024
tklo@cs.ubc.ca

CPSC 540 Machine Learning 2014W
UBC Computer Science


A. Value-added Contributions

1) Weighting, model accepts options.weights.
In addition to passing it to the regression model, matLearn_ordinal_regression also uses options.weights to learn the thresholds

2) Variants: irregular class labels
In addition to the traditional {1, 2, 3, ..., C} class labels, the current implementation abstracts class labels so that the model accepts any form of classes, both regular (e.g. {-1 and 1}) and irregular ({-8, -7, -2, 3, 6, 12, ... cK}).
The only requirement is that the class labels remain ordinal e.g. c1 < c2 < c3 < ... < cK.
For example c1=-1 < c2=1, would work seamlessly in the current model: please see demo2_ordinal_regression for details.


B. Collaboration


C. Setup
The code assumes all the files are inside the folder ./matLearn

1) matLearn_ordinal_regression.m
implements the ordinal_regression model

2) demo_ordinal_regression.m - multiple demo's
1st demo, uses data from data_regressOnOne.mat
Here, we round the y-values so that there are 11 class labels: from -8 to 2

2nd demo, uses data from data_exponential.mat
Here, the class labels have distance 2, instead of 1 (the normal case).

3rd demo, using Yan Zhao's code on matLearn_regression_L2