Lo, Teng Kin
46867024
tklo@cs.ubc.ca

Yan Zhao
79098142
zhaoy27@cs.ubc.ca

Scott Sallinen
47778105
scotts@ece.ubc.ca


CPSC 540 Machine Learning 2014W
UBC Computer Science


A. Value-added Contributions

1) weights: give weights to training samples

2) lambdaL2: add a L2 regularization term options.lambdaL2

3) addBias: add a bias column to design matrix


B. Collaboration
refactoring: refactored the following into matLearn_regression_refactoredL2NB.m
1) matLearn_regression_NB.m 
2) matLearn_regression_L2.m
3) matLearn_regression_regressOnOne.m

comparison: compared the above in demo_regression_refactoredL2NB.m
2) and 3) are identical in the plot because the training data has only 1 feature.

C. Setup

The code assumes all the files are inside the folder ./matLearn

1) matLearn_regression_refactoredL2NB.m
implements the regression_refactoredL2NB model

2) demo_regression_refactoredL2NB.m
demo of regression_refactoredL2NB model

3)linearRegressionData.mat
data used in demo

4)minFunc_2012/
we need to call minFunc_2012/minFunc
