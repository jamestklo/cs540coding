function [] = demo_regression_NB()
clear all
close all

%% Load data
load linearRegressionData.mat

%% Add path of minFunc
addpath minFunc_2012\

%% Plot training data
% Plot data
figure(1);
plot(X,y,'b.')
title('Training Data');
hold on
yl = ylim;
xl = xlim;

liners = [];
liners.NB = 'r-';

%% NB regression
options=[];
options.addBias = 1;
exec_regression(@matLearn_regression_NB, options, X, y, Xtest, ytest, 2, 'NB Regression on Testing Data', xl, yl, liners.NB);

%% NB regression with L2 Regularization
options=[];
options.addBias = 1;
options.lambdaL2 = 10;
exec_regression(@matLearn_regression_NB, options, X, y, Xtest, ytest, 3, 'NB Regression (L2 regularization) on Testing Data', xl, yl, liners.NB);

%% NB regression with training weights
options=[];
options.addBias = 1;
options.weights = [ones(1, size(X,1) - 100), 0.1*ones(1,100)];
exec_regression(@matLearn_regression_NB, options, X, y, Xtest, ytest, 4, 'NB Regression (training weights) on Testing Data', xl, yl, liners.NB);
end

function exec_regression(method, options, Xtrain, ytrain, Xtest, ytest, fig, titler, xl, yl, liner)
  % train regression model
  model = method(Xtrain,ytrain,options);

  % compute test error
  yhat = model.predict(model,Xtest);
  testError = mean((yhat - ytest).^2);
  fprintf('MSE with %s is: %.3f\n',titler,testError);

  % visualization
  figure(fig);
  plot(Xtest,ytest,'b.');
  title(titler);
  hold on
  plot([0 1],[[1 0]*model.w [1 1]*model.w],liner);
  ylim(yl);
  xlim(xl);
end
