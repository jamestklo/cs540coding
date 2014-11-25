function [] = demo_regression_refactoredL2NB()
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
liners.O1 = 'c-';
liners.NB = 'r-';
liners.L2 = 'g-';

%% OnOne vs. NB vs. L2 regression
options=[];
options.addBias = 1;
exec_regression(@matLearn_regression_refactoredL2NB, options, X, y, Xtest, ytest, 2, 'Regression on Testing Data', xl, yl, liners);

%% OnOne vs. NB vs. L2 regression with L2 Regularization
options=[];
options.addBias = 1;
options.lambdaL2 = 10;
exec_regression(@matLearn_regression_refactoredL2NB, options, X, y, Xtest, ytest, 3, 'Regression (L2 regularization) on Testing Data', xl, yl, liners);

%% OnOne vs. NB vs. L2 regression with training weights
options=[];
options.addBias = 1;
options.weights = [ones(1, size(X,1) - 100), 0.1*ones(1,100)];
exec_regression(@matLearn_regression_refactoredL2NB, options, X, y, Xtest, ytest, 4, 'Regression (training weights) on Testing Data', xl, yl, liners);

%% OnOne vs. NB vs. L2 regression with L2 Regularization plus training weights
options.lambdaL2 = 10;
exec_regression(@matLearn_regression_refactoredL2NB, options, X, y, Xtest, ytest, 5, 'Regression (L2 regularization plus training weights) on Testing Data', xl, yl, liners);


end


function [testError]= exec_regression(method, options, Xtrain, ytrain, Xtest, ytest, fig, titler, xl, yl, liners)
  % train regression model
  
  modelL2 = method(Xtrain,ytrain,options);
  options.method = 'NB';
  modelNB = method(Xtrain,ytrain,options);
  options.method = 1;
  modelO1 = method(Xtrain,ytrain,options);

  % compute test error
  yhat = modelL2.predict(modelL2,Xtest);
  testErrorL2 = mean((yhat - ytest).^2);
  yhat = modelNB.predict(modelNB,Xtest);
  testErrorNB = mean((yhat - ytest).^2);
  yhat = modelO1.predict(modelO1,Xtest);
  testErrorO1 = mean((yhat - ytest).^2);
 
  fprintf('MSE with %s is:\n OnOne=%.3f NB=%.3f L2=%.3f\n',titler, testErrorO1, testErrorNB, testErrorL2);

  % visualization
  figure(fig);
  plot(Xtest,ytest,'b.');
  title(titler);
  hold on

  h1 = plot([0 1],[[1 0]*modelO1.w [1 1]*modelO1.w],liners.O1);
  h2 = plot([0 1],[[1 0]*modelNB.w [1 1]*modelNB.w],liners.NB);
  h3 = plot([0 1],[[1 0]*modelL2.w [1 1]*modelL2.w],liners.L2);
  ylim(yl);
  xlim(xl);
  legend([h1, h2, h3],{'OnOne', 'NB', 'L2'});
end
