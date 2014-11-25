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

%% NB regression
% train NB regression model
options=[];
options.addBias = 1;
model = matLearn_regression_NB(X,y,options);

% compute test error
yhat = model.predict(model,Xtest);
testError = sum((yhat - ytest).^2)/length(ytest);
fprintf('MSE with NB regression is: %.3f\n',testError);

% visualization
figure(2);
plot(Xtest,ytest,'b.');
title('NB Regression on Testing Data');
hold on
plot([0 1],[[1 0]*model.w [1 1]*model.w],'r-');
ylim(yl);
xlim(xl);

%% NB regression with NB Regularization
% train NB regression model
options=[];
options.addBias = 1;
options.lambdaL2 = 10;
model = matLearn_regression_NB(X,y,options);

% compute test error
yhat = model.predict(model,Xtest);
testError = sum((yhat - ytest).^2)/length(ytest);
fprintf('MSE with NB regression (L2 regularization) is: %.3f\n',testError);

% visualization
figure(3);
plot(Xtest,ytest,'b.');
title('NB Regression with L2 regularization on Testing Data');
hold on
plot([0 1],[[1 0]*model.w [1 1]*model.w],'r-');
ylim(yl);
xlim(xl);

%% NB regression with training weights
% train NB regression model
options=[];
options.addBias = 1;
options.weights = [ones(1, size(X,1) - 100), 0.1*ones(1,100)];
model = matLearn_regression_NB(X,y,options);

% compute test error
yhat = model.predict(model,Xtest);
testError = sum((yhat - ytest).^2)/length(ytest);
fprintf('MSE with NB regression (with training weights) is: %.3f\n',testError);

% visualization
figure(4);
plot(Xtest,ytest,'b.');
title('NB Regression with training weights on Testing Data');
hold on
plot([0 1],[[1 0]*model.w [1 1]*model.w],'r-');
ylim(yl);
xlim(xl);
