function [model] = matLearn_regression_NB(X,y,options)
% matLearn_regression_NB(X,y,options)
%
% Description:
%   - Fitting a linear regression model by minimizing the Naive Bayes squared loss
%
% Options:
%   - weights: giving the weight for each training example (default: ones(nTrain,1))
%   - lambdaL2: strenght of L2-regularization parameter (default: 0)
%   - addBias: adds a bias variable (default: 0)
%
% Authors:
% 	- Scott Sallinen, James Lo, Yan Zhao (2014)
%
[nTrain,nFeatures] = size(X);

% generate default model options
[z,lambdaL2,addBias] = myProcessOptions(options,'weights',ones(nTrain,1),'lambdaL2',0,'addBias',0);

% add a bias column to X
if addBias
    X = [ones(nTrain,1) X];
    nFeatures = nFeatures + 1;
end

% optimization options 
optimOptions.Display = 0; % Turn off display of optimization progress
optimOptions.useMex = 0; % Don't use compiled C files
%optimOptions.numDiff = 1; % Use numerical differencing
%optimOptions.derivativeCheck = 1; % Check derivative numerically    

% compute weight vector w using minFunc 
w = minFunc(@squaredLossNB,randn(nFeatures,1),optimOptions,X,y,lambdaL2,z);

% generate returned variable "model"
model.name = 'Naive Bayes Squared Loss Linear Regression';
model.w = w;
model.addBias = addBias;
model.predict = @predict;
end

% prediction function
function [yhat] = predict(model,Xhat)
    [nTest,nFeatures] = size(Xhat);
    if model.addBias
        Xhat = [ones(nTest,1) Xhat];
    end
    w = model.w;
    yhat = Xhat*w;
end

% Naive Bayes squared loss function with training sample weights and L2 regularization
function [f,g] = squaredLossNB(w,X,y,lambda,z)
  [nTrain, nFeatures] = size(X);
  f = lambda*(w'*w);
  g = 2*lambda*w;
  for j=1:nFeatures
    Xj = X(:,j);
	wj = w(j);
	difj = (Xj*wj-y);
	diagz = diag(z);
	f = f + transpose(difj)*diagz*(difj);
	g(j) = 2*transpose(Xj)*diagz*difj;
  end
end
