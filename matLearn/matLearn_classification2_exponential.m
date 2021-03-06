function [model,options] = matLearn_classification2_exponential(X,y,options)
% matLearn_classification2_exponential(X,y,options)
%
% Description:
%   - Fits a linear classifier by minimizing the exponential loss
%
% Options:
%   - addBias: adds a bias variable (default: 0)
%   - lambdaL2: strenght of L2-regularization parameter (default: 0)
%
% Authors:
% 	- Mark Schmidt (2014)

[nTrain,nFeatures] = size(X);

[addBias,lambdaL2] = myProcessOptions(options,'addBias',0,'lambdaL2',0);

model.name = 'Exponential Loss';
model.predict = @predict;

if addBias
   X = [ones(nTrain,1) X];
   nFeatures = nFeatures + 1;
   model.addBias = 1;
end

optimOptions.Display = 0; % Turn off display of optimization progress
optimOptions.useMex = 0; % Don't use compiled C files
%optimOptions.numDiff = 1; % Use numerical differencing
%optimOptions.derivativeCheck = 1; % Check derivative numerically
if lambdaL2 == 0
    w = minFunc(@exponentialLoss,randn(nFeatures,1),optimOptions,X,y);
else
    w = minFunc(@exponentialLossL2,randn(nFeatures,1),optimOptions,X,y,lambdaL2);
end

model.w = w;
end

function [yhat] = predict(model,Xhat)
[nTest,nFeatures] = size(Xhat);
if isfield(model, 'addBias') && model.addBias == 1
    Xhat = [ones(nTest,1) Xhat];
end
w = model.w;
yhat = sign(Xhat*w);
end

function [f,g] = exponentialLoss(w,X,y)
  Xw = X*w;
  f = sum(exp(-y.*(Xw)));
  g = -X'*(y.*exp(-y.*(Xw)));
end

function [f,g] = exponentialLossL2(w,X,y,lambda)
    Xw = X*w;
    f = sum(exp(-y.*(Xw))) + (lambda/2)*(w'*w);
    g = -X'*(y.*exp(-y.*(Xw))) + lambda*w;
end