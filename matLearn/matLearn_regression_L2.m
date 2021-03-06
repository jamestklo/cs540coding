function [model] = matLearn_regression_L2(X,y,options)
% matLearn_regression_L2(X,y,options)
%
% Description:
%	 - Fitting a linear regression model by minimizing the squared loss
%
% Options:
%	 - weights: giving the weight for each training example (default: ones(nTrain,1))
%	 - lambdaL2: strenght of L2-regularization parameter (default: 0)
%	 - addBias: adds a bias variable (default: 0)
%
% Authors:
% 	- Yan Zhao (2014)

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
w = minFunc(@squaredLossL2,randn(nFeatures,1),optimOptions,X,y,lambdaL2,z);

% generate returned variable "model"
model.name = 'Squared Loss Linear Regression';
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

% squared loss function with training sample weights and L2 regularization
function [f,g] = squaredLossL2(w,X,y,lambda,z)
	f = (X*w-y)'*diag(z)*(X*w-y) + lambda*(w'*w);
	g = 2*X'*diag(z)*(X*w-y) + 2*lambda*w;
end
