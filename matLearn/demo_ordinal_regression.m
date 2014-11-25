function [] = demo_ordinal_regression()
  %demo1_ordinal_regression();
  %demo2_ordinal_regression();
  demo3_ordinal_regression();
end

function [] = demo1_ordinal_regression
  % Train regression On One loss model
  clear all;
  load data_regressOnOne.mat;
  options.regression = @matLearn_regression_regressOnOne;
  options.discrete = @rounding;
  discrete_ordinal_regression(@matLearn_ordinal_regression, options, Xtrain, ytrain, Xtest, ytest);
end

function [] = demo2_ordinal_regression
  % Train exponential loss model
  clear all;
  load data_exponential.mat;
  options.regression = @matLearn_classification2_exponential;
  options.discrete = @rounding;
  options.addBias = 1;	
  options.lambdaL2 = 1;
  discrete_ordinal_regression(@matLearn_ordinal_regression, options, Xtrain, ytrain, Xtest, ytest);
end

function [] = demo3_ordinal_regression
  % Train L2 model
  clear all;
  load linearRegressionData.mat;
  
  options.addBias = 1;
  options.lambdaL2 = 1;
  options.weights = [ones(1, size(X,1) - 100), 0.1*ones(1,100)];

  options.regression = @matLearn_regression_regressOnOne;
  discrete_ordinal_regression(@matLearn_ordinal_regression, options, X, y, Xtest, ytest);
  
  options.regression = @matLearn_regression_L2;
  discrete_ordinal_regression(@matLearn_ordinal_regression, options, X, y, Xtest, ytest);
  
end

function [] = discrete_ordinal_regression(method, options, Xtrain, ytrain, Xtest, ytest)
  fprintf('threshold(yhat)\n');
  exec_ordinal_regression(method, options, Xtrain, ytrain, Xtest, ytest);
  
  options.discrete = @rounding
  exec_ordinal_regression(method, options, Xtrain, ytrain, Xtest, ytest);

  options.discrete = @flooring
  exec_ordinal_regression(method, options, Xtrain, ytrain, Xtest, ytest);

  options.discrete = @ceiling
  exec_ordinal_regression(method, options, Xtrain, ytrain, Xtest, ytest);
end

function [model] = exec_ordinal_regression(method, options, Xtrain, ytrain, Xtest, ytest)
  close all;
  addpath minFunc_2012;
  
times = zeros(1, 3);count = 1;  
times(count) = toc; count = count + 1;
  [model, options] = method(Xtrain, ytrain, options);

  discrete = @model.discrete;
  
times(count) = toc; count = count + 1;
  [yhat, yregressed] = model.predict(model, Xtest);
times(count) = toc; count = count + 1;
  
  tdiscrete = discrete(model, ytest);
  %testResults = [yhat, yregressed, tdiscrete, ytest]
  
  fprintf('Test error (squared)  with %s %s is: %.3f\n', model.name, model.regressed.name, mean((yregressed - ytest).^2));
  fprintf('Test error (absolute) with %s %s is: %.3f\n', model.name, model.regressed.name, mean(abs(yregressed-ytest)));
  fprintf('Test error (accuracy) with %s %s is: %.3f\n', model.name, model.regressed.name, mean(yhat~=tdiscrete));
  totalTime = times(length(times))-times(1);
  trainTime = times(2)-times(1);
  testTime  = times(3)-times(2);
  
  fprintf('Time (seconds): total=%.6f train=%.6f test=%.6f\n\n', totalTime, trainTime, testTime);
  
end

function [discreted] = rounding(model, num)
  discreted = round(num);
end

function [discreted] = flooring(model, num)
  discreted = floor(num);
end

function [discreted] = ceiling(model, num)
  discreted = ceil(num);
end
