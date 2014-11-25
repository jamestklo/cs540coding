function [] = demo_ordinal_regression()
  % Train regression On One loss model
  clear all;
  options1.regression = @matLearn_regression_regressOnOne;
  options1.discrete = @rounding;
  demo1_ordinal_regression('./data_regressOnOne.mat', @matLearn_ordinal_regression, options1);
  
  % Train exponential loss model
  pause;
  clear all;
  options2.regression = @matLearn_classification2_exponential;
  options2.discrete = @rounding;
  options2.addBias = 1;	
  options2.lambdaL2 = 1;
  demo1_ordinal_regression('./data_exponential.mat', @matLearn_ordinal_regression, options2);
  
  
end

function [model] = demo1_ordinal_regression(datafile, method, options)
  %% Load 1st set of synthetic {Xtrain,ytrain} and {Xtest,ytest}
  close all;
  addpath minFunc_2012;
  load(datafile);
  fprintf('Loading %s\n', datafile);
  
times = zeros(1, 3);count = 1;  
times(count) = toc; count = count + 1;
  [model, options] = method(Xtrain, ytrain, options);

  discrete = @model.discrete;
  
times(count) = toc; count = count + 1;
  [yhat, yregressed] = model.predict(model, Xtest);
times(count) = toc; count = count + 1;
  
  tdiscrete = discrete(model,ytest);
  [yhat, yregressed, tdiscrete, ytest];
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
