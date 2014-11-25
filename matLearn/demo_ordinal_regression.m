function [] = demo_ordinal_regression()
  clear all;
  options1.regression = @matLearn_regression_regressOnOne;
  options1.discrete = @rounding;
  demo1_ordinal_regression('./data_regressOnOne.mat', options1);
  
  % Train exponential loss model
  pause;
  clear all;
  options2.regression = @matLearn_classification2_exponential;
  options2.discrete = @rounding;
  options2.addBias = 1;	
  options2.lambdaL2 = 1;
  demo1_ordinal_regression('./data_exponential.mat', options2);
end

function [] = demo1_ordinal_regression(datafile, options)
  %% Load 1st set of synthetic {Xtrain,ytrain} and {Xtest,ytest}
  close all;
  addpath minFunc_2012;
  load(datafile);
  fprintf('Loading %s\n', datafile);
  [model, options] = matLearn_ordinal_regression(Xtrain, ytrain, options);
  discrete = @model.discrete;
  [yhat, yregressed] = model.predict(model, Xtest);
  tdiscrete = discrete(model,ytest);
  [yhat, yregressed, tdiscrete, ytest];
  fprintf('Test error (absolute) with %s %s is: %.3f\n', model.name, model.regressed.name, mean(abs(yregressed-ytest)));
  fprintf('Test error (accuracy) with %s %s is: %.3f\n', model.name, model.regressed.name, mean(yhat~=tdiscrete));
  fprintf('\n');
end

function [discreted] = rounding(model, num)
  discreted = round(num);
end
