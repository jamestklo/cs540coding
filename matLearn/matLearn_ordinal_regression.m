function [model, options] = matLearn_ordinal_regression(X, y, options)
% matLearn_ordinal_regression(X, y, options)
%
%% Description:
%
% # Uses a generic regression solver to predict ordinal classification
%
%% Options:
%
% # regression:	regression method
% # discrete:	function to turn regressed values into discrete values
% # thresholds:	thresholds for decision stumps; thresholds(c-1) < y(i) <= thresholds(c) implies y(i) = classes(c)
% # classes: 	discrete class labels, classes(c-1) must be less than classes(c)
%
%% Authors:
%
% # James Lo (2014)  UBC Computer Science (C) All rights reserved.
%
  model.name = 'Ordinal Regression';
  options = getDefaultOptions(X, y, options);

  % clean up options for regression
  model.thresholds	= options.thresholds;
  rmfield(options, 'thresholds');
  
  model.classes = options.classes;
  rmfield(options, 'classes');

  model.discrete = options.discrete;
  rmfield(options, 'discrete');
  
  regression = @options.regression;
  rmfield(options, 'regression');
  model.regressed = regression(X, y, options);

  model.weights = options.weights;
  if model.thresholds == false
    [model.thresholds, minErrors] = learnThresholds(model, X, y);
	[model.classes, [model.thresholds;Inf], [minErrors;Inf]];
  end
  model.predict = @predict; 
end

function [filled] = getDefaultOptions(X, y, options)
  filled = options;
  % define default options
  if isfield(filled, 'regression') == 0 || isa(@filled.regression, 'function_handle') == 0
    filled.regression = @fitlm;
  end
  if isfield(filled, 'thresholds') == 0
    filled.thresholds = false;
  end
  if isfield(filled, 'classes') == 0
    if isfield(filled, 'nClasses') && (filled.nClasses > 0)
	  minY = min(y);
	  maxY = max(y);
      filled.classes = minY:(maxY-minY)/(filled.nClasses-1):maxY;
	  rmfield(filled, 'nClasses');
      if isfield(filled, 'discrete') == 0
        filled.discrete = @discreteHash;
      end
	else
	  filled.classes = sort(unique(y));
    end
  end
  if isfield(filled, 'discrete') == 0
    filled.discrete = @discreteBinarySearch;
  end
  if isfield(filled, 'weights') == 0
    filled.weights = ones(length(y), 1);
  end
end

function [ydiscrete, yhat] = predict(model, Xhat)
  regressed = model.regressed;
  yhat = regressed.predict(regressed, Xhat);
  ydiscrete = yhat;
  discrete = @model.discrete;  
  nTest = length(yhat);
  for i=1:nTest
	ydiscrete(i) = discrete(model, yhat(i));
  end
end

function [yhat_i] = discreteBinarySearch(model, yhat_i)
  % scalability: do binary search, O(nTest*log(nClasses)) 
  % much better than O(nTest*nClasses) or O(nTest*log(nTest)) b/c nTest >> nClasses
  thresholds = model.thresholds;
  minPoint = 1;
  maxPoint = length(thresholds);
  classes = model.classes;
  
  while (minPoint < maxPoint)
    midPoint = minPoint + ceil( (maxPoint-minPoint)/2 );
    if yhat_i > thresholds(midPoint)
      minPoint = midPoint+1;
    elseif yhat_i <= thresholds(midPoint-1)
      maxPoint = midPoint-1;
    else 
      yhat_i = classes(midPoint);
      break;
    end
  end
  if (minPoint >= maxPoint)
    yhat_i = classes(minPoint);
  end
end

function [yhat_i] = discreteHash(model, yhat_i)
  % scalability2: hashing, O(nTest) 
  % assumes distance between each consecutive threshold is the same, and the distance is one.
  thresholds = model.thresholds;
  nClasses = length(thresholds);
  minT = thresholds(1);
  maxT = thresholds(nClasses);
  nClasses = nClasses + 1;

  % project regression yhat_i into classification yhat_i
  if (yhat_i <= minT)
    yhat_i = classes(1);
  elseif (yhat_i > maxT)
    yhat_i = classes(nClasses);
  else
    floored = floor(yhat_i);
    if yhat_i <= thresholds(floored-minT+1)
      yhat_i = floored;
    else
      yhat_i = ceil(yhat_i); 
    end
  end
end

function [thresholds, minErrors] = learnThresholds(model, X, y)
  classes = model.classes;
  nThresholds = length(classes) -1;
  thresholds = classes(1:nThresholds, 1);
  minErrors = zeros(nThresholds, 1);

  weights = model.weights;
  
  regressed = model.regressed;
  [nTrain, nFeatures] = size(X);
  yregressed = regressed.predict(regressed, X);
  candidates = sort(unique(yregressed));
  nCandidates = length(candidates);
  candidateAt = 1;
  for c=1:nThresholds
  	current = classes(c);
	minErr = Inf;
	minAt = candidateAt;
	candidateEnd = nCandidates-nThresholds+c;
	%if (candidateAt < candidateEnd)
      for t=candidateAt:candidateEnd
	    candidate = candidates(t);
	    %err = (number of false positives) + (number of false negatives)
		err  = sum(weights(y >  current).*(yregressed(y >  current) <= candidate)) + sum(weights(y <= current).*(yregressed(y <= current) >  candidate));
        if (err < minErr)
          minErr = err;
		  minAt = t;
	    end
	  end
	  minErrors(c) = minErr;
	  thresholds(c) = candidates(minAt);
	  candidateAt = minAt + 1;
    %end	
  end
  model.thresholds = thresholds;
  
  rdiscrete = zeros(nTrain,1);
  ydiscrete = zeros(nTrain,1);
  for i=1:nTrain  
    rdiscrete(i) = model.discrete(model, yregressed(i));
	ydiscrete(i) = model.discrete(model, y(i));
  end
  [yregressed, rdiscrete, y, ydiscrete];
  fprintf('Train error (absolute) with %s %s is: %.3f\n', model.name, model.regressed.name, mean(abs(yregressed-y)));
  fprintf('Train error (accuracy) with %s %s is: %.3f\n', model.name, model.regressed.name, mean(rdiscrete~=ydiscrete));
end
