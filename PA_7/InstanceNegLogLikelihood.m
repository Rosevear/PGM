% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over X_2 and X_3, which takes on a value of 1
    % if X_2 = 5 and X_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    
    featureSet = GenerateAllFeatures(X, modelParams);

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.
    
    nll = 0;
    grad = zeros(size(theta));
    %%%
    % Your code here:
    
    %Construct factors out of the feature set
    numFeatures = length(featureSet.features);
    numParams = length(theta);
    allFeatureFactors = repmat(EmptyFactorStruct, 1, numFeatures);
    for i = 1:numFeatures
        allFeatureFactors(i).var = featureSet.features(i).var;
        allFeatureFactors(i).card = repmat(modelParams.numHiddenStates, 1, length(featureSet.features(i).var));
        allFeatureFactors(i).val = zeros(prod(allFeatureFactors(i).card), 1);
        allFeatureFactors(i) = SetValueOfAssignment(allFeatureFactors(i), featureSet.features(i).assignment, theta(featureSet.features(i).paramIdx));
        allFeatureFactors(i).val = exp(allFeatureFactors(i).val);
    end
    
    %Create the clique tree and extract the partition function
    featureCliqueTree = CreateCliqueTree(allFeatureFactors);
    for i = 1:length(featureCliqueTree.cliqueList)
        featureCliqueTree.cliqueList(i).val = transpose(featureCliqueTree.cliqueList(i).val);
    end
    [featureCliqueTreeCalibrated, logZ] = CliqueTreeCalibrate(featureCliqueTree, 0);
    
    %NEG LOG-LIKELIHOOD COMPUTATION
    %Get the weighted feature counts
    weightedFeatureCounts = zeros(1, numParams);
    for i = 1:numFeatures
        if all(y(featureSet.features(i).var) == featureSet.features(i).assignment)
            weightedFeatureCounts(featureSet.features(i).paramIdx) = weightedFeatureCounts(featureSet.features(i).paramIdx) + theta(featureSet.features(i).paramIdx);
        end
    end
    weightedFeatureCountsSum = sum(weightedFeatureCounts);
    
    %Get the regularization term for the negative log-likelihood equation
    regTerm = 0;
    for i = 1:length(theta)
        regTerm = regTerm + (theta(i) * theta(i));
    end
    regTerm = (modelParams.lambda / 2.0) * regTerm; 

    %Compute the negative log-likelihood
    nll = logZ - weightedFeatureCountsSum + regTerm;
    
    %GRADIENT COMPUTATION
    %Get the empirical feature counts
    empiricalFeatureCounts = zeros(1, numParams);
    for i = 1:numFeatures
        if all(y(featureSet.features(i).var) == featureSet.features(i).assignment)
            empiricalFeatureCounts(featureSet.features(i).paramIdx) = empiricalFeatureCounts(featureSet.features(i).paramIdx) + 1;
        end
    end
    
    %Get the expected model feature counts via clique tree inference
    expectedModelFeatureCounts = zeros(1, numParams);
    cliqueListLength = length(featureCliqueTreeCalibrated.cliqueList);
    for i = 1:numFeatures
        j = 1;
        curClique = featureCliqueTreeCalibrated.cliqueList(j);
        curFeatureCliqueNotFound = ~all(ismember(featureSet.features(i).var, curClique.var));
        while j <= cliqueListLength && curFeatureCliqueNotFound
            j = j + 1;
            curClique = featureCliqueTreeCalibrated.cliqueList(j);
            curFeatureCliqueNotFound = ~all(ismember(featureSet.features(i).var, curClique.var));
        end
        curClique.val = curClique.val / sum(curClique.val);
        varToMarginalize = setdiff(curClique.var, featureSet.features(i).var);
        curFeatureFactor = FactorMarginalization(curClique, varToMarginalize);
        curProbability = GetValueOfAssignment(curFeatureFactor, featureSet.features(i).assignment, featureSet.features(i).var);
        expectedModelFeatureCounts(featureSet.features(i).paramIdx) = expectedModelFeatureCounts(featureSet.features(i).paramIdx) + curProbability;
    end
    
    %Get the regularization gradient term
    regGradientTerm = modelParams.lambda * theta;
    
    %Compute the gradient
    grad = expectedModelFeatureCounts - empiricalFeatureCounts + regGradientTerm;   
end
