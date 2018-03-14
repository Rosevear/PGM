function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
numExamples = N;
numLabels = K;
numBodyParts = 10;
paramStruct = struct('theta', [], 'mu_y', [], 'sigma_y', [], 'mu_x', [], 'sigma_x', [],'mu_angle', [],'sigma_angle', []);
P.clg = repmat(paramStruct, 1, numBodyParts);

%Fit the parameters for the class variables
for curLabelIdx = 1:numLabels
    curLabelExamples = labels(:, curLabelIdx);
    P.c(curLabelIdx) = FitGaussianParameters(transpose(curLabelExamples));
end

%Fit the parameters for the body variables
for curBodyPartIdx = 1:numBodyParts
    %Check if the current body part has a parent or not in the graph
    if G(curBodyPartIdx, 1) == 0
        P.clg(curBodyPartIdx) = getGaussianParams(curBodyPartIdx, dataset, labels, numLabels);
    else
        curBodyPartParentIdx = G(curBodyPartIdx, 2);
        P.clg(curBodyPartIdx) = getLinearGaussianParams(curBodyPartIdx, curBodyPartParentIdx, dataset, labels, numLabels);
    end
end
loglikelihood = ComputeLogLikelihood(P, G, dataset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('log likelihood: %f\n', loglikelihood);
end

function [params] = getGaussianParams(curBodyPartIdx, dataset, labels, numLabels)
    params = struct('theta', [], 'mu_y', [], 'sigma_y', [], 'mu_x', [], 'sigma_x', [],'mu_angle', [],'sigma_angle', []);
    curBodyPartExamples = squeeze(dataset(:, curBodyPartIdx, :));
    for curClassIdx = 1:numLabels
        %Get the relevant examples
        curClassExamples = labels(:, curClassIdx);
        curClassBodyPartExamples = [curBodyPartExamples curClassExamples];
        curClassBodyPartExamples = curClassBodyPartExamples(curClassBodyPartExamples(:, end) == 1, :);
        
        %Extract the individual values of the examples
        curClassBodyPartXVals = curClassBodyPartExamples(:, 2);
        curClassBodyPartYVals = curClassBodyPartExamples(:, 1);
        curClassBodyPartAngleVals = curClassBodyPartExamples(:, 3);
        
        %Fit the parameters based on the examples
        [params.mu_x(curClassIdx), params.sigma_x(curClassIdx)] = FitGaussianParameters(curClassBodyPartXVals);
        [params.mu_y(curClassIdx), params.sigma_y(curClassIdx)] = FitGaussianParameters(curClassBodyPartYVals);
        [params.mu_angle(curClassIdx), params.sigma_angle(curClassIdx)] = FitGaussianParameters(curClassBodyPartAngleVals);
    end
end

function [params] = getLinearGaussianParams(curBodyPartIdx, curBodyPartParentIdx, dataset, labels, numLabels)
    params = struct('theta', zeros(numLabels, 12), 'mu_y', [], 'sigma_y', [], 'mu_x', [], 'sigma_x', [],'mu_angle', [],'sigma_angle', []);
    curBodyPartExamples = squeeze(dataset(:, curBodyPartIdx, :));
    curBodyPartParentExamples = squeeze(dataset(:, curBodyPartParentIdx, :));
    for curClassIdx = 1:numLabels
        %Get the relevant examples
        curClassExamples = labels(:, curClassIdx);
        curClassBodyPartExamples = [curBodyPartExamples curClassExamples];
        curClassBodyPartExamples = curClassBodyPartExamples(curClassBodyPartExamples(:, end) == 1, :);

        %Extract the individual values of the examples
        curClassBodyPartXVals = curClassBodyPartExamples(:, 2);
        curClassBodyPartYVals = curClassBodyPartExamples(:, 1);
        curClassBodyPartAngleVals = curClassBodyPartExamples(:, 3);

        %Do the same for the parent examples
        curClassBodyPartParentExamples = [curBodyPartParentExamples curClassExamples];
        curClassBodyPartParentExamples = curClassBodyPartParentExamples(curClassBodyPartParentExamples(:, end) == 1, :);

        %Fit the parameters based on the current boy part and parent body part examples
        [theta2, params.sigma_y(curClassIdx)] = FitLinearGaussianParameters(curClassBodyPartYVals, curClassBodyPartParentExamples);
        [theta1, params.sigma_x(curClassIdx)] = FitLinearGaussianParameters(curClassBodyPartXVals, curClassBodyPartParentExamples);
        [theta3, params.sigma_angle(curClassIdx)] = FitLinearGaussianParameters(curClassBodyPartAngleVals, curClassBodyPartParentExamples);
        
        theta1 = theta1';
        theta2 = theta2';
        theta3 = theta3';
        
        %Make sure that the thetas are indexed properly
        theta2 = theta2(1:4);
        temp = theta2(4);
        theta2 = [temp theta2(1:3)];
        
        theta1 = theta1(1:4);
        temp = theta1(4);
        theta1 = [temp theta1(1:3)];
         
        theta3 = theta3(1:4);
        theta3 = theta3(1:4);
        temp = theta3(4);
        theta3 = [temp theta3(1:3)];
        
        params.theta(curClassIdx, :) = [theta2 theta1 theta3];
    end
end