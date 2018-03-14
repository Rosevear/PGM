function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numExamples = N;
numClasses = K;
numBodyParts = 10;
logProbs = zeros(1, numExamples);
for curExampleIdx = 1:numExamples
%     disp('EXAMPLE')
%     curExampleIdx
    
    %Want to get the probability of the current pose (current values of Oi
    %for i = 1 to 10), summed over all the possible values for the current
    %class
    curExampleProb = 0;
    for curClassIdx = 1:numClasses
%         disp('CLASS');
%         curClassIdx
        
        %Get the correct graph structure for the current class
        if ismatrix(G)
            curClassGraph = G;
        else
            curClassGraph = squeeze(G(:, :, curClassIdx));
        end
        
        %Get the probability for each body part conditioned on its parent
        %(if present) and the current value of k (the current class)
        curPoseClassProb = log(P.c(curClassIdx));
        for curBodyPartIdx = 1:numBodyParts
%             disp('POSE');
%             curBodyPartIdx
            curBodyPartOrientProb = calcCurBodyPartOrientationProb(P, curClassGraph, curExampleIdx, curClassIdx, curBodyPartIdx, dataset);
%             curBodyPartOrientProb
            curPoseClassProb = curPoseClassProb + curBodyPartOrientProb;
        end
%         disp('class log prob');
%         curPoseClassProb
        
        %Total probability of the current pose over all classes, computed
        %NOTE: this is in normal probability space, hence we mus sum the
        %exponent of the current class probability, since it is returned
        %as a log-probability
        curExampleProb = curExampleProb + exp(curPoseClassProb);
    end
%     disp('example log prob');
%     curExampleProb
    logProbs(curExampleIdx) = log(curExampleProb);
end

loglikelihood = sum(logProbs);
end

function curBodyPartOrientationProb = calcCurBodyPartOrientationProb(P, G, curExampleIdx, curClassIdx, curBodyPartIdx, dataset)
     %curBodyPartIdx;
     curExample = squeeze(dataset(curExampleIdx, :, :));
     curBodyOrientVector = curExample(curBodyPartIdx, :);
    
    %Calculate the means
    %Check if the current body part is has a parent node in the graph
    if G(curBodyPartIdx, 1) == 0
%         disp('compute root log prob');
%         curExample;
%         curBodyOrientVector;
        
        %The current example has no parent other than the current class, so
        %use a regular gaussian mean
        xMean = P.clg(curBodyPartIdx).mu_x(curClassIdx);
        yMean = P.clg(curBodyPartIdx).mu_y(curClassIdx);
        angleMean = P.clg(curBodyPartIdx).mu_angle(curClassIdx);
    else
%         disp('compute child log prob');
        
        %The current example has another parent, so use the mean of a
        %linear gaussian
        %curBodyOrientVector;
        curParentBodyPartIdx = G(curBodyPartIdx, 2);
        curParentBodyOrientVector = squeeze(dataset(curExampleIdx, curParentBodyPartIdx, :));
        
        xMean = P.clg(curBodyPartIdx).theta(curClassIdx, 5:8) * [1; curParentBodyOrientVector];
        yMean = P.clg(curBodyPartIdx).theta(curClassIdx, 1:4) * [1; curParentBodyOrientVector];
        angleMean = P.clg(curBodyPartIdx).theta(curClassIdx, 9:12) * [1; curParentBodyOrientVector];
    end
    
    %Get the variances
    xVariance = P.clg(curBodyPartIdx).sigma_x(curClassIdx);
    yVariance = P.clg(curBodyPartIdx).sigma_y(curClassIdx);
    angleVariance = P.clg(curBodyPartIdx).sigma_angle(curClassIdx);
    
    %Get the probabilities
    xProb = lognormpdf(curBodyOrientVector(2), xMean, xVariance);
    yProb = lognormpdf(curBodyOrientVector(1), yMean, yVariance);
    angleProb = lognormpdf(curBodyOrientVector(3), angleMean, angleVariance);
    
    curBodyPartOrientationProb = xProb + yProb + angleProb;
    %curBodyPartOrientationProb
end