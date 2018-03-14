% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)


% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
dataset = poseData;
numActions = L;
numExamples = N;
numLabels = K;
numBodyParts = 10;
paramStruct = struct('mu_y', [], 'sigma_y', [], 'mu_x', [], 'sigma_x', [],'mu_angle', [],'sigma_angle', [], 'theta', []);
P.clg = repmat(paramStruct, 1, numBodyParts);

%Fit the prior for the start state variables
firstStateIndices = zeros(numActions, 1);
for curActionIdx = 1:numActions
   firstStateIndices(curActionIndx)  = actionData(curActionIdx).marg_ind(1);
end
for curLabelIdx = 1:numLabels
     P.c(curLabelIdx) = sum(ClassProb(firstStateIndices, curLabelIdx)) / numActions;
end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);
  
  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb,1) * .05;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %TODO: restructure this implementation
  for poseIdx = 1:size(PairProb, 1)
      P.transMatrix = P.transMatrix + reshape(PairProb(poseIdx), K, K);
  end
  P.transMatrix = P.transMatrix ./ sum(P.transMatrix, 2);
  %   for i = 1:numLabels
%       for j = 1:numLabels
%           %TODO: I assume we don't need to replace the dirichlet prior, so
%           %we just add to it
%           P.transMatrix(i, j) = P.transMatrix(i, j) + 
%       end
%   end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
%Fit the parameters of the emission probabilities
for curBodyPartIdx = 1:numBodyParts
    %Check if the current body part has a parent or not in the graph
    if G(curBodyPartIdx, 1) == 0
        P.clg(curBodyPartIdx) = getGaussianParams(curBodyPartIdx, dataset, ClassProb, numLabels);
    else
        curBodyPartParentIdx = G(curBodyPartIdx, 2);
        P.clg(curBodyPartIdx) = getLinearGaussianParams(curBodyPartIdx, curBodyPartParentIdx, dataset, ClassProb, numLabels);
    end
end
  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
  
  logEmissionProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
exampleLogProbs = zeros(1, numExamples); %For computing the log likelihood
for curExampleIdx = 1:numExamples
%     disp('Cur Example');
%     curExampleIdx
    for curClassIdx = 1:numLabels
%         disp('Cur Class');
%         curClassIdx
        %Get the correct graph structure for the current class
        if ismatrix(G)
            curClassGraph = G;
        else
            curClassGraph = squeeze(G(:, :, curClassIdx));
        end
        
        %Get the probability for each body part conditioned on its parent
        %(if present) and the current class
        curPoseClassLogProb = 0;
        for curBodyPartIdx = 1:numBodyParts
%             disp('POSE');
%             curBodyPartIdx
            curBodyPartOrientProb = calcCurBodyPartOrientationProb(P, curClassGraph, curExampleIdx, curClassIdx, curBodyPartIdx, dataset);
%             curBodyPartOrientProb
            curPoseClassLogProb = curPoseClassLogProb + curBodyPartOrientProb;
        end
        %curPoseClassLogProb
        
        logEmissionProb(curExampleIdx, curClassIdx) = curPoseClassLogProb; 
    end
    %Get the log probability of the current example, over all possible classes
    %by summing in normal probability space and returning to log space
    exampleLogProbs(curExampleIdx) = logsumexp(logEmissionProb(curExampleIdx, :));
end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
  
  %Normalize the probabilities in log space, then convert to regular
  %probability space to fill in ClassProbs
  ClassProb = exp(logEmissionProb - logsumexp(logEmissionProb));
  
  %TODO: Check if this is actually the way to compute likelihood now, or do
  %we need to include summing up over the classes again in the emissions
  %section
  loglikelihood(iter) = sum(exampleLogProbs);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting by decreasing loglikelihood
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
end

function [params] = getGaussianParams(curBodyPartIdx, dataset, classProb, numLabels)
    params = struct('mu_y', [], 'sigma_y', [], 'mu_x', [], 'sigma_x', [],'mu_angle', [],'sigma_angle', [], 'theta', []);
    curBodyPartExamples = squeeze(dataset(:, curBodyPartIdx, :));
    for curClassIdx = 1:numLabels
        %Extract the individual values of the examples
        curBodyPartYVals = curBodyPartExamples(:, 1);
        curBodyPartXVals = curBodyPartExamples(:, 2);
        curBodyPartAngleVals = curBodyPartExamples(:, 3);
        
        %Get the probabilities that the examples are an instance of any
        %given class
        curClassProbWeights = classProb(:, curClassIdx);
        
        %Fit the parameters based on the examples
        [params.mu_y(curClassIdx), params.sigma_y(curClassIdx)] = FitG(curBodyPartYVals, curClassProbWeights);
        [params.mu_x(curClassIdx), params.sigma_x(curClassIdx)] = FitG(curBodyPartXVals, curClassProbWeights);
        [params.mu_angle(curClassIdx), params.sigma_angle(curClassIdx)] = FitG(curBodyPartAngleVals, curClassProbWeights);
    end
end

function [params] = getLinearGaussianParams(curBodyPartIdx, curBodyPartParentIdx, dataset, classProb, numLabels)
    params = struct('mu_y', [], 'sigma_y', [], 'mu_x', [], 'sigma_x', [],'mu_angle', [],'sigma_angle', [], 'theta', zeros(numLabels, 12));
    curBodyPartExamples = squeeze(dataset(:, curBodyPartIdx, :));
    curBodyPartParentExamples = squeeze(dataset(:, curBodyPartParentIdx, :));
    for curClassIdx = 1:numLabels
        %Get the relevant examples

        %Extract the individual values of the examples
        curBodyPartXVals = curBodyPartExamples(:, 2);
        curBodyPartYVals = curBodyPartExamples(:, 1);
        curBodyPartAngleVals = curBodyPartExamples(:, 3);
        
        %Get the probabilities that the examples are an instance of any
        %given class
        curClassProbWeights = classProb(:, curClassIdx);

        %Fit the parameters based on the current boy part and parent body part examples
        [theta2, params.sigma_y(curClassIdx)] = FitLG(curBodyPartYVals, curBodyPartParentExamples, curClassProbWeights);
        [theta1, params.sigma_x(curClassIdx)] = FitLG(curBodyPartXVals, curBodyPartParentExamples, curClassProbWeights);
        [theta3, params.sigma_angle(curClassIdx)] = FitLG(curBodyPartAngleVals, curBodyPartParentExamples, curClassProbWeights);
        
        %TRanspose
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

function curBodyPartOrientationProb = calcCurBodyPartOrientationProb(P, G, curExampleIdx, curClassIdx, curBodyPartIdx, dataset)
     %curBodyPartIdx;
     curExample = squeeze(dataset(curExampleIdx, :, :));
     curBodyOrientVector = curExample(curBodyPartIdx, :);
    
    %Calculate the means
    %Check if the current body part is has a parent node in the graph
    if G(curBodyPartIdx, 1) == 0
        %disp('compute root log prob');
        %curExample;
        %curBodyOrientVector;
        
        %The current example has no parent other than the current class, so
        %use a regular gaussian mean
        xMean = P.clg(curBodyPartIdx).mu_x(curClassIdx);
        yMean = P.clg(curBodyPartIdx).mu_y(curClassIdx);
        angleMean = P.clg(curBodyPartIdx).mu_angle(curClassIdx);
    else
        %disp('compute child log prob');
        
        %The current example has another parent, so use the mean of a
        %linear gaussian
        %curBodyOrientVector
        curParentBodyPartIdx = G(curBodyPartIdx, 2);
        curParentBodyOrientVector = squeeze(dataset(curExampleIdx, curParentBodyPartIdx, :));
        
        yMean = P.clg(curBodyPartIdx).theta(curClassIdx, 1:4) * [1; curParentBodyOrientVector];
        xMean = P.clg(curBodyPartIdx).theta(curClassIdx, 5:8) * [1; curParentBodyOrientVector];
        angleMean = P.clg(curBodyPartIdx).theta(curClassIdx, 9:12) * [1; curParentBodyOrientVector];
    end
    
    %Get the variances
    yVariance = P.clg(curBodyPartIdx).sigma_y(curClassIdx);
    xVariance = P.clg(curBodyPartIdx).sigma_x(curClassIdx);
    angleVariance = P.clg(curBodyPartIdx).sigma_angle(curClassIdx);
    
    %Get the probabilities
    yProb = lognormpdf(curBodyOrientVector(1), yMean, yVariance);
    xProb = lognormpdf(curBodyOrientVector(2), xMean, xVariance);
    angleProb = lognormpdf(curBodyOrientVector(3), angleMean, angleVariance);
    
    curBodyPartOrientationProb = xProb + yProb + angleProb;
    %curBodyPartOrientationProb
end
    