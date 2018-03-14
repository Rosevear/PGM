%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
allVars = unique([F.var]);
numVars = length(allVars);
M = repmat(struct('var', [], 'card', [], 'val', []), 1, numVars);
cliqueTree = CreateCliqueTree(F, E);
calibratedCliqueTree = CliqueTreeCalibrate(cliqueTree, isMax);

for i = 1:numVars
    j = 1;
    curClique = calibratedCliqueTree.cliqueList(j);
    while ~any(curClique.var == allVars(i))
        j = j + 1;
        curClique = calibratedCliqueTree.cliqueList(j);
    end
    
    if isMax == 1
        curMarginal = FactorMaxMarginalization(curClique, setdiff(curClique.var, allVars(i)));
    else
        curMarginal = FactorMarginalization(curClique, setdiff(curClique.var, allVars(i)));
        normalizingConstant = sum(curMarginal.val);
        for k = 1:length(curMarginal.val)
            curMarginal.val(k) = curMarginal.val(k) ./ normalizingConstant;
        end
    end
    M(i) = curMarginal;
end
end
