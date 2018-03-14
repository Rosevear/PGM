%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 %Convert the clique values to log space for max-sum
 if isMax == 1
     for i = 1:length(P.cliqueList)
         P.cliqueList(i).val = log(P.cliqueList(i).val);
     end
 end
 
[srcCliqueIndex, targetCliqueIndex] = GetNextCliques(P, MESSAGES);
while ~(srcCliqueIndex == 0 && targetCliqueIndex == 0)
    
    srcCliqueNeighbours = P.edges(srcCliqueIndex, :);
    
    %Compute the current message
    curCliquePotential = P.cliqueList(srcCliqueIndex);
    for k = 1:length(srcCliqueNeighbours)
        if k ~= targetCliqueIndex && P.edges(srcCliqueIndex, k) == 1
            if isMax == 1
                curCliquePotential = FactorSum(curCliquePotential, MESSAGES(k, srcCliqueIndex));
            else
                curCliquePotential = FactorProduct(curCliquePotential, MESSAGES(k, srcCliqueIndex));
            end
        end
    end
    
    if isMax == 1
        %Marginalize out the irrelevant variables that can't be passed on to
        %the target clique
        curCliquePotential = FactorMaxMarginalization(curCliquePotential, setdiff(curCliquePotential.var, intersect(curCliquePotential.var, P.cliqueList(targetCliqueIndex).var)));
    else
        %Marginalize out the irrelevant variables that can't be passed on to
        %the target clique
        curCliquePotential = FactorMarginalization(curCliquePotential, setdiff(curCliquePotential.var, intersect(curCliquePotential.var, P.cliqueList(targetCliqueIndex).var)));
   
        %Normalize the current clique potential to avoid numerical underflow
        normalizingConstant = sum(curCliquePotential.val);
        for i = 1:length(curCliquePotential.val)
            curCliquePotential.val(i) = curCliquePotential.val(i) ./ normalizingConstant;
        end
    end
    
    MESSAGES(srcCliqueIndex, targetCliqueIndex) = curCliquePotential;
    [srcCliqueIndex, targetCliqueIndex] = GetNextCliques(P, MESSAGES);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N
    curCliqueNeighbours = P.edges(i, :);
    curCliquePotential = P.cliqueList(i);
    for k = 1:length(curCliqueNeighbours)
        if P.edges(i, k) == 1
            if isMax == 1
                curCliquePotential = FactorSum(curCliquePotential, MESSAGES(k, i));
            else
                curCliquePotential = FactorProduct(curCliquePotential, MESSAGES(k, i));
            end
        end
    end
    P.cliqueList(i) = curCliquePotential;
 end

return