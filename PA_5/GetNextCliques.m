%GETNEXTCLIQUES Find a pair of cliques ready for message passing
%   [i, j] = GETNEXTCLIQUES(P, messages) finds ready cliques in a given
%   clique tree, P, and a matrix of current messages. Returns indices i and j
%   such that clique i is ready to transmit a message to clique j.
%
%   We are doing clique tree message passing, so
%   do not return (i,j) if clique i has already passed a message to clique j.
%
%	 messages is a n x n matrix of passed messages, where messages(i,j)
% 	 represents the message going from clique i to clique j. 
%   This matrix is initialized in CliqueTreeCalibrate as such:
%      MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);
%
%   If more than one message is ready to be transmitted, return 
%   the pair (i,j) that is numerically smallest. If you use an outer
%   for loop over i and an inner for loop over j, breaking when you find a 
%   ready pair of cliques, you will get the right answer.
%
%   If no such cliques exist, returns i = j = 0.
%
%   See also CLIQUETREECALIBRATE
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function [i, j] = GetNextCliques(P, messages)

breakOuterLoop = 0;
messagesDim = size(messages, 1);
for i = 1:messagesDim
    for j = 1:messagesDim
        if isempty(messages(i, j).var) && P.edges(i, j) == 1
            %Clique i has not sent a message to clique j, so we see if i is
            %ready to send a message based on its neighbours
            curCliqueNeighbourVector = P.edges(i, :);
            numNeighbours = sum(curCliqueNeighbourVector);
            numMessagesReceived = 0;
            for k = 1:length(curCliqueNeighbourVector)
                if curCliqueNeighbourVector(k) == 1 && ~isempty(messages(k, i).var) && k ~= j
                    numMessagesReceived = numMessagesReceived + 1;
                end
            end
            %There is only 1 neighbour that has not sent a message to the
            %current clique i: it must be the one we want to send a message to from i
            if numMessagesReceived == numNeighbours - 1
                breakOuterLoop = 1;
                break;
            end
        end
   end
    %We've found a clique j to pass from message i, so we can terminate
    if breakOuterLoop == 1
        break;
    end
end

%If we didn't break out of the loop, it was because we couldn't find any
%clique to pass a message to.
if breakOuterLoop == 0
    i = 0;
    j = 0;
end
return;
