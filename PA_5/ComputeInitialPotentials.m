
%COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
%passed in as a parameter.
%
%   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
%   struct with three fields:
%   - nodes: cell array representing the cliques in the tree.
%   - edges: represents the adjacency matrix of the tree.
%   - factorList: represents the list of factors that were used to build
%   the tree. 
%   
%   It returns the standard form of a clique tree P that we will use through 
%   the rest of the assigment. P is struct with two fields:
%   - cliqueList: represents an array of cliques with appropriate factors 
%   from factorList assigned to each clique. Where the .val of each clique
%   is initialized to the initial potential of that clique.
%   - edges: represents the adjacency matrix of the tree. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function P = ComputeInitialPotentials(C)

% number of cliques
N = length(C.nodes);

% initialize cluster potentials 
P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
P.edges = zeros(N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% First, compute an assignment of factors from factorList to cliques. 
% Then use that assignment to initialize the cliques in cliqueList to 
% their initial potentials. 

% C.nodes is a list of cliques.
% So in your code, you should start with: P.cliqueList(i).var = C.nodes{i};
% Print out C to get a better understanding of its structure.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P.edges = C.edges;

%Store the cardinality of each variable so that we can look it up later
%while assigning factors to cliques
allVariables = unique([C.factorList().var]);
cardList = zeros(1, length(allVariables));
for i = 1:length(allVariables)
    for j = 1:length(C.factorList)
        if (~isempty(find(C.factorList(j).var == i, 1)))
            cardList(i) = C.factorList(j).card(find(C.factorList(j).var == i));
            break;
        end
    end
end

%Assign factors to all the cliques and compute their initial potentials
for i = 1:N
    P.cliqueList(i).var = C.nodes{i};
    
    %Set the cardinality of the variables in the current clique
    for k = 1:length(P.cliqueList(i).var)
          P.cliqueList(i).card(k) = cardList(P.cliqueList(i).var(k));
    end
    
    %Assign the relevant factors to the current clique
    curCliqueFactors = [];
    j = 1;
    while j <= length(C.factorList)
        if all(ismember(C.factorList(j).var, P.cliqueList(i).var))
            curCliqueFactors = [curCliqueFactors C.factorList(j)];
            C.factorList(j) = [];
        else
            j = j + 1;
        end
    end
    
    %Compute the inital potential for the current clique
    curCliquePotential = struct('var', P.cliqueList(i).var, 'card', P.cliqueList(i).card, 'val', ones(1, prod(P.cliqueList(i).card)));
    for k = 1:length(curCliqueFactors)
         curCliquePotential = FactorProduct(curCliquePotential, curCliqueFactors(k));
    end
    P.cliqueList(i) = curCliquePotential;
end
