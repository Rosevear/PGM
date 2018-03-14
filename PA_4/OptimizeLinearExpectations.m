% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
  % Inputs: An influence diagram I with a single decision node and one or more utility nodes.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  % You may assume that there is a unique optimal decision.
  %
  % This is similar to OptimizeMEU except that we will have to account for
  % multiple utility factors.  We will do this by calculating the expected
  % utility factors and combining them, then optimizing with respect to that
  % combined expected utility factor.  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  % A decision rule for D assigns, for each joint assignment to D's parents, 
  % probability 1 to the best option from the EUF for that joint assignment 
  % to D's parents, and 0 otherwise.  Note that when D has no parents, it is
  % a degenerate case we can handle separately for convenience.
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  
  for i = 1:length(I.UtilityFactors)
     jointDistribution = FactorProduct(I.RandomFactors(1), I.UtilityFactors(i));
     for j = 2:length(I.RandomFactors)
        jointDistribution = FactorProduct(I.RandomFactors(j), jointDistribution);
     end
     variablesToEliminate = setdiff(union([I.RandomFactors.var], I.UtilityFactors(i).var), [I.DecisionFactors.var]);
     curEUF = VariableElimination(jointDistribution, variablesToEliminate);
     if i == 1
         expectedUtilityFactor = curEUF;
     else
         expectedUtilityFactor = FactorSum(expectedUtilityFactor, curEUF);
     end
  end
  
  D = I.DecisionFactors(1);
  
   %D has no parents
   MEU = 0;
   D.val(:) = 0;
   if length(D.var) == 1
       MEU = max(expectedUtilityFactor.val);
       curAssignment = 1;
       curValue = GetValueOfAssignment(expectedUtilityFactor, curAssignment);
       while curValue ~= MEU
            curAssignment = curAssignment + 1;
            curValue = GetValueOfAssignment(expectedUtilityFactor, curAssignment);
       end
       D = SetValueOfAssignment(D, curAssignment, 1);
   %D has parents
   else
       assignments = IndexToAssignment(1:prod(expectedUtilityFactor.card), expectedUtilityFactor.card);
       decisionParentAssignments = unique(assignments(:, 2:end));
       %disp(decisionParentAssignments);
       for i = 1:size(decisionParentAssignments, 1)
           curDecisionMaxUtility = GetValueOfAssignment(expectedUtilityFactor, [1 decisionParentAssignments(i, :)]);
           curDecisionMaxUtilityAssignment = [1 decisionParentAssignments(i, :)];
           for j = 2:D.card(1)
               curUtility = GetValueOfAssignment(expectedUtilityFactor, [j decisionParentAssignments(i, :)]);
               if curUtility > curDecisionMaxUtility
                   curDecisionMaxUtility = curUtility;
                   curDecisionMaxUtilityAssignment = [j decisionParentAssignments(i, :)];
               end
           end
           MEU = MEU + curDecisionMaxUtility;
           %Should the assignment from expected utility factor carry over
           %to the decision factor
           D = SetValueOfAssignment(D, curDecisionMaxUtilityAssignment, 1);
       end
   end
   OptimalDecisionRule = D;
end