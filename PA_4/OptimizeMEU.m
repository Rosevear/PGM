% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule
  % (represented again as a factor) that yields that expected utility.

  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  D = I.DecisionFactors(1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  %
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  expectedUtilityFactor = CalculateExpectedUtilityFactor(I);

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