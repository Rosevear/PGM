% Copyright (C) Daphne Koller, Stanford University, 2012

function EU = SimpleCalcExpectedUtility(I)

  % Inputs: An influence diagram, I (as described in the writeup).
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return Value: the expected utility of I
  % Given a fully instantiated influence diagram with a single utility node and decision node,
  % calculate and return the expected utility.  Note - assumes that the decision rule for the 
  % decision node is fully assigned.

  % In this function, we assume there is only one utility node.
  F = [I.RandomFactors I.DecisionFactors];
  U = I.UtilityFactors(1);
  EU = [];
 
  variablesToEliminate = setdiff(union([I.RandomFactors.var], [I.DecisionFactors.var]), U.var);
  
  jointDistribution = F(1);
  for i = 2:length(F)
      jointDistribution = FactorProduct(F(i), jointDistribution);
  end
  
  utilityValueProbs = VariableElimination(jointDistribution, variablesToEliminate);
  EUFactor = FactorProduct(utilityValueProbs, U);
  EU = sum(EUFactor.val);
 
  
  
end
