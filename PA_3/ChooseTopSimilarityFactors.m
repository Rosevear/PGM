function factors = ChooseTopSimilarityFactors (allFactors, F)
% This function chooses the similarity factors with the highest similarity
% out of all the possibilities.
%
% Input:
%   allFactors: An array of all the similarity factors.
%   F: The number of factors to select.
%
% Output:
%   factors: The F factors out of allFactors for which the similarity score
%     is highest.
%
% Hint: Recall that the similarity score for two images will be in every
%   factor table entry (for those two images' factor) where they are
%   assigned the same character value.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

% If there are fewer than F factors total, just return all of them.
if (length(allFactors) <= F)
    factors = allFactors;
    return;
end

% Initialize the top factors
factors = allFactors(1:F);
topSimilarityScores = zeros(1, F);
for i = 1:F
   topSimilarityScores(i) = GetValueOfAssignment(allFactors(i), [1, 1]);
end

for i = F + 1:length(allFactors)
    disp(topSimilarityScores);
    curSimilarityScore = GetValueOfAssignment(allFactors(i), [1, 1]);
    j = 1;
    while j <= F && curSimilarityScore <= topSimilarityScores(j)
        j = j + 1;
    end
    if j <= F
        [~, minScoreIndex] = min(topSimilarityScores);
        %Replace the score that has been overtaken with the newly
        %discovered score
        curOvertakenSimilarityScore = topSimilarityScores(j);
        curOvertakenFactor = factors(j);
        topSimilarityScores(j) = curSimilarityScore;
        factors(j) = allFactors(i);
        
        %Swap out the minimum top similarity score with the scire that was
        %just overtaken by the new similarity score, provided the one we
        %replaced is not already the minimum
        if j ~= minScoreIndex
            topSimilarityScores(minScoreIndex) = curOvertakenSimilarityScore;
            factors(minScoreIndex) = curOvertakenFactor;
        end
    end
end
end

