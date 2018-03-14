function [Beta sigma] = FitLinearGaussianParameters(X, U)

% Estimate parameters of the linear Gaussian model:
% X|U ~ N(Beta(1)*U(1) + ... + Beta(n)*U(n) + Beta(n+1), sigma^2);

% Note that Matlab/Octave index from 1, we can't write Beta(0).
% So Beta(n+1) is essentially Beta(0) in the text book.

% X: (M x 1), the child variable, M examples
% U: (M x N), N parent variables, M examples
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

M = size(U,1);
N = size(U,2);

Beta = zeros(N+1,1);
sigma = 1;

% collect expectations and solve the linear system
% A = [ E[U(1)],      E[U(2)],      ... , E[U(n)],      1     ; 
%       E[U(1)*U(1)], E[U(2)*U(1)], ... , E[U(n)*U(1)], E[U(1)];
%       ...         , ...         , ... , ...         , ...   ;
%       E[U(1)*U(n)], E[U(2)*U(n)], ... , E[U(n)*U(n)], E[U(n)] ]

% construct A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_examples = M;
A = ones(N, N);
for row = 1:N
    curRow = zeros(N, 1);
    for i = 1:N
        temp = U(:, i) .* U(:, row);
        curRow(i) = sum(temp) / num_examples;
    end 
    A(row, :) = curRow;
end
lastBaseVector = zeros(1, N);
for i = 1:N
    lastBaseVector(i) = sum(U(:, i)) / num_examples;
end
lastColumn = transpose([1 lastBaseVector]);
lastRow = lastBaseVector;
A = vertcat(lastRow, A);
A(:, end + 1) = lastColumn;

% B = [ E[X]; E[X*U(1)]; ... ; E[X*U(n)] ]

% construct B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = zeros(N, 1);
for i = 1:N
    temp = X .* U(:, i);
    B(i) = sum(temp) / num_examples;
end
ExpectedX = sum(X) / num_examples; 
B = [ExpectedX B'];
B = B';

% solve A*Beta = B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Beta = A \ B;

% then compute sigma according to eq. (11) in PA description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CovXX = getCovariance(X, X, num_examples);
sigmaSubTerm = 0;
for i = 1:N
    for j = 1:N
        sigmaSubTerm = sigmaSubTerm + (Beta(i) .* Beta(j) .* getCovariance(U(:, i), U(:, j), num_examples));
    end
end

sigma = sqrt((CovXX - sigmaSubTerm));
end

function coVariance = getCovariance(x, y, numSamples)
    coVariance = (sum(x .* y) / numSamples) - ((sum(x) / numSamples) .* (sum(y) / numSamples));
end