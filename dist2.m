function D = dist2(X, C)
% Calculates squared distance between two matrices with Euclidean distance
% Both X and C should be of same column dimensions
% Eg. X is M x N and C is L x N, result D is M x L

[ndata, dimX] = size(X);
[ncentres, dimC] = size(C);

if dimX ~= dimC
    error('Data dimensions mismatch!');
end

D = ( ones(ncentres,1) * sum((X.^2)',1) )' + ...
    ones(ndata,1) * sum((C.^2)',1) - ...
    2.*(X*(C'));

% Rounding off rare errors
if any(any(D<0))
    D(D<0) = 0;
end
