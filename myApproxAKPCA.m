function [L, R, Xnew] = myApproxAKPCA(X, numKernels, d, g)
% X--training data
% numKernels--number of kernels
% ITE--iteration steps
% d---L reduced dim
% g---R reduced dim

N = size(X,2);
mX = X - mean(X,2)*ones(1,N);
s0 = sqrt(trace(mX'*mX)/(N-1));
K = zeros(N,numKernels,N);

% Step 1: Form kernel matrix
for k=1:N,
    for j=1:numKernels,
        K(:,j,k)=kernel_exp(X,X(:,k),j,s0);
    end
end
Km = mean(K,3); % Kernel mean

% Sum of all kernels
% for k=1:nk,
%     KK=KK+kernel_exp(X,X,k,s0);
% end

% Step 2:
ML = 0;
for i=1:N,
    TL = (K(:,:,i)-Km);
    ML = ML + TL*TL';
end
[L,D0] = eigs(ML,d,'LM');

% Step 3:
MV = 0;
for i=1:N,
    TR = (K(:,:,i)-Km);
    MV = MV + TR'*TR;
end
[U1,D1,V1] = svd(MV,0);
R = V1(:,1:g);

% Projection of data
Xnew = zeros(d*g,N);
for k=1:N,
    T = L' * (K(:,:,k)-Km) * R;
    Xnew(:,k) = T(:);
end
