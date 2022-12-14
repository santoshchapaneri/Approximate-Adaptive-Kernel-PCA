%% Adaptive Kernel PCA demo
clear all; close all; clc;

fData = 5; % kpcaScholopf
% 1: Clusters Data from kpcaScholopf
% 2: Concentric Circles
% 3: Random 3D Cluster Dataset (Linearly Seperable)
% 4: CheckerBoard Dataset (Non-Linearly Seperable)
% 5: breast-cancer-wisconsin (high dim data)
% 6: Ionosphere (high dim data)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            1)  Generate Data                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (fData==1)
% Data from kpcaScholopf
rbf_var = 0.1;
xnum = 4;
ynum = 2;
max_ev = xnum*ynum;
% (extract features from the first <max_ev> Eigenvectors)
x_test_num = 15;
y_test_num = 15;
cluster_pos = [-0.5 -0.2; 0 0.6; 0.5 0];
cluster_size = 30;
num_clusters = size(cluster_pos,1);
train_num = num_clusters*cluster_size;
patterns = zeros(train_num, 2);
range = 1;
randn('seed', 0);
for i=1:num_clusters,
  patterns((i-1)*cluster_size+1:i*cluster_size,1) = cluster_pos(i,1)+0.1*randn(cluster_size,1);
  patterns((i-1)*cluster_size+1:i*cluster_size,2) = cluster_pos(i,2)+0.1*randn(cluster_size,1);
end
X = patterns;
% Plot original data
plot_options            = [];
plot_options.is_eig     = false;
plot_options.labels     = [];
plot_options.title      = 'Circles Dataset';

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',plot_options);
end

%% Concentric Circles data
if (fData==2)
num_samples = 500;
dim_samples = 2;
num_classes = 2;

[X,labels]  = ml_circles_data(num_samples,dim_samples,num_classes);

% Adjust data to N x M (dimension x samples)
X = X';
[N,M] = size(X);

% Plot original data
plot_options            = [];
plot_options.is_eig     = false;
plot_options.labels     = labels;
plot_options.title      = 'Circles Dataset';

if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = ml_plot_data(X',plot_options);
end

%% Generate Random 3D Cluster Dataset (Linearly Seperable)
if (fData==3)
num_samples     = 300;
num_classes     = 3;
dim             = 3;
[X,labels,gmm]  = ml_clusters_data(num_samples,dim,num_classes);

% Adjust data to N x M (dimension x samples)
X = X';
[N,M] = size(X);

% Plot original data
plot_options            = [];
plot_options.is_eig     = false;
plot_options.labels     = labels;
plot_options.title      = 'Random Cluster Dataset';

if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = ml_plot_data(X',plot_options);
end

%% Generate CheckerBoard Dataset (Non-Linearly Seperable)
if (fData==4)
num_samples_p_quad = 100; % Number of points per quadrant
[X,labels] = ml_checkerboard_data(num_samples_p_quad);
labels(find(labels ==-1)) = 2;

% Adjust data to N x M (dimension x samples)
X = X';
[N,M] = size(X);

% Plot original data
plot_options            = [];
plot_options.is_eig     = false;
plot_options.labels     = labels;
plot_options.title      = 'CheckerBoard Dataset';

if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = ml_plot_data(X',plot_options);
end

%% High dim data
if (fData==5)
% [~,~,data] = xlsread('breast-cancer-wisconsin.csv');
% X = data(:,1:9);
% X=cell2mat(X); % got N x D matrix
% % Adjust data to N x M (dimension x samples)
% X = X';
% [N, M] = size(X);
% labels = data(:,end);
% class_names  = unique(labels);
% % if labels are strings convert them to numbers
% if iscellstr(class_names) == true
%    labels_tmp = zeros(size(labels,1),1);
%    cellfind   = @(string)(@(cell_contents)(strcmp(string,cell_contents)));
% 
%    for i=1:size(class_names,1)
%        idx              = cellfun(cellfind(class_names{i}),labels);            
%        labels_tmp(idx)  = i;
%    end
% 
%    labels = labels_tmp;
% end
load('BCData.mat');
X = X_BC;
[N, M] = size(X);
labels = labels_BC;
% Plot original data
plot_options            = [];
plot_options.is_eig     = false;
plot_options.labels     = labels;
plot_options.title      = 'Breast Cancer Dataset';

if exist('hd','var') && isvalid(hd), delete(hd);end
hd = ml_plot_data(X',plot_options);
end

%% High dim data
if (fData==6)
load('Ionosphere.mat');
[N, M] = size(X);
% Plot original data
plot_options            = [];
plot_options.is_eig     = false;
plot_options.labels     = labels;
plot_options.title      = 'Ionosphere Dataset';

if exist('hi','var') && isvalid(hi), delete(hi);end
hi = ml_plot_data(X',plot_options);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            2)  Apply PCA on Dataset                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2a) Compute PCA with ML_toolbox
options = [];
options.method_name       = 'PCA';
options.nbDimensions      = N;
[pca_X, mappingPCA]       = ml_projection(X',options);
% Extract Principal Directions, Components and Projection
V  = mappingPCA.M;      % Eigenvectors
L  = mappingPCA.lambda; % Eigenvalues diagonal
Mu = mappingPCA.mean';  % Mean of Dataset (for reconstruction)

% 2b) Compute Mapping Function and Visualize Embedding
p = 2;%3;
A = V(:,1:p)';
% Compute the new embedded points
y = A*X;

% Plot PCA projections
if exist('h2b','var') && isvalid(h2b), delete(h2b);end
plot_options             = [];
plot_options.is_eig      = true;
plot_options.labels      = labels;
plot_options.plot_labels = {'$y_1$','$y_2$','$y_3$'};
plot_options.title       = 'Projected data with Linear PCA';
h2b = ml_plot_data(y',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            3)  Apply Kernel PCA on Dataset                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3a) Compute kernel PCA of Dataset and Check Eigenvalues
% Compute kPCA with ML_toolbox
options = [];
options.method_name       = 'KPCA';  % Choosing kernel-PCA method
options.nbDimensions      = 10;       % Number of Eigenvectors to keep.
options.kernel            = 'gauss'; % Type of Kernel: {'poly', 'gauss'}
options.kpar              = [0.75];   % Variance for the RBF Kernel
                                     % For 'poly' kpar = [offset degree]
options.norm_K            = true;    % Normalize the Gram Matrix                                 
[kpca_X, mappingkPCA]     = ml_projection(X',options);

% Extract Eigenvectors and Eigenvalues
V     = real(mappingkPCA.V);
K     = mappingkPCA.K;
L     = real(mappingkPCA.L);

% 3b) Choose p, Compute Mapping Function and Visualize Embedded Points 
% Chosen Number of Eigenvectors to keep
% p = intrinsic_dim(X', 'EigValue'); 
p = 2;%3;

% Compute square root of eigenvalues matrix L
sqrtL = diag(sqrt(L));

% Compute inverse of square root of eigenvalues matrix L
invsqrtL = diag(1 ./ diag(sqrtL));

% Compute the new embedded points
% y = 1/lambda * sum(alpha)'s * Kernel (non-linear projection)
% y = sqrtL(1:p,1:p) * V(:,1:p)' = invsqrtL(1:p,1:p) * V(:,1:p)' * K;
y = sqrtL(1:p,1:p) * V(:,1:p)';

% Plot result of Kernel PCA
if exist('h3','var') && isvalid(h3), delete(h3);end
plot_options              = [];
plot_options.is_eig       = false;
plot_options.labels       = labels;
plot_options.plot_labels  = {'$y_1$','$y_2$','$y_3$'};
plot_options.title        = 'Projected data with Kernel PCA';
if exist('h3b','var') && isvalid(h3b), delete(h3b);end
h3b = ml_plot_data(y',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            4)  Apply Adaptive Kernel PCA on Dataset                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% numKernels = 10;
% ITE = 20;
% % llow = 2; rlow = 1;
% 
% d = intrinsic_dim(X', 'EigValue'); 
% g = 1;
% 
% % %fa--training data
% % %fb--test data
% % %nk--number of kernels
% % %ITE--iteration steps
% % %llow---L reduced dim
% % %rlow--V reduced dim
% % [LX,VX,trainFX,testFX]=akpca(X, [], numKernels, ITE, d, g);
% [L, R, Xnew] = myAKPCA(X, numKernels, ITE, d, g);
% 
% % X is D x N
% % Xnew is d x N, where d << D
% 
% % Plot result of Adaptive Kernel PCA
% if exist('h4','var') && isvalid(h4), delete(h4);end
% plot_options              = [];
% plot_options.is_eig       = false;
% plot_options.labels       = labels;
% plot_options.plot_labels  = {'$y_1$','$y_2$','$y_3$'};
% plot_options.title        = 'Projected data with Adaptive Kernel PCA';
% if exist('h4b','var') && isvalid(h4b), delete(h4b);end
% h4b = ml_plot_data(Xnew',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            4)  Apply Approximate Adaptive Kernel PCA on Dataset                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numKernels = 10;
% d = intrinsic_dim(X', 'EigValue'); 
d = 2;
g = 1;

% %fa--training data
% %fb--test data
% %nk--number of kernels
% %ITE--iteration steps
% %llow---L reduced dim
% %rlow--V reduced dim
% [LX,VX,trainFX,testFX]=akpca(X, [], numKernels, ITE, d, g);
[L_approx, R_approx, Xnew_approx] = myApproxAKPCA(X, numKernels, d, g);

% X is D x N
% Xnew is d x N, where d << D

% Plot result of Adaptive Kernel PCA
if exist('h5','var') && isvalid(h5), delete(h5);end
plot_options              = [];
plot_options.is_eig       = false;
plot_options.labels       = labels;
plot_options.plot_labels  = {'$y_1$','$y_2$','$y_3$'};
plot_options.title        = 'Projected data with Approximate Adaptive Kernel PCA';
if exist('h5b','var') && isvalid(h5b), delete(h5b);end
h5b = ml_plot_data(Xnew_approx',plot_options);