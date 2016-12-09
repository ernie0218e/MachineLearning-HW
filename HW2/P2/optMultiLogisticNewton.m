%==========================================================================%
% Filename: optMultiLogistcNewton.m
% Purpose: Get cost, gradient and Hessian matrix when given parameters 
%          of softmax regression model and training data
% Input: W - World state which is in the form of a 'index' matrix (N x I)
%        x - all training data (I x D)
%        phi - parameters (N x D)
% Output: L - cross entropy
%         g - gradient (N x D)
%         H - hessian matrix (N*D x N*D)
%
%==========================================================================%
function [L, g, H] = optMultiLogisticNewton(W, x, phi)
    % initailize common const variables

    % I: number of data
    I = size(x, 1);
    % D: dimension of data
    D = size(x, 2);
    % N: number of classes
    N = size(W, 1);
    
    H = zeros(N*D, N*D);
    
    % calculate softmax of all data
    % y:(N x I)
    y = linearSoftMax(phi, x);
    
    % calculate sum of log probability (cross entropy)
    L = -sum(log(y(logical(W))));
    
    % calculate gradient
    g = (y - W)*x;
    
    % calculate hessian matrix
    delta = eye(N, N);
    for i = 1:I
        for n = 1:N
            for m = 1:N
                H(D*(n-1)+ 1:n*D, D*(m-1)+ 1:m*D) = H(D*(n-1)+ 1:n*D, D*(m-1)+ 1:m*D) ...
                    + y(m, i)*(delta(n, m) - y(n, i)).*x(i, :)'*x(i, :);
            end
        end
    end
    
end