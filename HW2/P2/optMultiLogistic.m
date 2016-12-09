%==========================================================================%
% Filename: optMultiLogistc.m
% Purpose: Get cost and gradient when given parameters of softmax
%          regression model and training data
% Input: W - World state which is in the form of a 'index' matrix (N x I)
%        x - all training data (I x D)
%        phi - parameters (N x D)
% Output: L - cross entropy
%         g - gradient (N x D)
%
%==========================================================================%
function [L, g] = optMultiLogistic(W, x, phi)
    
    % calculate softmax of all data
    y = linearSoftMax(phi, x);
    
    % calculate sum of log probability (cross entropy)
    L = -sum(log(y(logical(W))));
    
    % calculate gradient
    g = (y - W)*x;
    
end