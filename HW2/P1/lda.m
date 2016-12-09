% Filename: lda.m
% Purpose: compute lda of data
% Input: x - Data (I x D)
%        w - label of Data (I x 1)
%        classNum - number of classes
%        K - new size of dimension of data
% Output: phi - Dimension-reduced data (I x K)
%         V - Coordinate transform matrix (D x K)
function [phi, V] = lda(x, w, classNum, K)
    
    I = size(x, 1);
    D = size(x, 2);
    
    % data mean (D x 1)
    mean = sum(x, 1)' ./ I;
    
    mu = zeros(classNum, D);
    Sw = zeros(D, D);
    Sb = zeros(D, D);
    
    for i = 1:classNum   
         class_size = size(x(w == i, :), 1);
        
         % \mu_i=\frac{\sum_{n=1}^{N}\mathbf{x}_nt_n }{N_i}
         % calculate mean for each class
         mu(i, :) = sum(x(w == i, :), 1) ./  class_size;

         % compute within-class scatter matrix
         sw = x' - repmat(mu(i, :)', 1, I);
         Sw = Sw + (sw*sw');
         
         % compute between-class scatter matrix
         sb = mu(i, :)' - mean;
         Sb = Sb + class_size.*(sb*sb');
    end
    
    % choose K eigenvectors with K greatest eigenvalues
    [V, lambda] = eigs(Sw\Sb, K);
    
    phi = (V' * x')';
    
end