%==========================================================================%
% Filename: multiLogistic.m
% Purpose: Train softmax regression model with gradient decent method
% Input: w: World state (target value) (I x 1)
%        x: Training data (I x D)
%        N: number of class
%        eta: learning rate of gradient decent method
%        
% Output: parameters phi (N X D)
%         crossEntropy - all cross Entropy (number of iterations x 1)
%==========================================================================%
function [test_error, train_error, crossEntropy, phi]...
    = multiLogistic(w, x, N, eta, test_w, test_x)
    
    I = size(x, 1);
    D = size(x, 2);
    
    % initialize parameters phi 
    phi = 2*rand(N, D) - 1;
    
    count = 1;
    times = 25000;
    crossEntropy = zeros(times, 1);
    test_error = zeros(times, 1);
    train_error = zeros(times, 1);
    
    % set up world state matrix W
    W = zeros(N, I);
    for i = 1:I
        W(w(i), i) = 1;
    end
    
    while true
       
       % get cost and gradient base on current parameters and data
       [L, g] = optMultiLogistic(W, x, phi);
       
       % update parameters using gradient descent method
       phi = phi - eta*g;
       
       % store value of cross entropy
       crossEntropy(count) = L;
       
       test_error(count) = testMulticlassLogistic(test_x, test_w, phi);
       train_error(count) = testMulticlassLogistic(x, w - 1, phi);
       
       % algorithm termination condition
       if count >= times
           break;
       end
       display(L);
       
       count = count + 1;
       display(count);
    end

end