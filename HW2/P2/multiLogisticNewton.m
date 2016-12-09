%==========================================================================%
% Filename: multiLogisticNewton.m
% Purpose: Train softmax regression model with newton-raphson algorithm
% Input: w: World state (target value) (I x 1)
%        x: Training data (I x D)
%        N: number of class
%        eta: learning rate of gradient decent method
%        
% Output: parameters phi (N X D)
%         crossEntropy - all cross Entropy (number of iterations x 1)
%==========================================================================%
function [test_error, train_error, crossEntropy, phi]...
    = multiLogisticNewton(w, x, N, eta, test_w, test_x)
    
    I = size(x, 1);
    D = size(x, 2);
    
    % initialize parameters phi   
    phi = 2*rand(N, D) - 1;
    
    count = 1;
    times = 1000;
    crossEntropy = zeros(times, 1);
    test_error = zeros(times, 1);
    train_error = zeros(times, 1);
    
    % set up world state matrix W
    W = zeros(N, I);
    for i = 1:I
        W(w(i), i) = 1;
    end
    
    while true
       
       % get cost, gradient and hessian matrix base on current parameters and data
       [L, g, H] = optMultiLogisticNewton(W, x, phi);
       
       % concatenate vectors
       g_temp = zeros(N*D, 1);
       phi_temp = zeros(N*D, 1);
       for n = 1:N
           g_temp(D*(n-1) + 1:D*n, :) = g(n, :)';
           phi_temp(D*(n-1) + 1:D*n, :) = phi(n, :)';
       end
       
       % update parameters
       phi_temp = phi_temp - eta.*pinv(H)*g_temp;
       
       % transform back
       for n = 1:N
           phi(n, :) = phi_temp(D*(n-1) + 1:D*n, :)';
       end
       
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