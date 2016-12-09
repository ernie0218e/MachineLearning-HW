clear;
% read data
x = load('Data\x3.mat');
t = load('Data\t3.mat');

% classify data
train_x = x.x3_v2.train_x;
test_x = x.x3_v2.test_x;
train_y = t.t3_v2.train_y;
test_y = t.t3_v2.test_y;

% constant initialization
maxOrder = 9;
training_error = zeros(maxOrder, 1);
test_error = zeros(maxOrder, 1);

validationSet = 3;

all_training_error = zeros(validationSet, maxOrder);

% iteration over different order of model
for order = 1:maxOrder
    temp_train_error = 0;
    temp_test_error = 0; 
    minErrorVal = 10000000000;
    for i = 1:validationSet
        
        % extract data from validation set
        % 2/3 of data
        temp_train_x = train_x;
        temp_train_y = train_y;
        temp_test_x = temp_train_x(validationSet*(i - 1)+1:validationSet*i);
        temp_train_x(validationSet*(i - 1)+1:validationSet*i) = [];
        temp_test_y = temp_train_y(validationSet*(i - 1)+1:validationSet*i);
        temp_train_y(validationSet*(i - 1)+1:validationSet*i) = [];
        
        % construct matrix phi
        phi = zeros(length(temp_train_x), order + 1);
        for j = 1:order + 1
            phi(1:length(temp_train_x), j) = temp_train_x.^(j - 1);
        end
        
        % compute pseudo inverse to get coefficient w
        w = inv(phi'*phi)*phi'*temp_train_y;
        
        % compute test error of subset
        phi = zeros(length(temp_test_x), order + 1);
        for j = 1:order + 1
            phi(1:length(temp_test_x), j) = temp_test_x.^(j - 1);
        end
        temp_y = phi*w;
        temp_error = sqrt(sum((temp_y - temp_test_y).^2)/length(temp_y));
        
        % choose the best w
        if temp_error < minErrorVal    
            w_best = w;
            minErrorVal = temp_error;
        end
        
        % store all test error
        all_training_error(i, order) = temp_error;
        temp_train_error = temp_train_error + temp_error;
    end
    
    % use best w to compute test error and training error
    phi = zeros(length(temp_test_x), order + 1);
    for j = 1:order + 1
        phi(1:length(test_x), j) = test_x.^(j - 1);
    end
    temp_y = phi*w_best;
    
    test_error(order) = sqrt(sum((temp_y - test_y).^2)/length(temp_y));
    
    training_error(order) = temp_train_error / validationSet;
end

% plot the test error and training error
figure(1);
plot(1:length(test_error), test_error);
hold on;
plot(1:length(training_error), training_error);
legend('test error', 'training error');


iteration = 1e5;
step = (exp(5) - exp(-20))/iteration;

lambda_test_error = zeros(iteration, 1);
lambda_train_error = zeros(iteration, 1);
order = 9;
% train with the regulation parameter
for i = 1:iteration
    
    % set new regulation parameter lambda
    lambda = step*i + exp(-20);
    
    % construct phi
    phi = zeros(length(train_x), order + 1);
    for j = 1:order + 1
        phi(1:length(train_x), j) = train_x.^(j - 1);
    end
    
    % compute w
    w = inv(phi'*phi + lambda*eye(order+1))*phi'*train_y;
    
    % compute training error
    temp_y = phi*w;
    lambda_train_error(i) = sqrt(sum((temp_y - train_y).^2)/length(temp_y));
    
    % compute test error
    phi = zeros(length(test_x), order + 1);
    for j = 1:order + 1
        phi(1:length(test_x), j) = test_x.^(j - 1);
    end
    
    temp_y = phi*w;
    lambda_test_error(i) = sqrt(sum((temp_y - test_y).^2)/length(temp_y));
end

figure(2)
semilogx(lambda_train_error);
hold on;
semilogx(lambda_test_error);
legend('training error', 'test error');

