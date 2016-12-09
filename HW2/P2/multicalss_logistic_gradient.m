% Filename: multicalss_logistic_mnist.m
% Purpose: Train softmax regression model and test it
clear;

% open and read the data
train_data = csvread('kdd99_training_data.csv', 1, 0);
test_data = csvread('kdd99_testing_data.csv', 1, 0);

D = 10;
train_size = size(train_data, 1);
test_size = size(test_data, 1);

train_label = train_data(1:end, D + 1);
train_data = train_data(1:end, 1:D);

test_label = test_data(1:end, D + 1);
test_data = test_data(1:end, 1:D);

disp('End of read Data');

% add ones before data
train_data = [ones(train_size, 1) train_data];

test_data = [ones(test_size, 1) test_data];


% number of class
classNum = 5;

% learning rate
eta = 0.05;

w = train_label + 1;
x = train_data;

% train model and get parameters phi
[test_error, train_error, crossEntropy, phi]...
    = multiLogistic(w, x, classNum, eta, test_label, test_data);

figure(1);
plot(crossEntropy);
title('Cross Entropy versus Iteration');
xlabel('Iteration');
ylabel('Cross Entropy');
grid on;

figure(2);
plot(test_error);
hold on;
plot(train_error);
title('Error versus Iteration');
xlabel('Iteration');
ylabel('Error Rate');
legend('Testing error', 'Training error');
grid on;