% clear previous data
clear;

% import data
[data, raw_label] = xlsread('Irisdat.xls');

% remove redundant part of label
raw_label = raw_label(2:end, 5);

% label raw_label
label = zeros(size(raw_label, 1), 1);
for i = 1:size(raw_label, 1)
    if strcmp(raw_label(i), 'SETOSA')
           label(i) = 1;
    elseif strcmp(raw_label(i), 'VIRGINIC')
            label(i) = 2;
    elseif strcmp(raw_label(i), 'VERSICOL')
            label(i) = 3;
    end
end

% define size of training set and test set
training_size = 120;
test_size = 30;

% number of class
classNum = 3;

% assign data and label to training set and test set
training_data = data(1:training_size, :);
test_data = data(training_size+1:training_size+test_size, :);
training_label = label(1:training_size, :);
test_label = label(training_size+1:training_size+test_size, :);

newDimesions = [1;2;3];

test_classChart = cell(size(newDimesions, 1), 1);
train_classChart = cell(size(newDimesions, 1), 1);
test_error = zeros(size(newDimesions, 1), 1);
training_error = zeros(size(newDimesions, 1), 1);

for c = 1:size(newDimesions, 1)
    
    [training_data_t, V] = ...
        lda(training_data, training_label, classNum, newDimesions(c));
    
    test_data_t = (V'*test_data')';

    % Use maximum likelihood method to estimate parameters

    % theta: prior prob. of that class
    theta = zeros(classNum, 1);

    % dimension of data
    D = size(training_data_t, 2);

    % mu: mean of that class
    mu = zeros(classNum, D);
    S = zeros(D, D);

    for i = 1:classNum
         % \theta_i=\frac{N_i}{\sum_{j=1}^{Classnum}{N_j}}
         % \N_i stands for the number of member which belongs to class 'i'
         theta(i) = size(training_data_t(training_label == i, :), 1) / training_size;

         % \mu_i=\frac{\sum_{n=1}^{N}\mathbf{x}_nt_n }{N_i}
         % calculate mean for each class
         mu(i, :) = sum(training_data_t(training_label == i, :), 1) ./ theta(i) ./ training_size;

         % compute scatter matrix of each class
         s = training_data_t' - repmat(mu(i, :)', 1, training_size);
         S = S + theta(i).*(s*s');
    end

    test_classChart{c} = zeros(classNum, classNum);
    % compute test error
    for n = 1:test_size
        lambda = zeros(classNum, 1);
        for i = 1:classNum
            lambda(i) = 1/((2*pi)^(D/2))*1/sqrt(det(S))*...
                    exp(-0.5*(test_data_t(n, :)-mu(i, :))*(S\(test_data_t(n, :)'-mu(i, :)')));
        end
        [maxValue, maxLabel] = max(lambda);
        if maxLabel ~= test_label(n)
           test_error(c) = test_error(c) + 1; 
        end
        test_classChart{c}(test_label(n), maxLabel) ...
            = test_classChart{c}(test_label(n), maxLabel) + 1;
    end

    test_error(c) = test_error(c) ./ test_size;

    train_classChart{c} = zeros(classNum, classNum);
    % compute training e rror
    for n = 1:training_size
        lambda = zeros(classNum, 1);
        for i = 1:classNum
            lambda(i) = 1/((2*pi)^(D/2))*1/sqrt(det(S))*...
                    exp(-0.5*(training_data_t(n, :)-mu(i, :))*(S\(training_data_t(n, :)'-mu(i, :)')));
        end
        [maxValue, maxLabel] = max(lambda);
        if maxLabel ~= training_label(n)
           training_error(c) = training_error(c) + 1; 
        end
        train_classChart{c}(training_label(n), maxLabel) ...
            = train_classChart{c}(training_label(n), maxLabel) + 1;
    end

    training_error(c) = training_error(c) ./ training_size;
end
