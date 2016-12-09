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

    
[training_data_LDA, V] = ...
    lda(training_data, training_label, classNum, 3);

test_data_LDA = (V'*test_data')';


[training_data_PCA, V] = pca(training_data, 3);
test_data_PCA = (V'*test_data')';

figure(1);
for i = 1:classNum
    plot3(training_data_LDA(training_label == i,1),...
        training_data_LDA(training_label == i,2),...
        training_data_LDA(training_label == i,3),'o');
    hold on;
end
title('LDA - Traning Set');
legend('SETOSA', 'VIRGINIC', 'VERSICOL');
grid on;

figure(2);
for i = 1:classNum
    plot3(test_data_LDA(test_label == i,1),...
        test_data_LDA(test_label == i,2),...
        test_data_LDA(test_label == i,3),'o');
    hold on;
end
title('LDA - Test Set');
legend('SETOSA', 'VIRGINIC', 'VERSICOL');
grid on;

figure(3);
for i = 1:classNum
    plot3(training_data_PCA(training_label == i,1),...
        training_data_PCA(training_label == i,2),...
        training_data_PCA(training_label == i,3),'o');
    hold on;
end
title('PCA - Traning Set');
legend('SETOSA', 'VIRGINIC', 'VERSICOL');
grid on;

figure(4);
for i = 1:classNum
    plot3(test_data_PCA(test_label == i, 1),...
        test_data_PCA(test_label == i, 2),...
        test_data_PCA(test_label == i, 3),'o');
    hold on;
end
title('PCA - Test Set');
legend('SETOSA', 'VIRGINIC', 'VERSICOL');
grid on;
