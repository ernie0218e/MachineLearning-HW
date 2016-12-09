clear;
% Read data from file
data = xlsread('Data\data.xlsx');

% Initialize constants
test_size = 100;
total_size = length(data);
train_size = total_size - test_size;

% Produce Matrix Phi
% The content of Phi is decided by setting partial derivative of w to zero
sigma = [ones(train_size,1) data(1:train_size, 1) data(1:train_size, 2) data(1:train_size, 3) data(1:train_size, 4)];
for i = 1:4
    for j = 1:4
        sigma = cat(2, sigma, sigma(:, j+1).*sigma(:, i+1));
    end
end
for i = 1:21
    for j = 1:21
       phi(i, j) = sum(sigma(:,i).*sigma(:,j)); 
    end
    target(i) = sum(sigma(:,i).*data(1:train_size, 5));
end

% compute inverse of phi to get w
w = inv(phi)*target';

% Compute training error and test error
omega = [w(6:9)';w(10:13)';w(14:17)';w(18:21)'];

for i = (total_size-test_size+1):total_size
    y(i-(total_size-test_size)) = w(1) + data(i, 1:4)*w(2:5) + data(i, 1:4)*omega*data(i, 1:4)';
end
test_e = sqrt(sum((data((total_size-test_size+1):total_size, 5)' - y).^2)/test_size);

for i = 1:train_size
    train_y(i) = w(1) + data(i, 1:4)*w(2:5) + data(i, 1:4)*omega*data(i, 1:4)';
end
train_e = sqrt(sum((data(1:train_size, 5)' - train_y).^2)/train_size);


% 3rd order
sigma_third = [ones(train_size,1) data(1:train_size, 1) data(1:train_size, 2) data(1:train_size, 3) data(1:train_size, 4)];

% construct phi_third
for i = 1:4
    for j = 1:4
        sigma_third = cat(2, sigma_third, sigma_third(:, j+1).*sigma_third(:, i+1));
    end
end
for i = 1:4
    for j = 1:4
        for k = 1:4
            sigma_third = cat(2, sigma_third, sigma_third(:, k+1).*sigma_third(:, j+1).*sigma_third(:, i+1));
        end
    end
end
for i = 1:85
    for j = 1:85
       phi_third(i, j) = sum(sigma_third(:,i).*sigma_third(:,j)); 
    end
    target_third(i) = sum(sigma_third(:,i).*data(1:train_size, 5));
end

% compute w of third order
w_third = pinv(phi_third)*target_third';

omega_third = [w_third(6:9)';w_third(10:13)';w_third(14:17)';w_third(18:21)'];

% compute test error
for i = (total_size-test_size+1):total_size
    high_order = 0;
    for a = 1:4
        for b = 1:4
            for c = 1:4
                high_order = high_order + w_third(16*(a-1)+4*(b-1)+c+21)*data(i, a)*data(i, b)*data(i, c);
            end
        end
    end
    y_third(i-(total_size-test_size)) = w_third(1) + data(i, 1:4)*w_third(2:5) + data(i, 1:4)*omega_third*data(i, 1:4)' + high_order;
    
end

test_e_third = sqrt(sum((data((total_size-test_size+1):total_size, 5)' - y_third).^2)/test_size);

% find the data contribute minimum error
[min_test_error, min_test_index] = min(abs(data((total_size-test_size+1):total_size, 5)' - y_third));
min_test_attribute = data(min_test_index + (total_size-test_size+1), 1:4);

% compute training error
for i = 1:train_size
    high_order = 0;
    for a = 1:4
        for b = 1:4
            for c = 1:4
                high_order = high_order + w_third(16*(a-1)+4*(b-1)+c+21)*data(i, a)*data(i, b)*data(i, c);
            end
        end
    end
    train_y_third(i) = w_third(1) + data(i, 1:4)*w_third(2:5) + data(i, 1:4)*omega_third*data(i, 1:4)' + high_order;
end

train_e_third = sqrt(sum((data(1:train_size, 5)' - train_y_third).^2)/train_size);

% find the data contribute minimum error
[min_train_error, min_train_index] = min(abs(data(1:train_size, 5)' - train_y_third));
min_train_attribute = data(min_test_index, 1:4);