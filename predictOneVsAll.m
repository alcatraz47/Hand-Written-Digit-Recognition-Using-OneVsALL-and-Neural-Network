function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);
 
p = zeros(size(X, 1), 1);

% Adding ones to the X data matrix
X = [ones(m, 1) X];

z = X * all_theta';
g = sigmoid(z);
[value, index] = max(g, [], 2);
p = index;
% =========================================================================


end
