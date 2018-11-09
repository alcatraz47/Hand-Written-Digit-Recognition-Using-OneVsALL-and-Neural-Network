function [all_theta] = oneVsAll(X, y, num_labels, lambda)

% m size and n dimentions of dataset
m = size(X, 1);
n = size(X, 2);


all_theta = zeros(num_labels, n + 1);

% Adding ones to the X data matrix
X = [ones(m, 1) X];

initial_theta = zeros(n+1,1);
options = optimset('GradObj', 'on', 'MaxIter', 50);
for c=1:num_labels,
  [theta] = fmincg(@(t)(lrCostFunction(t,X,(y==c),lambda)), initial_theta, options);
  all_theta(c,:) = theta;
end
% =========================================================================


end
