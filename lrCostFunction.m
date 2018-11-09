function [J, grad] = lrCostFunction(theta, X, y, lambda)
J = 0;
grad = zeros(size(theta));

z = X * theta;
h = sigmoid(z);
modifiedTheta = theta;
modifiedTheta(1) = 0;
J = (1/m)*sum(-y.*log(h)-(1-y).*log(1-h))+(lambda/(2*m))*sum(modifiedTheta.^2);
grad = (1/m) * ((h-y)'*X) + ((lambda/m)*modifiedTheta');

grad = grad(:);

end
