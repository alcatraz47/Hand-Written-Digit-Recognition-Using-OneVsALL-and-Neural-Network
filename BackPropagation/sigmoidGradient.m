function g = sigmoidGradient(z)

g = zeros(size(z));

gradient = sigmoid(z).*(1-sigmoid(z));
g = gradient;













% =============================================================




end
