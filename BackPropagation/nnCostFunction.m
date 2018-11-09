function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
% Reshaping nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m,1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
yVector = zeros(m,num_labels);
for i=1:m
  yVector(i,y(i)) = 1;
end
J = (1/m) * sum(sum(-yVector.*log(a3) - (1-yVector).*log(1-a3)));

nonBiasedTheta1 = Theta1(:, 2:end);
nonBiasedTheta2 = Theta2(:, 2:end);

regularized = sum(sum(nonBiasedTheta1.^2)) + sum(sum(nonBiasedTheta2.^2));

J = (1/m) * sum(sum(-yVector.*log(a3) - (1-yVector).*log(1-a3))) + (lambda/(2*m)) * regularized;
for t=1:m
  a1 = [1; X(t,:)'];
  z2 = Theta1*a1;
  a2 = [1; sigmoid(z2)];
%  a2 = [1; a2(t,:)];
%  a2 = [ones(size(a2,1),1) a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  ySigns = ([1:num_labels]==y(t))';
  del3 = a3 - ySigns;
%  zed2 = [ones(m,1) z2];
  del2 = Theta2' * del3 .* [1;sigmoidGradient(z2)];
  del2 = del2(2:end);
  Theta1_grad = Theta1_grad + del2 * a1';
  Theta2_grad = Theta2_grad + del3 * a2';
end
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1,1),1) Theta1(:, 2:end)];
Theta2_grad = (1/m) * Theta2_grad + [zeros(size(Theta2,1),1) Theta2(:, 2:end)];
end
