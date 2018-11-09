function W = debugInitializeWeights(fan_out, fan_in)

W = zeros(fan_out, 1 + fan_in);

% Initializing W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
W = reshape(sin(1:numel(W)), size(W)) / 10;

% =========================================================================

end
