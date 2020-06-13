function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% --------------------------- cost calculation -----------------------------

A1 = [ones(m,1) X];
Z2 = A1*Theta1';
A2 = sigmoid(Z2);
A2 = [ones(m,1) A2];
Z3 = A2*Theta2';
final =  A3 =  sigmoid(Z3);

[val, ind] = max(final,[],2);

for k=1:num_labels
	yk = (y==k);
	hx = final(:,k);
	sumTerm = -yk.*log(hx) - (1-yk).*log(1-hx);
	J = J + sum(sumTerm);
end;

tempTheta1 = Theta1;
tempTheta1(:,1) = 0;

tempTheta2 = Theta2;
tempTheta2(:,1) = 0;

J = (J/m) + (lambda /(2*m))*(sum(sum(tempTheta1.^2)) + sum(sum(tempTheta2.^2))) ;

% ------------------ gradient calculation ----------------------------------

for t = 1 : m

    a1 = X(t, :); 
    z2 = [1, a1] * Theta1'; %'
    a2 = sigmoid(z2); 
    z3 = [1, a2] * Theta2'; % '
    a3 = sigmoid(z3); 

    delta3 = a3; 
    delta3(y(t)) = delta3(y(t)) - 1;

    delta2 = delta3 * Theta2;
    delta2 = delta2(2:end); 
    delta2 = delta2 .* sigmoidGradient(z2);

    Theta2_grad = Theta2_grad + delta3' * [1, a2];
    Theta1_grad = Theta1_grad + delta2' * [1, a1]; 

end


Theta2_grad = (Theta2_grad/m) + lambda / m * tempTheta2;
Theta1_grad = (Theta1_grad/m) + lambda / m * tempTheta1;
	


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
