import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import AutoModel, AutoTokenizer 

#==============================================================================================
# Model
## (1) Multinomial Logistic Regression
class MulLR:
    def __init__(self, max_iter=1000, lr=0.1, lambda_=0.01, penalty = 'l2'):
        """
        Initialize a multinomial logistic regression object.

        Parameters
        ----------
        max_iter : int, optional (default=1000)
            Maximum number of iterations for gradient descent
        lr : float, optional (default=0.1)
            Learning rate for gradient descent
        lambda_ : float, optional (default=0.01)
            Regularization parameter
        penalty : {'l1', 'l2'}, optional (default='l2')
            Type of regularization penalty to use
        """
        self.max_iter = max_iter
        self.lr = lr
        self.lambda_ = lambda_
        self.penalty = penalty
        
    def softmax(self, X):
        """
        Compute the softmax activation function.

        Parameters
        ----------
        X : numpy array
            The input data of shape (n_samples, n_classes).

        Returns
        -------
        numpy array
            The output of the softmax function of shape (n_samples, n_classes).
        """
        return np.exp(X)/ np.sum(np.exp(X), axis = 1).reshape(-1,1)
    
    def fit(self, X, Y):
        """
        Fit the multinomial logistic regression model to the training data.

        Parameters
        ----------
        X : numpy array
            The input data of shape (n_samples, n_features).
        Y : numpy array
            The target values of shape (n_samples,).

        Returns
        -------
        None
        """
        if not isinstance(X, (np.ndarray, np.generic)):
            X = np.array(X)
        self.theta, self.learning_curve = self.Gradient_descent(X, Y, self.max_iter,
                                                                self.lr, self.lambda_, self.penalty)

    def predict(self, X_new):
        """
        Predict the class labels for new data.

        Parameters
        ----------
        X_new : numpy array
            The input data of shape (n_samples, n_features).

        Returns
        -------
        numpy array
            The predicted class labels of shape (n_samples,).
        """
        if isinstance(X_new, (pd.Series, pd.DataFrame)):
            X_new = np.array(X_new)
        Z = -X_new @ self.theta
        pr = self.softmax(Z)
        pred = np.argmax(pr, axis=1)
        return np.array([self.label_key[value] for value in pred])
    
    def Gradient_descent(self, X, Y, max_iter, lr, lambda_, penalty):
        """
        Perform gradient descent to optimize the cost function and obtain the weights.

        Parameters
        ----------
        X : numpy array
            The input data of shape (n_samples, n_features).
        Y : numpy array
            The target values of shape (n_samples,).
        max_iter : int
            Maximum number of iterations for gradient descent
        lr : float
            Learning rate for gradient descent
        lambda_ : float
            Regularization parameter
        penalty : {'l1', 'l2'}
            Type of regularization penalty to use

        Returns
        -------
        theta : numpy array
            The weights of the logistic regression model of shape (n_features, n_classes).
        learning_df : pandas dataframe
            A dataframe containing the iteration number and loss for each iteration of gradient descent.
        """
        def Cost(X, Y, theta, lambda_, penalty):
            Z = - X @ theta
            m = X.shape[0]
            loss = 1/m * (np.trace(X @ theta @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis = 1)))) 
            if penalty == 'l2':
                regularization = lambda_/(2*m) * np.sum(np.square(theta[1:]))  # exclude bias term from regularization
            elif penalty == 'l1':
                regularization = lambda_/(m) * np.sum(np.abs(theta[1:]))
            else:
                regularization = 0
            cost = loss + regularization
            return cost
        def Gradient(X, Y, theta, lambda_, penalty):
            Z = - X @ theta
            pr = self.softmax(Z)
            m = X.shape[0]
            d_loss = 1/m * (X.T @ (Y-pr))
            if penalty == 'l2':
                d_regul =  lambda_ * theta
            elif penalty == 'l1':
                d_regul = lambda_ * np.sign(theta)
            else:
                d_regul = np.zeros(theta.shape)
            d_regul[0, :] = 0
            gradient = d_loss + d_regul
            return gradient
        def OneHot(y):
            u = np.unique(y)
            n_unique = u.size
            n_sample = len(y)
            label_key = dict(zip(u, range(n_unique)))
            encoded_array = np.zeros((n_sample, n_unique), dtype=int)
            encoded_array[np.arange(n_sample), [label_key[value] for value in y]] = 1
            # switch label key for prediction access
            self.label_key = dict(zip(label_key.values(), label_key.keys()))
            return encoded_array
        Y_one = OneHot(Y)
        theta = np.zeros((X.shape[1], Y_one.shape[1]))
        iter_ = 0
        learning_curve = []
        while iter_ < max_iter:
            iter_ += 1
            theta = theta - lr * Gradient(X, Y_one, theta, lambda_, penalty)
            learning_curve.append(Cost(X, Y_one, theta, lambda_, penalty))
        learning_df = pd.DataFrame({
            'iter': range(iter_),
            'loss': learning_curve
        })
        return theta, learning_df

    def loss_plot(self):
        """
        Plots the learning curve.

        Returns:
        --------
        A matplotlib plot of the learning curve.
        """
        return self.learning_curve.plot(
            x='iter',
            y='loss',
            xlabel='iter',
            ylabel='loss'
        )

## (2) Multi-layer Neural Network
### Dense layer
class Linear:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
    # Set weights and biases in a layer instance
    def set_parameters ( self , weights , biases ):
        self.weights = weights
        self.biases = biases

### Input "layer"
class Layer_Input:

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs

### ReLU activation
class ReLU:

    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

### Softmax activation
class Softmax:

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

### SGD optimizer
class SGD:

    # Initialize optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)

                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

### Adam optimizer
class Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)

        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

### Common loss class
class Loss:

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data losses
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        return data_loss

    # Calculates accumulated loss
    def calculate_accumulated(self):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        return data_loss

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

### Cross-entropy loss(Negative log likelihood loss)
class NLLLOSS(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

### Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class CrossEntropyLoss(Loss):

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

### Common accuracy class
class Accuracy:

    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self):

        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return the data losses
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

### Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):

    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary

    # No initialization is needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

### Model class
class MNN:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the model
    def Sequential(self, *args):
        for layer in args:
            self.layers.append(layer)

    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
        if loss is not None :
            self.loss = loss
        if optimizer is not None :
            self.optimizer = optimizer
        if accuracy is not None :
            self.accuracy = accuracy

    # Finalize the model
    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])


        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(
            self.trainable_layers
        )

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Softmax) and \
           isinstance(self.loss, NLLLOSS):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = \
                CrossEntropyLoss()

    # Train the model
    def train(self, X, y, *, n_epochs=1, batch_size=None,
              print_every=1, validation_data=None, verbose = True):
        # Encode y
        u = np.unique(y)
        n_unique = u.size
        self.label_key = dict(zip(u, range(n_unique)))
        y = np.array([self.label_key[val] for val in y])
    
        self.n_epochs = n_epochs
        # Initialize accuracy object
        self.accuracy.init(y)
        
        # Store learning curve:
        # [epoch_accuracy, epoch_loss, validation_accuracy, validation_loss]
        self.loss_values = []
        # Default value if batch size is not being set
        train_steps = 1

        # If there is validation data passed,
        # set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1

            # For better readability
            X_val, y_val = validation_data
            y_val = np.array([self.label_key[val] for val in y_val])
        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size

                # Dividing rounds down. If there are some remaining
                # data but nor full batch, this won't include it
                # Add `1` to include this not full batch
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Main training loop
        for epoch in range(1, n_epochs+1):

            # Print epoch number
            if verbose:
                print(f'epoch: {epoch}')

            # Reset loss and accuracy in each epoch
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss = self.loss.calculate(output, batch_y)
                loss = data_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(
                                  output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print a summary
                if verbose:
                    if not step % print_every or step == train_steps - 1:
                        print(f'step: {step}, ' +
                              f'acc: {accuracy:.3f}, ' +
                              f'loss: {loss:.3f} (' +
                              f'data_loss: {data_loss:.3f}, ' +
                              f'lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss = self.loss.calculate_accumulated()
            epoch_loss = epoch_data_loss 
            epoch_accuracy = self.accuracy.calculate_accumulated()
            if verbose:
                print(f'training, ' +
                      f'acc: {epoch_accuracy:.3f}, ' +
                      f'loss: {epoch_loss:.3f} (' +
                      f'data_loss: {epoch_data_loss:.3f}, ' +
                      f'lr: {self.optimizer.current_learning_rate}')

            # If there is the validation data
            if validation_data is not None:

                # Reset accumulated values in loss
                # and accuracy objects
                self.loss.new_pass()
                self.accuracy.new_pass()

                # Iterate over steps
                for step in range(validation_steps):

                    # If batch size is not set -
                    # train using one step and full dataset
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val


                    # Otherwise slice a batch
                    else:
                        batch_X = X_val[
                            step*batch_size:(step+1)*batch_size
                        ]
                        batch_y = y_val[
                            step*batch_size:(step+1)*batch_size
                        ]

                    # Perform the forward pass
                    output = self.forward(batch_X, training=False)

                    # Calculate the loss
                    self.loss.calculate(output, batch_y)

                    # Get predictions and calculate an accuracy
                    predictions = self.output_layer_activation.predictions(
                                      output)
                    self.accuracy.calculate(predictions, batch_y)

                # Get and print validation loss and accuracy
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()
                if verbose:
                    print(f'validation, ' +
                          f'acc: {validation_accuracy:.3f}, ' +
                          f'loss: {validation_loss:.3f}')
            else:
                validation_accuracy = None
                validation_loss = None
            self.loss_values.append([epoch_accuracy, epoch_loss,
                         validation_accuracy, validation_loss])
    
    # Predicts on the samples
    def predict(self, X, *, batch_size=None):

            # Default value if batch size is not being set
            prediction_steps = 1

            # Calculate number of steps
            if batch_size is not None:
                prediction_steps = len(X) // batch_size

                # Dividing rounds down. If there are some remaining
                # data but not a full batch, this won't include it
                # Add `1` to include this not full batch
                if prediction_steps * batch_size < len(X):
                    prediction_steps += 1

            # Model outputs
            output = []

            # Iterate over steps
            for step in range(prediction_steps):

                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X

                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]

                # Perform the forward pass
                batch_output = self.forward(batch_X, training=False)

                # Append batch prediction to the list of predictions
                output.append(batch_output)

            # Stack and return results
            pr = np.vstack(output)
            pred = np.argmax(pr, axis=1)
            output_label_key = dict(zip(self.label_key.values(), self.label_key.keys()))
            return np.array([output_label_key[value] for value in pred])

    # Performs forward pass
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # return its output
        return layer.output

    # Performs backward pass
    def backward(self, output, y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):
        # Create a list for parameters
        parameters = []
        # Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        # Return a list
        return parameters

    # Updates the model with new parameters
    def set_parameters(self, parameters):
        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Saves the parameters to a file
    def save(self, path):
        # Open a file in the binary-write mode
        # and save parameters to it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Loads the weights and updates a model instance with them
    def load(self, path):
        # Open file in the binary-read mode,
        # load weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
    
    # Plots the learning curve.
    def loss_plot(self):
        learning_df = pd.DataFrame(self.loss_values,
                                   columns = ['training_accuracy', 'training_loss',
                                              'validation_accuracy', 'validation_loss'])
        learning_df['epoch'] = range(1, self.n_epochs+1)
        self.learning_df = learning_df
        
        learning_df.set_index('epoch', inplace=True)
        learning_df[['training_loss', 'validation_loss']].plot()

#==============================================================================================
# Cross Validation
def CV(X_pre, y, k_fold, stratified, clf, scoref):
    # Pre-deifined function to split data
    def CV_data(X, y, k_fold=5, stratified=True):
        X = np.array(X)
        n_sample = X.shape[0]
        keys = np.array(range(n_sample))
        np.random.shuffle(keys)
        X, y = X[keys], y[keys] # shuffle
        temp_return_data = []
        if stratified:
            category = y.unique()
            category_list = []
            n_unit_list = []
            for label in category:
                index = y[y == label].index
                category_list.append(index)
                n_unit_list.append(len(index) // k_fold)
            for i in range(k_fold):
                one_fold = [[], []]
                for j in range(len(category)):
                    category_idx = category_list[j]
                    n_unit = n_unit_list[j]
                    start_idx = i*n_unit
                    end_idx = (i+1)*n_unit
                    if i != (k_fold-1):
                        original_idx = category_idx[start_idx:end_idx]
                    else:
                        original_idx = category_idx[start_idx:]
                    one_fold[0] += X[original_idx, ].tolist()
                    one_fold[1] += y[original_idx].tolist()
                temp_return_data.append([np.array(one_fold[0]), np.array(one_fold[1])])     
        else:
            n_unit = n_sample // k_fold
            for i in range(k_fold):
                start_idx = i*n_unit
                end_idx = (i+1)*n_unit
                if i != (k_fold-1):
                    temp_return_data.append([X[start_idx:end_idx, ], y[start_idx:end_idx, ]])
                else:
                    temp_return_data.append([X[start_idx:, ], y[start_idx:, ]])
        return_data = []
        # Organize data for modeling
        for i in range(k_fold):
            train_idx = [j for j in range(k_fold) if j != i]
            X_train = np.concatenate([temp_return_data[j][0] for j in range(k_fold) if j in train_idx])
            y_train = np.concatenate([temp_return_data[j][1] for j in range(k_fold) if j in train_idx])
            return_data.append([[X_train, y_train], temp_return_data[i]])
        # return_data = [[[X_train, y_train]], [[X_val, y_val]],...]
        return return_data
    # (1) Split data by CV
    data = CV_data(X_pre, y, k_fold, stratified)
    # (2) Iterate each CV data
    ## (2-2) Modeling
    ## (2-3) Evaluation
    idx = 1
    metric_list = []
    for df_train, df_val in data:
        # Currently, this part is flexible
        clf.fit(df_train[0], df_train[1])
        pred = clf.predict(df_val[0])
        metric = scoref(pred, df_val[1])
        metric_list.append(metric)
        print(f'{idx} th fold is finished!')
        idx += 1
    return pd.DataFrame({'Fold': ['Fold_' + str(i) for i in range(1, k_fold+1)],
                        'Accuracy': metric_list})

#==============================================================================================
# Preprocessing
## (1) Text Prep
def text_preprocessing(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_word_ntlk = list(stopwords.words("english"))
    sentence_words_1 = RegexpTokenizer("[a-zA-Z]*[aeiouAEIOU][a-zA-Z]+|[a-zA-Z]+[aeiouAEIOU][a-zA-Z]*").tokenize(sentence.lower())
    sentence_words_2 = [lemmatizer.lemmatize(w, pos = "v") for w in sentence_words_1]
    sentence_words_3 = [i for i in sentence_words_2 if not i in stop_word_ntlk]
    return sentence_words_3

#==============================================================================================
# Feature Transformation
## (1) Bag-of-Word
def BoW(X, X_new = None):
    """
    Return df with each word as columns
    """
    if isinstance(X_new, pd.Series):
        n_sample = X_new.size
        X_iter = X_new
    else:
        n_sample = X.size
        X_iter = X
    X_word = X.sum()
    u = np.unique(X_word)
    n_unique = u.size
    label_key = dict(zip(u, range(n_unique)))
    encoded_array = np.zeros((n_sample, n_unique), dtype=int)
    for idx, row in X_iter.items():
        for value in row:
            if value in label_key.keys():
                encoded_array[idx, label_key[value]] += 1
    bow_df = pd.DataFrame(encoded_array)
    bow_df.columns = u
    return bow_df

## (2) BertTweet
# def BertTweetEmbed(X):
#     tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", normalization=True)
#     bertweet = AutoModel.from_pretrained("vinai/bertweet-large")    
#     encodings = tokenizer(X.squeeze().values.tolist(), return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         last_hidden_states = bertweet(encodings['input_ids'], attention_mask=encodings['attention_mask']).last_hidden_state
#     input_features = torch.mean(last_hidden_states, dim=1).numpy()
#     return input_features

#==============================================================================================
# Metric
## (1) Accuracy
def accuracy_score(y_pred, y_true):
    if len(y_pred.shape) == 2:
        y = np.argmax(y_pred, axis=1)
    else:
        y = y_pred
    return np.mean(y == y_true)

#==============================================================================================
# Main functions for the assignment
def LR(X_train, y_train, X_test, test_df, max_iter=1000, lr=0.01, lambda_=0.1, penalty = 'l2'):
    # Initiate model
    clf = MulLR(max_iter=max_iter, lr=lr, lambda_=lambda_, penalty = penalty)
    # Fit
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    export_df = test_df.copy()
    export_df['emotions'] = pred
    # Export
    export_df.to_csv('test_lr.csv', index=False)
    return None

def NN(X_train, y_train, X_test, test_df):
    # Instantiate the model
    model = MNN()
    # Architecture
    model.Sequential(
        Linear(X_train.shape[1], 64),
        ReLU(),
        Linear(64, 128),
        ReLU(),
        Linear(128, 6),
        Softmax()
    )
    # Components 
    model.set(
        loss=NLLLOSS(),
        optimizer=Adam(learning_rate=0.0001, decay = 0.001),
        accuracy=Accuracy_Categorical()
    )

    # Finalize the model
    model.finalize()
    # Train the model
    model.train(X_train, y_train, validation_data=None,
                n_epochs=50, batch_size=128, print_every=100, verbose = False)
    # Predict
    pred = model.predict(X_test)
    export_df = test_df.copy()
    export_df['emotions'] = pred
    # Export
    export_df.to_csv('test_nn.csv', index=False)
    return None

if __name__ == '__main__':
    # Load data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    X_train, y_train = train_df[['text']], train_df['emotions']
    X_test = test_df[['text']]
    # Text preprocessing(for BoW)
    X_train_processed = X_train['text'].map(text_preprocessing)
    X_test_processed = X_test['text'].map(text_preprocessing)
    # Bow
    X_train_bow = BoW(X_train_processed)
    X_test_bow = BoW(X_train_processed, X_test_processed)
    print ("..................Beginning of Logistic Regression................")
    LR(X_train_bow, y_train, X_test_bow, test_df, max_iter=1000, lr=0.01, lambda_=0.1, penalty = 'l2')
    print ("..................End of Logistic Regression................")

    print("------------------------------------------------")

    print ("..................Beginning of Neural Network................")
    NN(X_train_bow, y_train, X_test_bow, test_df)
    print ("..................End of Neural Network................")
