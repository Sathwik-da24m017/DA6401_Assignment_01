import numpy as np
import pandas as pd
import argparse
import wandb
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist
import time
import os

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid', weight_init='random', weight_decay=0.0):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.weight_decay = weight_decay
        
        # Initialize weights and biases
        self.parameters = {}
        self.gradients = {}
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights based on the specified method
        for i in range(1, len(self.layer_sizes)):
            if weight_init == 'random':
                self.parameters[f'W{i}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * 0.01
            elif weight_init == 'Xavier':
                # Xavier initialization
                self.parameters[f'W{i}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(1 / self.layer_sizes[i-1])
            
            self.parameters[f'b{i}'] = np.zeros((self.layer_sizes[i], 1))
        
        # Initialize optimizer-specific variables
        self.v = {}  # For momentum, nag, adam, nadam
        self.s = {}  # For rmsprop, adam, nadam
        self.m_hat = {}  # For nadam
        self.v_hat = {}  # For nadam
        
        for i in range(1, len(self.layer_sizes)):
            self.v[f'W{i}'] = np.zeros_like(self.parameters[f'W{i}'])
            self.v[f'b{i}'] = np.zeros_like(self.parameters[f'b{i}'])
            self.s[f'W{i}'] = np.zeros_like(self.parameters[f'W{i}'])
            self.s[f'b{i}'] = np.zeros_like(self.parameters[f'b{i}'])
            self.m_hat[f'W{i}'] = np.zeros_like(self.parameters[f'W{i}'])
            self.m_hat[f'b{i}'] = np.zeros_like(self.parameters[f'b{i}'])
            self.v_hat[f'W{i}'] = np.zeros_like(self.parameters[f'W{i}'])
            self.v_hat[f'b{i}'] = np.zeros_like(self.parameters[f'b{i}'])
    
    def activation_function(self, Z, activation=None):
        """Apply activation function to the input Z"""
        if activation is None:
            activation = self.activation
            
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'ReLU':
            return np.maximum(0, Z)
        elif activation == 'identity':
            return Z
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def activation_derivative(self, Z, activation=None):
        """Compute the derivative of the activation function"""
        if activation is None:
            activation = self.activation
            
        if activation == 'sigmoid':
            A = self.activation_function(Z, 'sigmoid')
            return A * (1 - A)
        elif activation == 'tanh':
            return 1 - np.power(np.tanh(Z), 2)
        elif activation == 'ReLU':
            return (Z > 0).astype(float)
        elif activation == 'identity':
            return np.ones_like(Z)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def softmax(self, Z):
        """Compute softmax values for each set of scores in Z"""
        # Shift Z for numerical stability
        shifted_Z = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(shifted_Z)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def forward_propagation(self, X):
        """Forward propagation through the network"""
        cache = {'A0': X}
        A = X
        L = len(self.layer_sizes) - 1  # Number of layers
        
        # Forward propagation through hidden layers
        for l in range(1, L):
            Z = np.dot(self.parameters[f'W{l}'], A) + self.parameters[f'b{l}']
            cache[f'Z{l}'] = Z
            A = self.activation_function(Z)
            cache[f'A{l}'] = A
        
        # Output layer (softmax for classification)
        Z = np.dot(self.parameters[f'W{L}'], A) + self.parameters[f'b{L}']
        cache[f'Z{L}'] = Z
        
        # Use softmax for the output layer
        A = self.softmax(Z)
        cache[f'A{L}'] = A
        
        return A, cache
    
    def compute_loss(self, Y_pred, Y, loss_type='cross_entropy'):
        """Compute the loss based on predictions and true labels"""
        m = Y.shape[1]
        
        if loss_type == 'cross_entropy':
            # Cross-entropy loss for classification
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            log_probs = -np.log(Y_pred + epsilon) * Y
            loss = np.sum(log_probs) / m
            
            # Add L2 regularization if weight_decay > 0
            if self.weight_decay > 0:
                l2_reg = 0
                for l in range(1, len(self.layer_sizes)):
                    l2_reg += np.sum(np.square(self.parameters[f'W{l}']))
                loss += (self.weight_decay / (2 * m)) * l2_reg
                
            return loss
            
        elif loss_type == 'mean_squared_error':
            # Mean squared error
            loss = np.mean(np.sum((Y_pred - Y) ** 2, axis=0)) / 2
            
            # Add L2 regularization if weight_decay > 0
            if self.weight_decay > 0:
                l2_reg = 0
                for l in range(1, len(self.layer_sizes)):
                    l2_reg += np.sum(np.square(self.parameters[f'W{l}']))
                loss += (self.weight_decay / (2 * m)) * l2_reg
                
            return loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def backward_propagation(self, Y, cache, loss_type='cross_entropy'):
        """Backward propagation to compute gradients"""
        m = Y.shape[1]
        L = len(self.layer_sizes) - 1  # Number of layers
        
        # Initialize gradients dictionary
        gradients = {}
        
        # Output layer gradient
        if loss_type == 'cross_entropy':
            # For softmax + cross-entropy, the gradient is simplified
            dZ = cache[f'A{L}'] - Y
        elif loss_type == 'mean_squared_error':
            # For MSE
            dZ = (cache[f'A{L}'] - Y) * self.activation_derivative(cache[f'Z{L}'], 'identity')
        
        # Compute gradients for the output layer
        gradients[f'dW{L}'] = (1/m) * np.dot(dZ, cache[f'A{L-1}'].T)
        gradients[f'db{L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Add L2 regularization gradient if weight_decay > 0
        if self.weight_decay > 0:
            gradients[f'dW{L}'] += (self.weight_decay / m) * self.parameters[f'W{L}']
        
        # Backpropagate through hidden layers
        for l in reversed(range(1, L)):
            dA = np.dot(self.parameters[f'W{l+1}'].T, dZ)
            dZ = dA * self.activation_derivative(cache[f'Z{l}'])
            
            gradients[f'dW{l}'] = (1/m) * np.dot(dZ, cache[f'A{l-1}'].T)
            gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            # Add L2 regularization gradient if weight_decay > 0
            if self.weight_decay > 0:
                gradients[f'dW{l}'] += (self.weight_decay / m) * self.parameters[f'W{l}']
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate, optimizer='sgd', momentum=0.9, 
                          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
        """Update parameters using the specified optimizer"""
        L = len(self.layer_sizes) - 1  # Number of layers
        
        if optimizer == 'sgd':
            # Standard gradient descent
            for l in range(1, L + 1):
                self.parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
                self.parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']
                
        elif optimizer == 'momentum':
            # Momentum-based gradient descent
            for l in range(1, L + 1):
                self.v[f'W{l}'] = momentum * self.v[f'W{l}'] + (1 - momentum) * gradients[f'dW{l}']
                self.v[f'b{l}'] = momentum * self.v[f'b{l}'] + (1 - momentum) * gradients[f'db{l}']
                
                self.parameters[f'W{l}'] -= learning_rate * self.v[f'W{l}']
                self.parameters[f'b{l}'] -= learning_rate * self.v[f'b{l}']
                
        elif optimizer == 'nag':
            # Nesterov Accelerated Gradient
            for l in range(1, L + 1):
                v_prev_W = self.v[f'W{l}']
                v_prev_b = self.v[f'b{l}']
                
                self.v[f'W{l}'] = momentum * self.v[f'W{l}'] - learning_rate * gradients[f'dW{l}']
                self.v[f'b{l}'] = momentum * self.v[f'b{l}'] - learning_rate * gradients[f'db{l}']
                
                self.parameters[f'W{l}'] += -momentum * v_prev_W + (1 + momentum) * self.v[f'W{l}']
                self.parameters[f'b{l}'] += -momentum * v_prev_b + (1 + momentum) * self.v[f'b{l}']
                
        elif optimizer == 'rmsprop':
            # RMSprop
            for l in range(1, L + 1):
                self.s[f'W{l}'] = beta * self.s[f'W{l}'] + (1 - beta) * np.square(gradients[f'dW{l}'])
                self.s[f'b{l}'] = beta * self.s[f'b{l}'] + (1 - beta) * np.square(gradients[f'db{l}'])
                
                self.parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}'] / (np.sqrt(self.s[f'W{l}']) + epsilon)
                self.parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}'] / (np.sqrt(self.s[f'b{l}']) + epsilon)
                
        elif optimizer == 'adam':
            # Adam optimizer
            for l in range(1, L + 1):
                # Update biased first moment estimate
                self.v[f'W{l}'] = beta1 * self.v[f'W{l}'] + (1 - beta1) * gradients[f'dW{l}']
                self.v[f'b{l}'] = beta1 * self.v[f'b{l}'] + (1 - beta1) * gradients[f'db{l}']
                
                # Update biased second moment estimate
                self.s[f'W{l}'] = beta2 * self.s[f'W{l}'] + (1 - beta2) * np.square(gradients[f'dW{l}'])
                self.s[f'b{l}'] = beta2 * self.s[f'b{l}'] + (1 - beta2) * np.square(gradients[f'db{l}'])
                
                # Bias correction
                v_corrected_W = self.v[f'W{l}'] / (1 - np.power(beta1, t))
                v_corrected_b = self.v[f'b{l}'] / (1 - np.power(beta1, t))
                s_corrected_W = self.s[f'W{l}'] / (1 - np.power(beta2, t))
                s_corrected_b = self.s[f'b{l}'] / (1 - np.power(beta2, t))
                
                # Update parameters
                self.parameters[f'W{l}'] -= learning_rate * v_corrected_W / (np.sqrt(s_corrected_W) + epsilon)
                self.parameters[f'b{l}'] -= learning_rate * v_corrected_b / (np.sqrt(s_corrected_b) + epsilon)
                
        elif optimizer == 'nadam':
            # Nadam optimizer (Adam with Nesterov momentum)
            for l in range(1, L + 1):
                # Update biased first moment estimate
                self.v[f'W{l}'] = beta1 * self.v[f'W{l}'] + (1 - beta1) * gradients[f'dW{l}']
                self.v[f'b{l}'] = beta1 * self.v[f'b{l}'] + (1 - beta1) * gradients[f'db{l}']
                
                # Update biased second moment estimate
                self.s[f'W{l}'] = beta2 * self.s[f'W{l}'] + (1 - beta2) * np.square(gradients[f'dW{l}'])
                self.s[f'b{l}'] = beta2 * self.s[f'b{l}'] + (1 - beta2) * np.square(gradients[f'db{l}'])
                
                # Bias correction
                v_corrected_W = self.v[f'W{l}'] / (1 - np.power(beta1, t))
                v_corrected_b = self.v[f'b{l}'] / (1 - np.power(beta1, t))
                s_corrected_W = self.s[f'W{l}'] / (1 - np.power(beta2, t))
                s_corrected_b = self.s[f'b{l}'] / (1 - np.power(beta2, t))
                
                # Nesterov momentum update
                v_nesterov_W = beta1 * v_corrected_W + (1 - beta1) * gradients[f'dW{l}'] / (1 - np.power(beta1, t))
                v_nesterov_b = beta1 * v_corrected_b + (1 - beta1) * gradients[f'db{l}'] / (1 - np.power(beta1, t))
                
                # Update parameters
                self.parameters[f'W{l}'] -= learning_rate * v_nesterov_W / (np.sqrt(s_corrected_W) + epsilon)
                self.parameters[f'b{l}'] -= learning_rate * v_nesterov_b / (np.sqrt(s_corrected_b) + epsilon)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    def predict(self, X):
        """Make predictions on input data X"""
        A, _ = self.forward_propagation(X)
        return np.argmax(A, axis=0)
    
    def one_hot_encode(self, y, num_classes):
        """Convert class labels to one-hot encoding"""
        m = y.shape[0]
        y_one_hot = np.zeros((num_classes, m))
        y_one_hot[y.flatten(), np.arange(m)] = 1
        return y_one_hot
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.01, 
              optimizer='sgd', momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, 
              loss_type='cross_entropy', use_wandb=True):
        """Train the neural network"""
        # Convert labels to one-hot encoding
        Y_train = self.one_hot_encode(y_train, self.output_size)
        Y_val = self.one_hot_encode(y_val, self.output_size)
        
        # Number of training examples
        m = X_train.shape[1]
        
        # Number of complete mini-batches
        num_complete_minibatches = m // batch_size
        
        # Initialize lists to store metrics
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Shuffle the training data
            permutation = list(np.random.permutation(m))
            shuffled_X = X_train[:, permutation]
            shuffled_Y = Y_train[:, permutation]
            
            # Initialize epoch loss and accuracy
            epoch_train_loss = 0
            epoch_train_correct = 0
            
            # Mini-batch training
            for k in range(num_complete_minibatches):
                # Get mini-batch
                mini_batch_start = k * batch_size
                mini_batch_end = min((k + 1) * batch_size, m)
                mini_batch_X = shuffled_X[:, mini_batch_start:mini_batch_end]
                mini_batch_Y = shuffled_Y[:, mini_batch_start:mini_batch_end]
                mini_batch_Y = shuffled_Y[:, mini_batch_start:mini_batch_end]
                
                # Forward propagation
                A, cache = self.forward_propagation(mini_batch_X)
                
                # Compute loss
                mini_batch_loss = self.compute_loss(A, mini_batch_Y, loss_type)
                epoch_train_loss += mini_batch_loss
                
                # Backward propagation
                gradients = self.backward_propagation(mini_batch_Y, cache, loss_type)
                
                # Update parameters
                self.update_parameters(gradients, learning_rate, optimizer, momentum, beta, beta1, beta2, epsilon, epoch + 1)
                
                # Compute accuracy for the mini-batch
                predictions = np.argmax(A, axis=0)
                labels = np.argmax(mini_batch_Y, axis=0)
                epoch_train_correct += np.sum(predictions == labels)
            
            # Compute metrics for the epoch
            train_loss = epoch_train_loss / num_complete_minibatches
            train_accuracy = epoch_train_correct / m
            
            # Validation metrics
            A_val, _ = self.forward_propagation(X_val)
            val_loss = self.compute_loss(A_val, Y_val, loss_type)
            val_predictions = np.argmax(A_val, axis=0)
            val_labels = np.argmax(Y_val, axis=0)
            val_accuracy = np.sum(val_predictions == val_labels) / X_val.shape[1]
            
            # Append metrics to lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            # Log metrics to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy
                })
            
            # Print epoch summary
            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s")
        
        return train_losses, val_losses, train_accuracies, val_accuracies

def load_data(dataset_name):
    """Load and preprocess the dataset"""
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Reshape and normalize data
    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0
    
    # Manually split training data to create validation set
    np.random.seed(42)
    m = X_train.shape[1]
    val_size = int(0.1 * m)
    
    # Create random permutation of indices
    indices = np.random.permutation(m)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Split the data
    X_val = X_train[:, val_indices]
    y_val = y_train[val_indices]
    X_train = X_train[:, train_indices]
    y_train = y_train[train_indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_sample_images(dataset_name):
    """Plot sample images from the dataset"""
    if dataset_name == 'mnist':
        (X_train, y_train), (_, _) = mnist.load_data()
        class_names = [str(i) for i in range(10)]
    elif dataset_name == 'fashion_mnist':
        (X_train, y_train), (_, _) = fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    plt.figure(figsize=(10, 10))
    for i in range(10):
        # Find first instance of each class
        idx = np.where(y_train == i)[0][0]
        plt.subplot(3, 4, i+1)
        plt.imshow(X_train[idx], cmap='gray')
        plt.title(class_names[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    
    if wandb.run is not None:
        wandb.log({"sample_images": wandb.Image('sample_images.png')})

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST or Fashion-MNIST dataset')
    
    parser.add_argument('-wp', '--wandb_project', default='myprojectname', 
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='myname', 
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use for training')
    parser.add_argument('-e', '--epochs', type=int, default=10, 
                        help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=32, 
                        help='Batch size used to train neural network')
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function to use')
    parser.add_argument('-o', '--optimizer', default='adam', 
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, 
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, 
                        help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.9, 
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9, 
                        help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999, 
                        help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-8, 
                        help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, 
                        help='Weight decay used by optimizers')
    parser.add_argument('-w_i', '--weight_init', default='Xavier', choices=['random', 'Xavier'],
                        help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, 
                        help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128, 
                        help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', default='ReLU', 
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        help='Activation function to use')
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)
    
    # Plot sample images
    plot_sample_images(args.dataset)
    
    # Create hidden layer sizes based on num_layers and hidden_size
    hidden_sizes = [args.hidden_size] * args.num_layers
    
    # Create and train the neural network
    input_size = X_train.shape[0]  # Number of features (784 for MNIST/Fashion-MNIST)
    output_size = 10  # Number of classes
    
    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=args.activation,
        weight_init=args.weight_init,
        weight_decay=args.weight_decay
    )
    
    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        loss_type=args.loss,
        use_wandb=True
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Log test accuracy to wandb
    wandb.log({"test_accuracy": test_accuracy})
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()

                                     