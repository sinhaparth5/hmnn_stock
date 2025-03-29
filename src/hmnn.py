import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from bayesian_layer import BayesianLayer, GaussianMixtureLayer
from hmm_transition import MarkovTransitionKernel

class HMNN(nn.Module):
    """
    Hidden Markov Neural Network.
    
    A neural network where weights evolve over time according to
    a hidden Markov model, allowing the model to adapt to changing data.
    """
    def __init__(self, 
                 layer_sizes, 
                 activation=F.relu, 
                 prior_mean=0.0, 
                 prior_std=1.0,
                 transition_std=0.01,
                 use_mixture=False,
                 drop_prob=0.5,
                 time_index=0):
        """
        Initialize the HMNN.
        
        Args:
            layer_sizes: List of integers defining the network architecture
            activation: Activation function to use
            prior_mean: Mean of the prior distribution
            prior_std: Standard deviation of the prior distribution
            transition_std: Standard deviation for weight transitions
            use_mixture: Whether to use mixture of Gaussians (DropConnect)
            drop_prob: Probability of "dropping" weights in DropConnect
            time_index: Current time step index
        """
        super(HMNN, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.use_mixture = use_mixture
        self.time_index = time_index
        
        # Create layers based on specified architecture
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            if use_mixture:
                # Use GaussianMixture layers for DropConnect behavior
                layer = GaussianMixtureLayer(
                    layer_sizes[i], layer_sizes[i+1],
                    prior_mean=prior_mean,
                    prior_std=prior_std,
                    drop_prob=drop_prob
                )
            else:
                # Use standard Bayesian layers
                layer = BayesianLayer(
                    layer_sizes[i], layer_sizes[i+1],
                    prior_mean=prior_mean,
                    prior_std=prior_std
                )
            self.layers.append(layer)
        
        # Create transition kernel for the HMM dynamics
        self.transition_kernel = MarkovTransitionKernel(transition_std=transition_std)
        
        # Store parameter history for analysis
        self.parameter_history = []
        self.save_current_params()
    
    def forward(self, x, sample=True):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            sample: Whether to sample weights or use means
            
        Returns:
            Network output
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, sample=sample)
            if i < len(self.layers) - 1:  # No activation after last layer
                x = self.activation(x)
        return x
    
    def kl_divergence(self):
        """
        Calculate total KL divergence across all layers.
        
        Returns:
            Total KL divergence
        """
        return sum(layer.kl_divergence() for layer in self.layers)
    
    def save_current_params(self):
        """
        Save current parameter state to history.
        Useful for analyzing weight evolution over time.
        """
        params_dict = {}
        for i, layer in enumerate(self.layers):
            params_dict[f'layer_{i}_weight_mu'] = layer.weight_mu.detach().clone()
            params_dict[f'layer_{i}_weight_log_sigma'] = layer.weight_log_sigma.detach().clone()
            params_dict[f'layer_{i}_bias_mu'] = layer.bias_mu.detach().clone()
            params_dict[f'layer_{i}_bias_log_sigma'] = layer.bias_log_sigma.detach().clone()
        
        self.parameter_history.append((self.time_index, params_dict))
    
    def predict(self, x, n_samples=10):
        """
        Make predictions with uncertainty estimation.
        
        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
        """
        predictions = []
        for _ in range(n_samples):
            output = self.forward(x, sample=True)
            predictions.append(output)
        
        # Stack and calculate statistics
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
    
    def predict_proba(self, x, n_samples=10):
        """
        Make probabilistic predictions for classification.
        
        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            mean_probs: Mean class probabilities
            std_probs: Standard deviation of class probabilities
        """
        # Ensure inputs are properly flattened for image data like MNIST
        if len(x.shape) > 2:
            # For image data (batch_size, channels, height, width)
            batch_size = x.size(0)
            x = x.view(batch_size, -1)  # Flatten to (batch_size, features)
        
        predictions = []
        for _ in range(n_samples):
            logits = self.forward(x, sample=True)
            
            # Handle different output shapes
            if logits.shape[-1] > 1:  # Multi-class
                probs = F.softmax(logits, dim=-1)
            else:  # Binary
                probs = torch.sigmoid(logits)
                
            predictions.append(probs)
        
        # Stack and calculate statistics
        predictions = torch.stack(predictions)
        mean_probs = predictions.mean(dim=0)
        std_probs = predictions.std(dim=0)
        
        return mean_probs, std_probs
    
    def transition_to_next_time(self):
        """
        Apply Markov transition to move weights to the next time step.
        
        This is the "prediction" step in the HMM filtering recursion,
        where weights evolve according to the transition kernel.
        """
        # Increment time index
        self.time_index += 1
        
        # For each layer, apply the transition to its parameters
        for layer in self.layers:
            # Apply transition to weight parameters
            new_weight_mu, new_weight_log_sigma = self.transition_kernel.transition(
                layer.weight_mu, layer.weight_log_sigma
            )
            layer.weight_mu.data = new_weight_mu
            layer.weight_log_sigma.data = new_weight_log_sigma
            
            # Apply transition to bias parameters
            new_bias_mu, new_bias_log_sigma = self.transition_kernel.transition(
                layer.bias_mu, layer.bias_log_sigma
            )
            layer.bias_mu.data = new_bias_mu
            layer.bias_log_sigma.data = new_bias_log_sigma
        
        # Save new parameter state
        self.save_current_params()
    
    def get_model_at_time(self, time_index):
        """
        Create a copy of the model with parameters from a specific time.
        
        Args:
            time_index: The time step to retrieve
            
        Returns:
            model_copy: Model with parameters from specified time
        """
        # Find parameters for requested time
        for idx, params_dict in self.parameter_history:
            if idx == time_index:
                # Create new model with same architecture
                model_copy = HMNN(
                    self.layer_sizes,
                    activation=self.activation,
                    prior_mean=self.prior_mean,
                    prior_std=self.prior_std,
                    use_mixture=self.use_mixture,
                    time_index=time_index
                )
                
                # Set parameters to stored values
                for i, layer in enumerate(model_copy.layers):
                    layer.weight_mu.data = params_dict[f'layer_{i}_weight_mu']
                    layer.weight_log_sigma.data = params_dict[f'layer_{i}_weight_log_sigma']
                    layer.bias_mu.data = params_dict[f'layer_{i}_bias_mu']
                    layer.bias_log_sigma.data = params_dict[f'layer_{i}_bias_log_sigma']
                
                return model_copy
        
        raise ValueError(f"No parameters stored for time index {time_index}")