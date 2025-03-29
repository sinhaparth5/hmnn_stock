import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

class BayesianLayer(nn.Module):
    """
    A Bayesian layer that models weights as random variables.
    
    In Bayesian Neural Networks, weights are distributions rather than
    fixed values, allowing the network to express uncertainty.
    """
    def __init__(self, in_features, out_features, prior_mean=0.0, prior_std=1.0):
        super(BayesianLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Parameters of the prior distribution p[w]
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        # Parameters of the posterior distribution q(w|D)
        # We learn these parameters during training
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3))
        
        # For bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).fill_(0))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).fill_(-3))
        
    def forward(self, x, sample=True):
        """
        Forward pass through the Bayesian layer.

        Args:
            x : Input Tensor
            sample (bool, optional): If True, sample weights from posterior. If False, use the mean.
            
        Returns:
            Output of the layer
        """
        if sample:
            # Sample weights using the representation trick
            weight_epsilon = torch.randn_like(self.weight_mu)
            bias_epsilon = torch.rand_like(self.bias_mu)
            
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * weight_epsilon
            bias = self.bias_mu + torch.exp(self.bias_log_sigma) * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """
        Calculate KL divergence between posterior and prior distribution
        
        KL(q(w|D) || p(w)) measures how much the posterior diverges from the prior
        This serves as a regularization term in variational interence
        
        Returns:
            KL divergence value
        """
        q_w = Normal(self.weight_mu, torch.exp(self.weight_log_sigma))
        p_w = Normal(torch.zeros_like(self.weight_mu) + self.prior_mean, 
                    torch.zeros_like(self.weight_log_sigma) + self.prior_std)
        
        q_b = Normal(self.bias_mu, torch.exp(self.bias_log_sigma))
        p_b = Normal(torch.zeros_like(self.bias_mu) + self.prior_mean,
                    torch.zeros_like(self.bias_log_sigma) + self.prior_std)
        
        # Calculate KL for weights and biases
        kl_weights = kl_divergence(q_w, p_w).sum()
        kl_bias = kl_divergence(q_b, p_b).sum()
        
        return kl_weights + kl_bias
    
class GaussianMixtureLayer(BayesianLayer):
    """
    Extension of BayesianLayer using a mixture of Gaussians for the posterior.
    
    This implements the Variational DropConnect technique described in the paper,
    which models weights as a mixture of a "normal" component and a "dropped" component.
    """
    def __init__(self, in_features, out_features, prior_mean=0.0, prior_std=1.0, 
                 drop_prob=0.5, drop_std=0.01):
        super(GaussianMixtureLayer, self).__init__(
            in_features, out_features, prior_mean, prior_std
        )
        
        # Mixture parameters
        self.drop_prob = drop_prob
        self.drop_mean = 0.0  # Typically zero for the "dropped" weights
        self.drop_std = drop_std  # Small standard deviation for the "dropped" component
        
    def forward(self, x, sample=True, component=None):
        """
        Forward pass with mixture sampling.
        
        Args:
            x: Input tensor
            sample: Whether to sample weights or use means
            component: If provided, use a specific mixture component:
                      0 = main weights, 1 = dropped weights
        
        Returns:
            Output of the layer
        """
        if not sample:
            # Use posterior mean (similar to standard BayesianLayer)
            return super().forward(x, sample=False)
        
        # Determine which component to sample from
        if component is None:
            # Randomly choose component based on drop probability
            use_drop = torch.rand(1).item() < self.drop_prob
            component = 1 if use_drop else 0
        
        if component == 0:
            # Sample from main component (regular weights)
            return super().forward(x, sample=True)
        else:
            # Sample from "drop" component (near-zero weights)
            weight_epsilon = torch.randn_like(self.weight_mu)
            bias_epsilon = torch.randn_like(self.bias_mu)
            
            weight = torch.zeros_like(self.weight_mu) + self.drop_mean + self.drop_std * weight_epsilon
            bias = torch.zeros_like(self.bias_mu) + self.drop_mean + self.drop_std * bias_epsilon
            
            return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """
        Compute approximate KL divergence for mixture posterior.
        
        We make a simplification by computing a weighted average of KL
        for the main component only, scaled by the probability of using it.
        
        Returns:
            Approximate KL divergence value
        """
        # Get KL for the main component
        main_kl = super().kl_divergence()
        
        # Weight by probability of using the main component
        return main_kl * (1 - self.drop_prob)
        