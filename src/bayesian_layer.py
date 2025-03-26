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