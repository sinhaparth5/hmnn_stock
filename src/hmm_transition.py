import torch
import numpy as np

class MarkovTransitionKernel:
    """
    Implements a transition kernel for the Hidden Markov Model.
    
    The transition kernel governs how weights evolve from one time step
    to the next, implementing the temporal dynamics of the HMNN.
    """
    def __init__(self, transition_std=0.01, transition_momentum=0.0):
        """
        Initialize the transition kernel.
        
        Args:
            transition_std: Standard deviation of the transition noise.
                           Controls how quickly weights can change over time.
            transition_momentum: Optional momentum parameter for directed changes.
                               0.0 means pure random walk.
        """
        self.transition_std = transition_std
        self.transition_momentum = transition_momentum
        self.prev_transition = None
    
    def transition(self, param_mean, param_log_std):
        """
        Apply transition to move parameters to the next time step.
        
        Implements a Gaussian random walk, optionally with momentum.
        
        Args:
            param_mean: Mean parameters from the current time step
            param_log_std: Log standard deviation parameters
            
        Returns:
            new_param_mean, new_param_log_std: Updated parameters
        """
        # Generate Gaussian noise for means
        noise = torch.randn_like(param_mean) * self.transition_std
        
        # Apply momentum if specified
        if self.transition_momentum > 0 and self.prev_transition is not None:
            noise = noise + self.transition_momentum * self.prev_transition
            self.prev_transition = noise
        else:
            self.prev_transition = noise
        
        # Update means with noise
        new_param_mean = param_mean + noise
        
        # Smaller noise for the log_std parameters
        # This makes uncertainty evolve more slowly than the means
        log_std_noise = torch.randn_like(param_log_std) * (self.transition_std * 0.1)
        new_param_log_std = param_log_std + log_std_noise
        
        return new_param_mean, new_param_log_std


class FactorialHMMTransition(MarkovTransitionKernel):
    """
    Implements a factorial HMM transition as described in the paper.
    
    A factorial HMM allows different groups of weights to evolve
    independently, enabling more flexible dynamic behavior.
    """
    def __init__(self, feature_groups=None, group_transition_stds=None):
        """
        Initialize a factorial transition kernel.
        
        Args:
            feature_groups: List of indices for different weight groups
            group_transition_stds: List of transition stds for each group
        """
        super(FactorialHMMTransition, self).__init__()
        self.feature_groups = feature_groups
        self.group_transition_stds = group_transition_stds
    
    def transition(self, param_mean, param_log_std):
        """
        Apply factorial transition to parameters.
        
        Different groups of weights evolve with different dynamics.
        
        Args:
            param_mean: Mean parameters from the current time step
            param_log_std: Log standard deviation parameters
            
        Returns:
            new_param_mean, new_param_log_std: Updated parameters
        """
        # If no groups specified, use standard transition
        if self.feature_groups is None:
            return super().transition(param_mean, param_log_std)
        
        # Otherwise, apply different transitions to each group
        new_param_mean = param_mean.clone()
        new_param_log_std = param_log_std.clone()
        
        for i, (group, std) in enumerate(zip(self.feature_groups, self.group_transition_stds)):
            # Apply transition only to the specified group
            if param_mean.dim() > 1:  # For 2D weight matrices
                noise = torch.randn_like(param_mean[:, group]) * std
                new_param_mean[:, group] = param_mean[:, group] + noise
                
                log_std_noise = torch.randn_like(param_log_std[:, group]) * (std * 0.1)
                new_param_log_std[:, group] = param_log_std[:, group] + log_std_noise
            else:  # For 1D bias vectors
                noise = torch.randn_like(param_mean[group]) * std
                new_param_mean[group] = param_mean[group] + noise
                
                log_std_noise = torch.randn_like(param_log_std[group]) * (std * 0.1)
                new_param_log_std[group] = param_log_std[group] + log_std_noise
        
        return new_param_mean, new_param_log_std


class DriftDetectingTransition(MarkovTransitionKernel):
    """
    A more advanced transition kernel that adjusts its dynamics
    based on detected concept drift.
    
    If the model detects significant changes in the data distribution,
    it can increase the transition noise to adapt more quickly.
    """
    def __init__(self, base_std=0.01, max_std=0.1, recent_losses=None):
        """
        Initialize the drift-detecting transition kernel.
        
        Args:
            base_std: Base transition noise when no drift detected
            max_std: Maximum transition noise when strong drift detected
            recent_losses: Optional buffer of recent loss values
        """
        super(DriftDetectingTransition, self).__init__(transition_std=base_std)
        self.base_std = base_std
        self.max_std = max_std
        self.recent_losses = [] if recent_losses is None else recent_losses
        self.window_size = 10  # Number of batches to consider
    
    def update_transition_std(self, new_loss):
        """
        Update transition standard deviation based on recent loss trends.
        
        A sharp increase in loss may indicate concept drift.
        
        Args:
            new_loss: The most recent training loss
            
        Returns:
            Updated transition_std value
        """
        # Add new loss to buffer
        self.recent_losses.append(new_loss)
        
        # Keep buffer at window size
        if len(self.recent_losses) > self.window_size:
            self.recent_losses.pop(0)
        
        # Need enough data to detect trends
        if len(self.recent_losses) < 3:
            return self.base_std
        
        # Check for trend of increasing loss (potential drift)
        is_increasing = self.recent_losses[-1] > self.recent_losses[-2] > self.recent_losses[-3]
        
        if is_increasing:
            # Calculate loss difference
            loss_diff = self.recent_losses[-1] - self.recent_losses[-3]
            
            # Scale transition_std based on loss difference
            drift_factor = min(1.0, loss_diff / self.recent_losses[-3])
            new_std = self.base_std + (self.max_std - self.base_std) * drift_factor
            
            self.transition_std = new_std
        else:
            # Gradually return to base_std when no drift detected
            self.transition_std = self.base_std + 0.9 * (self.transition_std - self.base_std)
        
        return self.transition_std