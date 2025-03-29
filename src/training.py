import torch
import torch.nn.functional as F
import numpy as np

def train_hmnn_epoch(model, optimizer, dataloader, device, kl_weight=1.0):
    """
    Train the HMNN for one epoch using variational inference.
    
    Args:
        model: HMNN model
        optimizer: PyTorch optimizer
        data_loader: DataLoader with training data
        device: Device to use for computation
        kl_weight: Weight for the KL divergence term in ELBO
        
    Returns:
        mean_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Ensure inputs are properly flattened for MNIST
        if len(inputs.shape) > 2:
            # For image data like MNIST (batch_size, channels, height, width)
            batch_size = inputs.size(0)
            inputs = inputs.view(batch_size, -1)  # Flatten to (batch_size, features)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with sampled weights
        outputs = model(inputs, sample=True)
        
        # Calculate negative log likelihood (NLL) based on task type
        if outputs.shape[-1] > 1:  # Multi-class classification
            nll = F.cross_entropy(outputs, targets)
        elif targets.dim() == 1 and outputs.dim() > 1:  # Binary classification with logits
            outputs = outputs.view(-1)
            nll = F.binary_cross_entropy_with_logits(outputs, targets.float())
        else:  # Regression
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            nll = F.mse_loss(outputs, targets)
        
        # Calculate KL divergence
        kl = model.kl_divergence()
        
        # ELBO = -NLL - KL
        # We minimize -ELBO = NLL + KL
        loss = nll + kl_weight * kl
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    mean_loss = total_loss / num_batches
    return mean_loss


def sequential_learning_step(model, optimizer, data_loader, device, 
                             kl_weight=1.0, n_epochs=1, apply_transition=True):
    """
    Perform a complete sequential learning step:
    1. Transition (weight evolution)
    2. Correction (training on new data)
    
    This implements one step of the Bayesian filtering process in the HMNN.
    
    Args:
        model: HMNN model
        optimizer: PyTorch optimizer
        data_loader: DataLoader with current time step's data
        device: Device to use for computation
        kl_weight: Weight for the KL divergence term
        n_epochs: Number of training epochs for this time step
        apply_transition: Whether to apply the transition first
        
    Returns:
        losses: List of losses for each epoch
    """
    # Step 1: Transition - Apply the Markov transition to evolve weights
    if apply_transition:
        model.transition_to_next_time()
    
    # Step 2: Correction - Update the posterior with new data
    losses = []
    for epoch in range(n_epochs):
        loss = train_hmnn_epoch(model, optimizer, data_loader, device, kl_weight)
        losses.append(loss)
    
    return losses


def train_hmnn_sequence(model, optimizer, data_sequence, device, 
                       kl_weight=1.0, epochs_per_seq=1, 
                       eval_func=None, verbose=True):
    """
    Train an HMNN on a sequence of datasets.
    
    Args:
        model: HMNN model
        optimizer: PyTorch optimizer
        data_sequence: List of (train_loader, test_loader) tuples
        device: Device to use for computation
        kl_weight: Weight for the KL divergence term
        epochs_per_seq: Number of epochs per sequence
        eval_func: Optional evaluation function to call after each time step
        verbose: Whether to print progress
        
    Returns:
        losses: List of training losses
        metrics: List of evaluation metrics (if eval_func provided)
    """
    all_losses = []
    all_metrics = []
    
    for t, (train_loader, test_loader) in enumerate(data_sequence):
        if verbose:
            print(f"\nTime step {t+1}/{len(data_sequence)}")
        
        # Apply sequential learning step
        losses = sequential_learning_step(
            model, optimizer, train_loader, device,
            kl_weight=kl_weight,
            n_epochs=epochs_per_seq,
            apply_transition=(t > 0)  # No transition for first step
        )
        
        all_losses.extend(losses)
        
        # Evaluate if function provided
        if eval_func is not None:
            metrics = eval_func(model, test_loader, device)
            all_metrics.append(metrics)
            
            if verbose:
                # Print key metrics - FIX: Handle None values properly
                for key, value in metrics.items():
                    if value is not None:
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: None")
    
    return all_losses, all_metrics


def expected_calibration_error(model, data_loader, device, n_bins=10, n_samples=10):
    """
    Calculate Expected Calibration Error to evaluate uncertainty calibration.
    
    A well-calibrated model should be confident when it's correct and
    express uncertainty when it's likely to be wrong.
    
    Args:
        model: HMNN model
        data_loader: DataLoader with evaluation data
        device: Device for computation
        n_bins: Number of confidence bins
        n_samples: Number of Monte Carlo samples
        
    Returns:
        ece: Expected Calibration Error
        bin_accuracies: Accuracy for each bin
        bin_confidences: Average confidence for each bin
        bin_counts: Number of predictions in each bin
    """
    model.eval()
    confidences = []
    predictions = []
    accuracies = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get probabilistic predictions
            if inputs[0].dim() > 1:  # Multi-class or multi-dim
                mean_probs, _ = model.predict_proba(inputs, n_samples)
                confidence, pred_class = torch.max(mean_probs, dim=1)
                accuracy = (pred_class == targets).float()
            else:  # Binary
                mean_probs, _ = model.predict_proba(inputs, n_samples)
                confidence = mean_probs.view(-1)
                pred_class = (confidence > 0.5).float()
                accuracy = (pred_class == targets).float()
            
            # Store results
            confidences.append(confidence.cpu().numpy())
            accuracies.append(accuracy.cpu().numpy())
    
    # Concatenate results
    confidences = np.concatenate(confidences)
    accuracies = np.concatenate(accuracies)
    
    # Create bins and calculate ECE
    bin_indices = np.digitize(confidences, np.linspace(0, 1, n_bins+1))
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_mask = (bin_indices == i+1)
        if np.any(bin_mask):
            bin_accuracies[i] = np.mean(accuracies[bin_mask])
            bin_confidences[i] = np.mean(confidences[bin_mask])
            bin_counts[i] = np.sum(bin_mask)
    
    # Calculate ECE
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / np.sum(bin_counts)
    
    return ece, bin_accuracies, bin_confidences, bin_counts