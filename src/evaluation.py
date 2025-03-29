import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error

def evaluate_classification(model, data_loader, device, n_samples=10):
    """
    Evaluate classification performance with uncertainty.
    
    Args:
        model: HMNN model
        data_loader: DataLoader with test data
        device: Device for computation
        n_samples: Number of Monte Carlo samples
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_uncertainties = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Make predictions with uncertainty
            if targets.dtype == torch.long:  # Multi-class classification
                # Get class probabilities
                mean_probs, std_probs = model.predict_proba(inputs, n_samples)
                preds = mean_probs.argmax(dim=1)
                
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(preds.cpu().numpy())
                all_probabilities.append(mean_probs.cpu().numpy())
                all_uncertainties.append(std_probs.cpu().numpy())
                
            else:  # Binary classification
                # Get sigmoid probabilities
                outputs_list = []
                for _ in range(n_samples):
                    outputs = model(inputs, sample=True)
                    probs = torch.sigmoid(outputs)
                    outputs_list.append(probs)
                
                probs_samples = torch.stack(outputs_list)
                mean_probs = probs_samples.mean(dim=0)
                std_probs = probs_samples.std(dim=0)
                
                # Get binary predictions
                preds = (mean_probs > 0.5).float()
                
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(preds.cpu().numpy())
                all_probabilities.append(mean_probs.cpu().numpy())
                all_uncertainties.append(std_probs.cpu().numpy())
    
    # Concatenate batch results
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    uncertainties = np.concatenate(all_uncertainties)
    
    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    
    # Calculate metrics based on classification type
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:  # Multi-class
        # Use macro averaging for multi-class
        precision = precision_score(targets, predictions, average='macro')
        recall = recall_score(targets, predictions, average='macro')
        f1 = f1_score(targets, predictions, average='macro')
        
        # For multi-class AUC, use one-vs-rest approach
        try:
            auc = roc_auc_score(targets, probabilities, multi_class='ovr')
        except:
            auc = None
    else:  # Binary
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        f1 = f1_score(targets, predictions)
        
        # For binary classification, use probability of positive class
        try:
            if probabilities.ndim > 1:
                auc = roc_auc_score(targets, probabilities[:, 1])
            else:
                auc = roc_auc_score(targets, probabilities)
        except:
            auc = None
    
    # Average uncertainty
    mean_uncertainty = np.mean(uncertainties)
    
    # Uncertainty correlation with errors
    # Higher uncertainty should correlate with prediction errors
    errors = (predictions != targets).astype(float)
    if uncertainties.ndim > 1:
        # For multi-class, use average uncertainty across classes
        avg_uncertainty = np.mean(uncertainties, axis=1)
        uncertainty_correlation = np.corrcoef(errors, avg_uncertainty)[0, 1]
    else:
        uncertainty_correlation = np.corrcoef(errors, uncertainties.flatten())[0, 1]
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mean_uncertainty': mean_uncertainty,
        'uncertainty_correlation': uncertainty_correlation
    }
    
    return metrics


def evaluate_regression(model, data_loader, device, n_samples=10):
    """
    Evaluate regression performance with uncertainty.
    
    Args:
        model: HMNN model
        data_loader: DataLoader with test data
        device: Device for computation
        n_samples: Number of Monte Carlo samples
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_uncertainties = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Make predictions with uncertainty
            mean, std = model.predict(inputs, n_samples)
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(mean.cpu().numpy())
            all_uncertainties.append(std.cpu().numpy())
    
    # Concatenate batch results
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    uncertainties = np.concatenate(all_uncertainties)
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate correlation between prediction errors and uncertainty
    errors = np.abs(predictions - targets)
    uncertainty_correlation = np.corrcoef(errors.flatten(), uncertainties.flatten())[0, 1]
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mean_uncertainty': np.mean(uncertainties),
        'uncertainty_correlation': uncertainty_correlation
    }
    
    return metrics


def plot_weight_evolution(model, layer_idx=0, weight_indices=None, figsize=(10, 6)):
    """
    Plot how weight distributions evolve over time.
    
    Args:
        model: Trained HMNN model
        layer_idx: Index of the layer to visualize
        weight_indices: Tuple of (row, col) indices for specific weight
                       If None, selects a random weight
    """
    # Extract weight means and stds from history
    times = []
    means = []
    stds = []
    
    for time_idx, params_dict in model.parameter_history:
        times.append(time_idx)
        
        weight_mu = params_dict[f'layer_{layer_idx}_weight_mu']
        weight_log_sigma = params_dict[f'layer_{layer_idx}_weight_log_sigma']
        
        # If weight_indices not specified, choose random indices
        if weight_indices is None:
            r = np.random.randint(0, weight_mu.shape[0])
            c = np.random.randint(0, weight_mu.shape[1])
            weight_indices = (r, c)
        
        r, c = weight_indices
        means.append(weight_mu[r, c].item())
        stds.append(torch.exp(weight_log_sigma[r, c]).item())
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot mean with uncertainty band
    plt.plot(times, means, 'b-', label='Weight Mean')
    plt.fill_between(times, 
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3, color='b')
    
    plt.xlabel('Time Step')
    plt.ylabel('Weight Value')
    plt.title(f'Evolution of Weight {weight_indices} in Layer {layer_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()


def plot_decision_boundary(model, X, y, device, n_samples=10, figsize=(12, 5)):
    """
    Plot decision boundary with uncertainty visualization.
    
    Args:
        model: HMNN model
        X: Features (numpy array)
        y: Labels (numpy array)
        device: Device for computation
        n_samples: Number of Monte Carlo samples
        figsize: Figure size
    """
    # Use fixed size grid to avoid dimension issues
    n_points = 100  # Use a square grid of 100x100 points
    
    # Define boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create evenly spaced grid
    x_grid = np.linspace(x_min, x_max, n_points)
    y_grid = np.linspace(y_min, y_max, n_points)
    
    # Create the mesh grid (will be square)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Get the actual total size (should be n_points*n_points but let's be safe)
    total_grid_points = xx.size
    
    # Create flattened points for prediction
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    grid_tensor = torch.FloatTensor(grid_points).to(device)
    
    # Make predictions with uncertainty
    model.eval()
    with torch.no_grad():
        if len(np.unique(y)) > 2:  # Multi-class
            mean_probs, std_probs = model.predict_proba(grid_tensor, n_samples)
            probs = mean_probs.detach().cpu().numpy()
            
            # Get class with highest probability
            pred_class = np.argmax(probs, axis=1)
            pred_class_prob = np.max(probs, axis=1)
            
            # Get average uncertainty
            uncertainty = np.mean(std_probs.detach().cpu().numpy(), axis=1)
        else:  # Binary
            # Sample multiple predictions
            outputs_list = []
            for _ in range(n_samples):
                outputs = model(grid_tensor, sample=True)
                probs = torch.sigmoid(outputs)
                outputs_list.append(probs)
            
            probs_samples = torch.stack(outputs_list)
            mean_probs = probs_samples.mean(dim=0).detach().cpu().numpy().flatten()
            std_probs = probs_samples.std(dim=0).detach().cpu().numpy().flatten()
            
            pred_class = (mean_probs > 0.5).astype(int)
            pred_class_prob = mean_probs
            uncertainty = std_probs
    
    # Debug info
    print(f"Grid shape: {xx.shape}, total points: {total_grid_points}")
    print(f"Prediction array size: {pred_class.size}")
    
    # Check for size mismatch and fix if needed
    if pred_class.size != total_grid_points:
        print(f"Size mismatch! Truncating or padding predictions to match grid...")
        # If prediction array is larger, truncate it
        if pred_class.size > total_grid_points:
            pred_class = pred_class[:total_grid_points]
            pred_class_prob = pred_class_prob[:total_grid_points]
            uncertainty = uncertainty[:total_grid_points]
        # If prediction array is smaller, pad with zeros
        else:
            pad_size = total_grid_points - pred_class.size
            pred_class = np.pad(pred_class, (0, pad_size), 'constant')
            pred_class_prob = np.pad(pred_class_prob, (0, pad_size), 'constant')
            uncertainty = np.pad(uncertainty, (0, pad_size), 'constant')
    
    # Reshape with grid dimensions (not n_points, n_points)
    pred_class = pred_class.reshape(xx.shape)
    pred_class_prob = pred_class_prob.reshape(xx.shape)
    uncertainty = uncertainty.reshape(xx.shape)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot decision regions
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, pred_class, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', alpha=0.7)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot uncertainty
    plt.subplot(1, 2, 2)
    uncertainty_map = plt.contourf(xx, yy, uncertainty, cmap='viridis', alpha=0.8)
    plt.colorbar(uncertainty_map, label='Uncertainty (std)')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', alpha=0.7)
    plt.title('Prediction Uncertainty')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    return plt.gcf()

def plot_weight_distribution(model, layer_idx=0, n_weights=5, figsize=(12, 5)):
    """
    Plot the distribution of selected weights over time.
    
    Args:
        model: Trained HMNN model
        layer_idx: Layer to visualize
        n_weights: Number of random weights to plot
        figsize: Figure size
    """
    # Get final weight parameters
    weight_mu = model.layers[layer_idx].weight_mu
    weight_log_sigma = model.layers[layer_idx].weight_log_sigma
    
    # Select random weights
    rows = torch.randint(0, weight_mu.shape[0], (n_weights,))
    cols = torch.randint(0, weight_mu.shape[1], (n_weights,))
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot each weight's evolution
    for i in range(n_weights):
        r, c = rows[i].item(), cols[i].item()
        
        # Extract values from history
        times = []
        means = []
        stds = []
        
        for time_idx, params_dict in model.parameter_history:
            times.append(time_idx)
            
            layer_mu = params_dict[f'layer_{layer_idx}_weight_mu']
            layer_log_sigma = params_dict[f'layer_{layer_idx}_weight_log_sigma']
            
            means.append(layer_mu[r, c].item())
            stds.append(torch.exp(layer_log_sigma[r, c]).item())
        
        # Plot this weight's evolution
        plt.subplot(1, n_weights, i+1)
        plt.plot(times, means, label=f'Weight ({r},{c})')
        plt.fill_between(times, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3)
        
        plt.xlabel('Time Step')
        plt.ylabel('Weight Value')
        plt.title(f'Weight ({r},{c})')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()