import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from hmnn import HMNN
from training import train_hmnn_sequence
from concept_drift import create_concept_drift_data
from evaluation import evaluate_classification, plot_decision_boundary, plot_weight_evolution

def main():
    """
    Example of using HMNN for logistic regression with concept drift.
    
    This example demonstrates the model's ability to handle a changing
    decision boundary over time, comparing HMNN with a standard BNN.
    """
    parser = argparse.ArgumentParser(description='HMNN on Logistic Regression with Concept Drift')
    parser.add_argument('--n-sequences', type=int, default=10, 
                        help='Number of time sequences')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Samples per sequence')
    parser.add_argument('--drift-factor', type=float, default=0.5,
                        help='Drift factor for boundary change')
    parser.add_argument('--epochs-per-seq', type=int, default=3,
                        help='Epochs per sequence')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--kl-weight', type=float, default=1e-3,
                        help='Weight for KL divergence term')
    parser.add_argument('--transition-std', type=float, default=0.05,
                        help='Standard deviation for transition kernel')
    parser.add_argument('--use-mixture', action='store_true', default=True,
                        help='Use mixture of Gaussians (DropConnect)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--save-dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create data with concept drift
    print("Creating synthetic dataset with concept drift...")
    data_sequence = create_concept_drift_data(
        n_samples=args.n_samples,
        n_sequences=args.n_sequences,
        drift_factor=args.drift_factor,
        noise=0.1
    )
    
    # Create HMNN model
    hmnn_model = HMNN(
        layer_sizes=[2, 10, 1],  # Input -> Hidden -> Output
        activation=torch.nn.functional.relu,
        prior_mean=0.0,
        prior_std=1.0,
        transition_std=args.transition_std,
        use_mixture=args.use_mixture,
        drop_prob=0.5
    ).to(device)
    
    # Create standard BNN (no transition) for comparison
    standard_model = HMNN(
        layer_sizes=[2, 10, 1],
        activation=torch.nn.functional.relu,
        prior_mean=0.0,
        prior_std=1.0,
        transition_std=0.0,  # No transition
        use_mixture=args.use_mixture,
        drop_prob=0.5
    ).to(device)
    
    # Create optimizers
    hmnn_optimizer = optim.Adam(hmnn_model.parameters(), lr=args.lr)
    standard_optimizer = optim.Adam(standard_model.parameters(), lr=args.lr)
    
    # Create evaluation function
    def eval_func(model, data_loader, device):
        return evaluate_classification(model, data_loader, device)
    
    # Train and evaluate HMNN
    print("\nTraining HMNN with Markov transitions...")
    hmnn_losses, hmnn_metrics = train_hmnn_sequence(
        hmnn_model, hmnn_optimizer, data_sequence, device,
        kl_weight=args.kl_weight,
        epochs_per_seq=args.epochs_per_seq,
        eval_func=eval_func,
        verbose=True
    )
    
    # Train and evaluate standard BNN
    print("\nTraining standard BNN (no transitions)...")
    standard_losses, standard_metrics = train_hmnn_sequence(
        standard_model, standard_optimizer, data_sequence, device,
        kl_weight=args.kl_weight,
        epochs_per_seq=args.epochs_per_seq,
        eval_func=eval_func,
        verbose=True
    )
    
    # Extract accuracies for comparison
    hmnn_accuracies = [metrics['accuracy'] for metrics in hmnn_metrics]
    standard_accuracies = [metrics['accuracy'] for metrics in standard_metrics]
    
    # Plot comparison of accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.n_sequences + 1), hmnn_accuracies, 'bo-', label='HMNN')
    plt.plot(range(1, args.n_sequences + 1), standard_accuracies, 'ro-', label='Standard BNN')
    plt.xlabel('Time Step')
    plt.ylabel('Accuracy')
    plt.title('Performance Comparison: HMNN vs Standard BNN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{args.save_dir}/accuracy_comparison.png")
    
    # Plot decision boundaries
    # Get a test set for visualization
    _, test_loader = data_sequence[-1]
    X_test = []
    y_test = []
    for inputs, targets in test_loader:
        X_test.append(inputs.numpy())
        y_test.append(targets.numpy())
    
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    
    # Plot HMNN decision boundary
    plt.figure(figsize=(12, 5))
    fig = plot_decision_boundary(hmnn_model, X_test, y_test, device)
    fig.savefig(f"{args.save_dir}/hmnn_decision_boundary.png")
    
    # Plot standard BNN decision boundary
    plt.figure(figsize=(12, 5))
    fig = plot_decision_boundary(standard_model, X_test, y_test, device)
    fig.savefig(f"{args.save_dir}/standard_decision_boundary.png")
    
    # Plot weight evolution for HMNN
    plt.figure(figsize=(10, 6))
    fig = plot_weight_evolution(hmnn_model, layer_idx=0)
    fig.savefig(f"{args.save_dir}/weight_evolution.png")
    
    print("\nResults:")
    print(f"Final HMNN accuracy: {hmnn_accuracies[-1]:.4f}")
    print(f"Final Standard BNN accuracy: {standard_accuracies[-1]:.4f}")
    print(f"Results saved to {args.save_dir}/")
    
    # Calculate average difference in performance
    avg_diff = np.mean(np.array(hmnn_accuracies) - np.array(standard_accuracies))
    print(f"Average improvement of HMNN over standard BNN: {avg_diff:.4f}")

if __name__ == "__main__":
    main()