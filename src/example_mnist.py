import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from hmnn import HMNN
from training import train_hmnn_sequence
from concept_drift import TimeVaryingMNIST
from evaluation import evaluate_classification, plot_weight_evolution

def main():
    """
    Example of using HMNN for a time-varying MNIST dataset.
    
    This demonstrates how HMNN adapts to changes in the MNIST dataset,
    such as rotations or changes in class distribution.
    """
    parser = argparse.ArgumentParser(description='HMNN on Time-Varying MNIST')
    parser.add_argument('--n-sequences', type=int, default=5, 
                        help='Number of time sequences')
    parser.add_argument('--max-angle', type=float, default=60,
                        help='Maximum rotation angle (for rotating MNIST)')
    parser.add_argument('--mode', type=str, default='rotation',
                        choices=['rotation', 'class'],
                        help='Type of variation: rotation or class distribution')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='Batch size for testing')
    parser.add_argument('--epochs-per-seq', type=int, default=2,
                        help='Epochs per sequence')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--kl-weight', type=float, default=1e-4,
                        help='Weight for KL divergence term')
    parser.add_argument('--transition-std', type=float, default=0.01,
                        help='Standard deviation for transition kernel')
    parser.add_argument('--use-mixture', action='store_true', default=True,
                        help='Use mixture of Gaussians (DropConnect)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--save-dir', type=str, default='./results_mnist',
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
    
    # Create time-varying MNIST dataset
    print("Creating time-varying MNIST dataset...")
    mnist_dataset = TimeVaryingMNIST()
    
    if args.mode == 'rotation':
        print(f"Using rotating MNIST (0° to {args.max_angle}°)")
        data_sequence = mnist_dataset.create_rotating_sequence(
            n_sequences=args.n_sequences,
            max_angle=args.max_angle,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size
        )
    else:  # class
        print("Using MNIST with changing class distribution")
        data_sequence = mnist_dataset.create_class_shifting_sequence(
            n_sequences=args.n_sequences,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size
        )
    
    # Create HMNN model
    hmnn_model = HMNN(
        layer_sizes=[784, 128, 10],  # Input -> Hidden -> Output (10 classes)
        activation=torch.nn.functional.relu,
        prior_mean=0.0,
        prior_std=1.0,
        transition_std=args.transition_std,
        use_mixture=args.use_mixture,
        drop_prob=0.5
    ).to(device)
    
    # Create standard BNN (no transition) for comparison
    standard_model = HMNN(
        layer_sizes=[784, 128, 10],
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
        return evaluate_classification(model, data_loader, device, n_samples=5)
    
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
    plt.title(f'MNIST Performance Comparison ({args.mode} variation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{args.save_dir}/mnist_{args.mode}_accuracy.png")
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(hmnn_losses, 'b-', alpha=0.7, label='HMNN Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.title('HMNN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{args.save_dir}/mnist_{args.mode}_loss.png")
    
    # Plot weight evolution for HMNN
    plt.figure(figsize=(10, 6))
    fig = plot_weight_evolution(hmnn_model, layer_idx=0)
    fig.savefig(f"{args.save_dir}/mnist_{args.mode}_weight_evolution.png")
    
    # Calculate per-class performance if available
    if all('precision' in m for m in hmnn_metrics):
        hmnn_precision = [metrics['precision'] for metrics in hmnn_metrics]
        standard_precision = [metrics['precision'] for metrics in standard_metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, args.n_sequences + 1), hmnn_precision, 'bs-', label='HMNN')
        plt.plot(range(1, args.n_sequences + 1), standard_precision, 'rs-', label='Standard BNN')
        plt.xlabel('Time Step')
        plt.ylabel('Precision (Macro Avg)')
        plt.title(f'Precision Comparison ({args.mode} variation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{args.save_dir}/mnist_{args.mode}_precision.png")
    
    # Print summary metrics
    print("\nResults:")
    print(f"Final HMNN accuracy: {hmnn_accuracies[-1]:.4f}")
    print(f"Final Standard BNN accuracy: {standard_accuracies[-1]:.4f}")
    print(f"Results saved to {args.save_dir}/")
    
    # Calculate average difference in performance
    avg_diff = np.mean(np.array(hmnn_accuracies) - np.array(standard_accuracies))
    print(f"Average improvement of HMNN over standard BNN: {avg_diff:.4f}")

if __name__ == "__main__":
    main()