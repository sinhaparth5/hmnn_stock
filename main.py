# Fixed HMM Implementation with Visualization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os

class SimpleHMM:
    """
    A simple Hidden Markov Model implementation from scratch.
    
    This class demonstrates the basic concepts of HMMs including:
    - Forward algorithm (to compute observation probability)
    - Viterbi algorithm (to find most likely state sequence)
    - Baum-Welch algorithm (for parameter estimation)
    """
    
    def __init__(self, n_states, n_emissions):
        """
        Initialize the HMM with random parameters.
        
        Parameters:
        -----------
        n_states : int
            Number of hidden states
        n_emissions : int
            Number of possible emission values
        """
        self.n_states = n_states
        self.n_emissions = n_emissions
        
        # Initialize random model parameters
        # Initial state probabilities
        self.pi = np.random.rand(n_states)
        self.pi = self.pi / np.sum(self.pi)  # Normalize
        
        # Transition probabilities (A matrix)
        self.A = np.random.rand(n_states, n_states)
        self.A = self.A / np.sum(self.A, axis=1).reshape(-1, 1)  # Normalize rows
        
        # Emission probabilities (B matrix)
        self.B = np.random.rand(n_states, n_emissions)
        self.B = self.B / np.sum(self.B, axis=1).reshape(-1, 1)  # Normalize rows
    
    def forward(self, observations):
        """
        Forward algorithm: Compute P(O|λ) - probability of observation sequence given the model.
        
        Parameters:
        -----------
        observations : list or np.ndarray
            Sequence of observations
            
        Returns:
        --------
        float
            Probability of the observation sequence
        np.ndarray
            Forward probability matrix (alpha)
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialize first step: alpha_1(i) = π_i * b_i(o_1)
        alpha[0, :] = self.pi * self.B[:, observations[0]]
        
        # Induction step
        for t in range(1, T):
            for j in range(self.n_states):
                # alpha_t(j) = b_j(o_t) * Σ_i alpha_{t-1}(i) * a_ij
                alpha[t, j] = self.B[j, observations[t]] * np.sum(alpha[t-1, :] * self.A[:, j])
        
        # Termination: P(O|λ) = Σ_i alpha_T(i)
        return np.sum(alpha[-1, :]), alpha
    
    def viterbi(self, observations):
        """
        Viterbi algorithm: Find the most likely state sequence for the given observations.
        
        Parameters:
        -----------
        observations : list or np.ndarray
            Sequence of observations
            
        Returns:
        --------
        list
            Most likely state sequence
        float
            Probability of the most likely state sequence
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialize first step: delta_1(i) = π_i * b_i(o_1)
        delta[0, :] = self.pi * self.B[:, observations[0]]
        psi[0, :] = 0  # No predecessor in first step
        
        # Recursion step
        for t in range(1, T):
            for j in range(self.n_states):
                # delta_t(j) = max_i [delta_{t-1}(i) * a_ij * b_j(o_t)]
                delta[t, j] = np.max(delta[t-1, :] * self.A[:, j]) * self.B[j, observations[t]]
                psi[t, j] = np.argmax(delta[t-1, :] * self.A[:, j])
        
        # Termination and path backtracking
        prob = np.max(delta[-1, :])
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1, :])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states, prob
    
    def _normalize(self, matrix, axis=None):
        """Helper function to normalize probability matrices"""
        if axis is None:
            # Vector normalization
            s = np.sum(matrix)
            if s > 0:
                return matrix / s
            return np.ones_like(matrix) / len(matrix)
        else:
            # Row-wise normalization
            s = np.sum(matrix, axis=axis, keepdims=True)
            zeros = (s == 0).flatten()
            if np.any(zeros):
                matrix[zeros, :] = 1.0 / matrix.shape[1]
                s[zeros] = 1.0
            return matrix / s
    
    def baum_welch(self, observations, max_iter=100, tol=1e-4):
        """
        Baum-Welch algorithm: Estimate model parameters using observations.
        
        Parameters:
        -----------
        observations : list or np.ndarray
            Sequence of observations
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        self
        """
        observations = np.array(observations)
        T = len(observations)
        prev_log_prob = -np.inf
        
        for iteration in range(max_iter):
            # Forward algorithm
            _, alpha = self.forward(observations)
            
            # Backward algorithm
            beta = np.zeros((T, self.n_states))
            beta[-1, :] = 1.0  # Initialize with 1s
            
            # Compute beta (backward probabilities)
            for t in range(T-2, -1, -1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        beta[t, i] += self.A[i, j] * self.B[j, observations[t+1]] * beta[t+1, j]
            
            # Compute xi (probability of being in state i at t and state j at t+1)
            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                denom = 0
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self.A[i, j] * self.B[j, observations[t+1]] * beta[t+1, j]
                        denom += xi[t, i, j]
                
                if denom > 0:
                    xi[t, :, :] /= denom
            
            # Compute gamma (probability of being in state i at time t)
            gamma = np.zeros((T, self.n_states))
            for t in range(T):
                denom = np.sum(alpha[t, :] * beta[t, :])
                if denom > 0:
                    gamma[t, :] = (alpha[t, :] * beta[t, :]) / denom
                else:
                    # If denominator is zero, distribute probability uniformly
                    gamma[t, :] = 1.0 / self.n_states
            
            # Re-estimate model parameters
            # Initial state probabilities
            self.pi = gamma[0, :]
            
            # Transition probabilities
            for i in range(self.n_states):
                denom = np.sum(gamma[:-1, i])
                if denom > 0:
                    for j in range(self.n_states):
                        self.A[i, j] = np.sum(xi[:, i, j]) / denom
                else:
                    # If denominator is zero, distribute probability uniformly
                    self.A[i, :] = 1.0 / self.n_states
            
            # Normalize transition probabilities
            for i in range(self.n_states):
                self.A[i, :] = self._normalize(self.A[i, :])
            
            # Emission probabilities
            for j in range(self.n_states):
                denom = np.sum(gamma[:, j])
                if denom > 0:
                    for k in range(self.n_emissions):
                        mask = (observations == k)
                        self.B[j, k] = np.sum(gamma[mask, j]) / denom
                else:
                    # If denominator is zero, distribute probability uniformly
                    self.B[j, :] = 1.0 / self.n_emissions
            
            # Normalize emission probabilities
            for i in range(self.n_states):
                self.B[i, :] = self._normalize(self.B[i, :])
            
            # Check for convergence
            log_prob = np.log(np.sum(alpha[-1, :]) + 1e-10)  # Add small value to prevent log(0)
            if np.abs(log_prob - prev_log_prob) < tol:
                print(f"Converged after {iteration+1} iterations")
                break
            
            prev_log_prob = log_prob
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}, log probability: {log_prob:.4f}")
        
        return self
    
    def generate_sequence(self, length):
        """
        Generate a random sequence of observations based on the model parameters.
        
        Parameters:
        -----------
        length : int
            Length of the sequence to generate
            
        Returns:
        --------
        tuple
            (states, observations) - Generated state and observation sequences
        """
        states = np.zeros(length, dtype=int)
        observations = np.zeros(length, dtype=int)
        
        # Initial state
        states[0] = np.random.choice(self.n_states, p=self.pi)
        # Initial observation
        observations[0] = np.random.choice(self.n_emissions, p=self.B[states[0], :])
        
        # Generate the rest of the sequence
        for t in range(1, length):
            # State transition
            states[t] = np.random.choice(self.n_states, p=self.A[states[t-1], :])
            # Observation emission
            observations[t] = np.random.choice(self.n_emissions, p=self.B[states[t], :])
        
        return states, observations


# Example usage with dummy stock market data
def create_dummy_stock_data(length=200):
    """Create dummy stock market data with three market regimes (states)."""
    # Define parameters for three market regimes
    # State 0: Bull market (upward trend)
    # State 1: Bear market (downward trend)
    # State 2: Sideways market (neutral)
    
    # Create transition matrix
    A = np.array([
        [0.95, 0.025, 0.025],  # Bull market likely stays bull
        [0.025, 0.95, 0.025],  # Bear market likely stays bear
        [0.05, 0.05, 0.90]     # Sideways market can more easily transition
    ])
    
    # Create emission probabilities
    # 0: Large up move, 1: Small up move, 2: No change, 3: Small down move, 4: Large down move
    B = np.array([
        [0.35, 0.40, 0.15, 0.07, 0.03],  # Bull market favors up moves
        [0.03, 0.07, 0.15, 0.40, 0.35],  # Bear market favors down moves
        [0.10, 0.20, 0.40, 0.20, 0.10]   # Sideways market mostly no change
    ])
    
    # Initial state probabilities
    pi = np.array([0.33, 0.33, 0.34])  # Roughly equal probability to start in any state
    
    # Generate the sequence
    states = np.zeros(length, dtype=int)
    observations = np.zeros(length, dtype=int)
    
    # Initial state
    states[0] = np.random.choice([0, 1, 2], p=pi)
    # Initial observation
    observations[0] = np.random.choice([0, 1, 2, 3, 4], p=B[states[0], :])
    
    # Generate the rest of the sequence
    for t in range(1, length):
        # State transition
        states[t] = np.random.choice([0, 1, 2], p=A[states[t-1], :])
        # Observation emission
        observations[t] = np.random.choice([0, 1, 2, 3, 4], p=B[states[t], :])
    
    # Convert observations to price movements and generate a price series
    price = 100
    prices = [price]
    
    # Map observations to price changes
    move_map = {0: 2.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -2.0}
    
    for obs in observations:
        price = price * (1 + move_map[obs]/100)
        prices.append(price)
    
    return states, observations, np.array(prices)


def main():
    """Main function to demonstrate the HMM."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create dummy data
    true_states, observations, prices = create_dummy_stock_data(length=200)
    
    # Initialize and train HMM
    hmm = SimpleHMM(n_states=3, n_emissions=5)
    hmm.baum_welch(observations, max_iter=50)
    
    # Find most likely state sequence
    predicted_states, prob = hmm.viterbi(observations)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_states == true_states)
    print(f"State prediction accuracy: {accuracy:.2f}")
    
    # Create a directory for saving plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Create figure with adjusted size and DPI for better clarity
    fig = Figure(figsize=(15, 10), dpi=100)
    canvas = FigureCanvas(fig)
    
    # Plot 1: Stock price
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(prices[1:], color='blue', linewidth=1.5)
    ax1.set_title('Dummy Stock Price Series', pad=10)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: True states
    ax2 = fig.add_subplot(3, 1, 2)
    colors = ['green', 'blue', 'red']  # Define colors for better distinction
    for i in range(3):
        indices = np.where(true_states == i)[0]
        ax2.scatter(indices, np.ones_like(indices) * (i + 1),  # Stack states vertically
                   label=f'State {i}', 
                   s=50, alpha=0.6, 
                   c=colors[i])
    ax2.set_title('True Market Regimes (States)', pad=10)
    ax2.set_ylabel('State')
    ax2.set_ylim(0.5, 3.5)  # Adjust y-axis limits
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Predicted states
    ax3 = fig.add_subplot(3, 1, 3)
    for i in range(3):
        indices = np.where(predicted_states == i)[0]
        ax3.scatter(indices, np.ones_like(indices) * (i + 1),  # Stack states vertically
                   label=f'State {i}', 
                   s=50, alpha=0.6, 
                   c=colors[i])
    ax3.set_title('Predicted Market Regimes (States)', pad=10)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('State')
    ax3.set_ylim(0.5, 3.5)  # Adjust y-axis limits
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Adjust layout with more spacing
    fig.tight_layout(pad=3.0)
    
    # Save figure with higher quality
    fig.savefig('plots/hmm_states.png', 
                bbox_inches='tight',
                dpi=150)
    print("Plot saved to 'plots/hmm_states.png'")
    
    # Print model parameters
    print("\nEstimated Model Parameters:")
    print("Initial State Probabilities:")
    print(hmm.pi)
    print("\nTransition Probabilities:")
    print(hmm.A)
    print("\nEmission Probabilities:")
    print(hmm.B)
    
    # Hidden state analysis
    move_map = {0: 2.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -2.0}
    state_behaviors = {}
    for i in range(3):
        indices = np.where(predicted_states == i)[0]
        if len(indices) > 0:
            avg_change = np.mean([move_map[observations[j]] for j in indices])
            emissions = [observations[j] for j in indices]
            emission_counts = np.zeros(5)
            for e in emissions:
                emission_counts[e] += 1
            most_common = np.argmax(emission_counts)
            
            if avg_change > 0.5:
                state_type = "Bull Market"
            elif avg_change < -0.5:
                state_type = "Bear Market"
            else:
                state_type = "Sideways Market"
                
            state_behaviors[i] = {
                'type': state_type,
                'avg_change': avg_change,
                'most_common_emission': most_common
            }
    
    print("\nHidden State Analysis:")
    for state, behavior in state_behaviors.items():
        print(f"State {state}: {behavior['type']}, Avg Change: {behavior['avg_change']:.2f}%, " +
              f"Most Common: {behavior['most_common_emission']}")


if __name__ == "__main__":
    main()