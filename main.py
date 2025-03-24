import numpy as np
from typing import Tuple, List

class HMMNFilter:
    def __init__(self, num_states: int, observation_dim: int):
        """
            Initialize HMM with neural network emission model

            Args:
                num_states: Number of hidden states
                observation_dim: Dimenstion of observation vector
        """
        # Transition Martix (A)
        self.A = np.random.random((num_states, num_states))
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Initial state distribution (pi)
        self.pi = np.random.random(num_states)
        self.pi /= self.pi.sum()

        # Neural network parameters for emission probabilites
        self.num_states = num_states
        self.observation_dim = observation_dim

        # Simple feedforward NN weights
        self.W1 = np.random.randn(observation_dim, 64) * 0.01
        self.W2 = np.random.randn(64, num_states) * 0.01
        self.b1 = np.zeros(64)
        self.b2 = np.zeros(num_states)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ ReLU activition function """
        return np.maximum(0, x)

    def emission_prob(self, observation: np.ndarray) -> np.ndarray:
        """
        Calculate emission probabilities using neural network

        Args:
            observation: Input observation vector

        Returns:
            Probability distribution over states
        """
        # Forward pass through neural network
        hidden = self.relu(np.dot(observation, self.W1) + self.b1)
        output = np.dot(hidden, self.W2) + self.b2

        # Softmax to get probabilites
        exp_output = np.exp(output - np.max(output))
        return exp_output / exp_output.sum()

    def forward_step(self, alpha_prev: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """
        Single forward step in filtering

        Args:
            alpha_prev: Previous filtering distribution
            observation: Current observation

        Returns:
            New filtering distribution
        """
        # Get emission probabilites for neural network
        b = self.emission_prob(observation)

        # Forward algorithm step
        alpha = b * np.dot(self.A.T, alpha_prev)
        return alpha / alpha.sum()

    def filter(self, observations: np.ndarray) -> List[np.ndarray]:
        """
        Run filtering algorithm over sequenc of observations

        Args:
            observations: Array of observations (T x observation_dim)

        Returns:
            List of filtering distribution
        """
        T = len(observations)
        alphas = []

        #Initialize with first observation
        alpha = self.pi * self.emission_prob(observations[0])
        alpha /= alpha.sum()
        alphas.append(alpha)

        # Forward pass for remaining observations
        for t in range(1, T):
            alpha = self.forward_step(alphas[-1], observations[t])
            alphas.append(alpha)

        return alphas

def main():
    # Create synthetic data
    np.random.seed(42)
    num_states = 3
    obs_dim = 5
    sequence_length = 10

    # Initialize model
    hmm = HMMNFilter(num_states, obs_dim)

    # Generate random observations
    observations = np.random.randn(sequence_length, obs_dim)

    # Run filtering
    filtering_distributions = hmm.filter(observations)

    # Print results
    for t, alpha in enumerate(filtering_distributions):
        print(f"Time {t}: {alpha.round(3)}")

if __name__ == "__main__":
    main()


