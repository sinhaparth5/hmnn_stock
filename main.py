# Enhanced HMM Implementation with Visualization using Indian Stock Market Data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

class EnhancedHMM:
    """
    An enhanced Hidden Markov Model implementation for stock market regime detection.
    """
    
    def __init__(self, n_states, n_emissions):
        """
        Initialize the HMM with better-informed parameters.
        
        Parameters:
        -----------
        n_states : int
            Number of hidden states (e.g., Bull, Sideways, Bear)
        n_emissions : int
            Number of possible emission values (e.g., price change categories)
        """
        self.n_states = n_states
        self.n_emissions = n_emissions
        
        # Initial state probabilities (roughly equal)
        self.pi = np.ones(n_states) / n_states
        
        # Transition probabilities (A matrix): Favor staying in the same state
        self.A = np.full((n_states, n_states), 0.15 / (n_states - 1))
        np.fill_diagonal(self.A, 0.7)  # Reduced persistence to allow more transitions
        
        # Emission probabilities (B matrix): More extreme initial values
        # Emissions: 0 (large down), 1 (small down), 2 (neutral), 3 (small up), 4 (large up)
        self.B = np.ones((n_states, n_emissions)) / n_emissions
        # Bull: Strongly favor up moves
        self.B[0, :] = [0.02, 0.03, 0.10, 0.35, 0.50]
        # Sideways: Strongly favor neutral moves
        self.B[1, :] = [0.05, 0.10, 0.70, 0.10, 0.05]
        # Bear: Strongly favor down moves
        self.B[2, :] = [0.50, 0.35, 0.10, 0.03, 0.02]
        
        # Normalize
        self.A = self.A / np.sum(self.A, axis=1).reshape(-1, 1)
        self.B = self.B / np.sum(self.B, axis=1).reshape(-1, 1)
    
    def forward(self, observations):
        """
        Forward algorithm: Compute P(O|λ) - probability of observation sequence given the model.
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        alpha[0, :] = self.pi * self.B[:, observations[0]]
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = self.B[j, observations[t]] * np.sum(alpha[t-1, :] * self.A[:, j])
        
        return np.sum(alpha[-1, :]), alpha
    
    def viterbi(self, observations):
        """
        Viterbi algorithm: Find the most likely state sequence for the given observations.
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0, :] = self.pi * self.B[:, observations[0]]
        psi[0, :] = 0
        
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t-1, :] * self.A[:, j]) * self.B[j, observations[t]]
                psi[t, j] = np.argmax(delta[t-1, :] * self.A[:, j])
        
        prob = np.max(delta[-1, :])
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1, :])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states, prob
    
    def _normalize(self, matrix, axis=None):
        """Helper function to normalize probability matrices"""
        if axis is None:
            s = np.sum(matrix)
            if s > 0:
                return matrix / s
            return np.ones_like(matrix) / len(matrix)
        else:
            s = np.sum(matrix, axis=axis, keepdims=True)
            zeros = (s == 0).flatten()
            if np.any(zeros):
                matrix[zeros, :] = 1.0 / matrix.shape[1]
                s[zeros] = 1.0
            return matrix / s
    
    def baum_welch(self, observations, max_iter=500, tol=1e-10, min_log_prob_change=1e-5):
        """
        Baum-Welch algorithm: Estimate model parameters using observations.
        """
        observations = np.array(observations)
        T = len(observations)
        prev_log_prob = -np.inf
        
        for iteration in range(max_iter):
            # Forward algorithm
            _, alpha = self.forward(observations)
            
            # Backward algorithm
            beta = np.zeros((T, self.n_states))
            beta[-1, :] = 1.0
            
            for t in range(T-2, -1, -1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        beta[t, i] += self.A[i, j] * self.B[j, observations[t+1]] * beta[t+1, j]
            
            # Compute xi
            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                denom = 0
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self.A[i, j] * self.B[j, observations[t+1]] * beta[t+1, j]
                        denom += xi[t, i, j]
                if denom > 0:
                    xi[t, :, :] /= denom
            
            # Compute gamma
            gamma = np.zeros((T, self.n_states))
            for t in range(T):
                denom = np.sum(alpha[t, :] * beta[t, :])
                if denom > 0:
                    gamma[t, :] = (alpha[t, :] * beta[t, :]) / denom
                else:
                    gamma[t, :] = 1.0 / self.n_states
            
            # Re-estimate model parameters
            self.pi = gamma[0, :]
            
            # Transition probabilities with minimum staying probability
            min_stay_prob = 0.7
            for i in range(self.n_states):
                denom = np.sum(gamma[:-1, i])
                if denom > 0:
                    for j in range(self.n_states):
                        self.A[i, j] = np.sum(xi[:, i, j]) / denom
                else:
                    self.A[i, :] = 1.0 / self.n_states
            
            # Enforce minimum staying probability
            for i in range(self.n_states):
                if self.A[i, i] < min_stay_prob:
                    excess = (min_stay_prob - self.A[i, i]) * (self.n_states - 1)
                    other_probs = self.A[i, [j for j in range(self.n_states) if j != i]]
                    total_other = np.sum(other_probs)
                    if total_other > 0:
                        self.A[i, [j for j in range(self.n_states) if j != i]] = other_probs * (1 - min_stay_prob) / total_other
                    else:
                        self.A[i, [j for j in range(self.n_states) if j != i]] = (1 - min_stay_prob) / (self.n_states - 1)
                    self.A[i, i] = min_stay_prob
            
            # Normalize transition probabilities
            for i in range(self.n_states):
                self.A[i, :] = self._normalize(self.A[i, :])
            
            # Emission probabilities with increased smoothing
            smoothing = 1e-2  # Increased smoothing to ensure differentiation
            for j in range(self.n_states):
                denom = np.sum(gamma[:, j])
                if denom > 0:
                    for k in range(self.n_emissions):
                        mask = (observations == k)
                        self.B[j, k] = (np.sum(gamma[mask, j]) + smoothing) / (denom + smoothing * self.n_emissions)
                else:
                    self.B[j, :] = 1.0 / self.n_emissions
            
            # Normalize emission probabilities
            for i in range(self.n_states):
                self.B[i, :] = self._normalize(self.B[i, :])
            
            # Check for convergence
            log_prob = np.log(np.sum(alpha[-1, :]) + 1e-10)
            log_prob_change = np.abs(log_prob - prev_log_prob)
            if log_prob_change < tol or log_prob_change < min_log_prob_change:
                print(f"Converged after {iteration+1} iterations")
                break
            
            prev_log_prob = log_prob
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}, log probability: {log_prob:.4f}")
        
        return self

# Fetch Indian stock market data (NIFTY 50) using yfinance
def fetch_stock_data(ticker="^NSEI", start_date=None, end_date=None, volatility_window=20):
    """
    Fetch historical stock data using yfinance for the NIFTY 50 index.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., '^NSEI' for NIFTY 50)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    volatility_window : int
        Window size for calculating rolling volatility
    
    Returns:
    --------
    tuple
        (dates, prices, returns, volatilities, observations) - Dates, price series, percentage returns, volatilities, and discretized observations
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    
    # Fetch data
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    # Extract dates and prices
    dates = df.index
    prices = df['Close'].values
    returns = np.diff(prices) / prices[:-1] * 100  # Percentage change
    
    # Calculate rolling volatility (standard deviation of returns)
    returns_series = pd.Series(returns)
    volatilities = returns_series.rolling(window=volatility_window).std().fillna(0).values
    
    # Discretize returns into emissions
    # Adjusted thresholds: 0: large down (<-3%), 1: small down (-3% to -1%), 2: neutral (-1% to 1%),
    # 3: small up (1% to 3%), 4: large up (>3%)
    observations = np.zeros(len(returns), dtype=int)
    for i, r in enumerate(returns):
        if r < -3.0:
            observations[i] = 0  # Large down
        elif -3.0 <= r < -1.0:
            observations[i] = 1  # Small down
        elif -1.0 <= r <= 1.0:
            observations[i] = 2  # Neutral
        elif 1.0 < r <= 3.0:
            observations[i] = 3  # Small up
        else:
            observations[i] = 4  # Large up
    
    return dates, prices, returns, volatilities, observations

def main():
    """Main function to demonstrate the HMM with Indian stock market data."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Fetch NIFTY 50 data
    dates, prices, returns, volatilities, observations = fetch_stock_data(
        ticker="LT.NS",
        start_date="2022-01-01",
        end_date="2025-01-01"
    )
    
    # Initialize and train HMM
    hmm = EnhancedHMM(n_states=3, n_emissions=5)
    hmm.baum_welch(observations, max_iter=500, tol=1e-10, min_log_prob_change=1e-5)
    
    # Find most likely state sequence
    predicted_states, prob = hmm.viterbi(observations)
    
    # Hidden state analysis to determine state labels
    state_behaviors = {}
    for i in range(3):
        indices = np.where(predicted_states == i)[0]
        if len(indices) > 0:
            avg_change = np.mean([returns[j] for j in indices])
            avg_volatility = np.mean([volatilities[j] for j in indices])
            emissions = [observations[j] for j in indices]
            emission_counts = np.zeros(5)
            for e in emissions:
                emission_counts[e] += 1
            most_common = np.argmax(emission_counts)
            
            # Classify state based on average change and volatility
            if avg_change > 0.5 and avg_volatility < np.median(volatilities):
                state_type = "Bull Market"
            elif avg_change < -0.5 or avg_volatility > np.percentile(volatilities, 75):
                state_type = "Bear Market"
            else:
                state_type = "Sideways Market"
                
            state_behaviors[i] = {
                'type': state_type,
                'avg_change': avg_change,
                'avg_volatility': avg_volatility,
                'most_common_emission': most_common
            }
    
    # Map states to labels based on analysis
    state_labels = {i: behavior['type'].split()[0] for i, behavior in state_behaviors.items()}
    state_colors = {'Bull': 'green', 'Sideways': 'blue', 'Bear': 'red'}
    
    # Create a directory for saving plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Create figure with adjusted size and DPI
    fig = Figure(figsize=(15, 8), dpi=100)
    canvas = FigureCanvas(fig)
    
    # Plot 1: Stock price with shaded regions for predicted states
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(prices, color='blue', linewidth=1.5, label='L&T Price')
    
    # Add shaded regions for predicted states
    state_changes = np.where(np.diff(predicted_states) != 0)[0] + 1
    state_changes = np.concatenate(([0], state_changes, [len(predicted_states)]))
    
    for i in range(len(state_changes) - 1):
        start, end = state_changes[i], state_changes[i+1]
        state = predicted_states[start]
        if state in state_labels:
            label = state_labels[state]
            ax1.axvspan(start, end, alpha=0.2, color=state_colors[label],
                        label=label if i == 0 or predicted_states[state_changes[i-1]] != state else None)
    
    ax1.set_title('L&T Price with Predicted Market Regimes (2022-2024)', pad=10)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Add date labels to x-axis
    date_indices = np.linspace(0, len(dates)-1, 5, dtype=int)
    date_labels = [dates[i].strftime('%Y-%m') for i in date_indices]
    ax1.set_xticks(date_indices)
    ax1.set_xticklabels(date_labels)
    
    # Plot 2: Predicted states
    ax2 = fig.add_subplot(2, 1, 2)
    for i in range(3):
        indices = np.where(predicted_states == i)[0]
        if i in state_labels:
            ax2.scatter(indices, np.ones_like(indices) * (i + 1),
                       label=state_labels[i],
                       s=50, alpha=0.6,
                       c=state_colors[state_labels[i]])
    ax2.set_title('Predicted Market Regimes (States)', pad=10)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('State')
    ax2.set_ylim(0.5, 3.5)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Add date labels to x-axis
    ax2.set_xticks(date_indices)
    ax2.set_xticklabels(date_labels)
    
    # Adjust layout
    fig.tight_layout(pad=3.0)
    
    # Save figure
    fig.savefig('plots/hmm_states_lt.png',
                bbox_inches='tight',
                dpi=150)
    print("Plot saved to 'plots/hmm_states_lt.png'")
    
    # Print model parameters
    print("\nEstimated Model Parameters:")
    print("Initial State Probabilities:")
    print(hmm.pi)
    print("\nTransition Probabilities:")
    print(hmm.A)
    print("\nEmission Probabilities:")
    print(hmm.B)
    
    # Print hidden state analysis
    print("\nHidden State Analysis:")
    for state, behavior in state_behaviors.items():
        print(f"State {state}: {behavior['type']}, Avg Change: {behavior['avg_change']:.2f}%, " +
              f"Avg Volatility: {behavior['avg_volatility']:.2f}, Most Common Emission: {behavior['most_common_emission']}")

if __name__ == "__main__":
    main()