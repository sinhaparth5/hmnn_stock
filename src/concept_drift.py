import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision

def create_concept_drift_data(n_samples=1000, n_sequences=5, drift_factor=0.5, 
                             noise=0.1, test_size=0.2, batch_size=32):
    """
    Create synthetic dataset with concept drift.
    
    This function generates a sequence of binary classification problems
    where the decision boundary changes over time, simulating concept drift.
    
    Args:
        n_samples: Number of samples per time step
        n_sequences: Number of time steps (sequences)
        drift_factor: Controls how quickly the boundary changes
        noise: Proportion of noisy labels
        test_size: Fraction of data to use for testing
        batch_size: Batch size for the data loaders
        
    Returns:
        data_sequence: List of (train_loader, test_loader) for each time step
    """
    data_sequence = []
    
    # Initialize decision boundary parameters
    slope = 1.0
    intercept = 0.0
    
    for t in range(n_sequences):
        # Generate 2D features
        X = np.random.uniform(-3, 3, (n_samples, 2))
        
        # Determine labels based on current decision boundary
        y_boundary = slope * X[:, 0] + intercept
        y = np.zeros(n_samples)
        y[X[:, 1] > y_boundary] = 1  # Points above the line are class 1
        
        # Add noise by flipping some labels
        if noise > 0:
            noise_mask = np.random.random(n_samples) < noise
            y[noise_mask] = 1 - y[noise_mask]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42+t)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)
        
        data_sequence.append((train_loader, test_loader))
        
        # Update decision boundary for the next time step (concept drift)
        slope += drift_factor * np.random.randn()  # Random walk for slope
        intercept += drift_factor * np.random.randn()  # Random walk for intercept
    
    return data_sequence


def create_rotating_moons(n_samples=1000, n_sequences=5, noise=0.1, 
                         rotation_degrees=180, test_size=0.2, batch_size=32):
    """
    Create make_moons dataset that rotates over time.
    
    The two moons rotate gradually, creating a concept drift where
    the decision boundary needs to adapt.
    
    Args:
        n_samples: Number of samples per time step
        n_sequences: Number of time steps
        noise: Noise level for make_moons
        rotation_degrees: Total rotation over all sequences
        test_size: Fraction of data for testing
        batch_size: Batch size for data loaders
        
    Returns:
        data_sequence: List of (train_loader, test_loader) for each time step
    """
    data_sequence = []
    
    # Calculate rotation angle for each step
    rotation_per_step = rotation_degrees / n_sequences
    
    for t in range(n_sequences):
        # Generate moons dataset
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42+t)
        
        # Apply rotation for this time step
        angle_rad = np.radians(t * rotation_per_step)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        X = X @ rotation_matrix  # Apply rotation
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42+t)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)  # Use LongTensor for class indices
        
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)
        
        data_sequence.append((train_loader, test_loader))
    
    return data_sequence


class TimeVaryingMNIST:
    """
    MNIST dataset that changes over time to simulate concept drift.
    
    Strategies include:
    1. Rotation: Images rotate gradually
    2. Class focus: Distribution of classes changes
    3. Noise: Increasing noise levels
    """
    def __init__(self, base_path='./data', download=True):
        """Initialize with base dataset"""
        import torchvision
        from torchvision import transforms
        
        self.base_path = base_path
        self.transforms = transforms
        
        # Download the base dataset
        self.train_dataset = torchvision.datasets.MNIST(
            base_path, train=True, download=download)
        self.test_dataset = torchvision.datasets.MNIST(
            base_path, train=False, download=download)
    
    def create_rotating_sequence(self, n_sequences=5, max_angle=90, 
                                batch_size=128, test_batch_size=1000):
        """Create sequence where images rotate over time"""
        data_sequence = []
        angles = np.linspace(0, max_angle, n_sequences)
        
        for angle in angles:
            # Create transform with rotation
            transform = self.transforms.Compose([
                self.transforms.ToTensor(),
                self.transforms.Lambda(lambda x: self.transforms.functional.rotate(
                    x, float(angle))),
                self.transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # Apply transform
            train_dataset = torchvision.datasets.MNIST(
                self.base_path, train=True, download=False, transform=transform)
            test_dataset = torchvision.datasets.MNIST(
                self.base_path, train=False, download=False, transform=transform)
            
            # Create loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=False)
            
            data_sequence.append((train_loader, test_loader))
        
        return data_sequence
    
    def create_class_shifting_sequence(self, n_sequences=5, 
                                     batch_size=128, test_batch_size=1000):
        """
        Create sequence where class distribution changes.
        Each time step emphasizes different digits.
        """
        import torch.utils.data
        
        data_sequence = []
        transform = self.transforms.Compose([
            self.transforms.ToTensor(),
            self.transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # For each time step, create different class distributions
        for t in range(n_sequences):
            # Select which classes to emphasize in this step
            # We use a moving window of classes
            primary_classes = set([(t + j) % 10 for j in range(4)])
            
            # Create indices for train set
            train_indices = []
            for idx, (_, label) in enumerate(self.train_dataset):
                # Keep all samples from primary classes and 20% of others
                if label in primary_classes or np.random.random() < 0.2:
                    train_indices.append(idx)
            
            # Create indices for test set
            test_indices = []
            for idx, (_, label) in enumerate(self.test_dataset):
                if label in primary_classes or np.random.random() < 0.2:
                    test_indices.append(idx)
            
            # Create subset datasets
            train_subset = torch.utils.data.Subset(
                torchvision.datasets.MNIST(
                    self.base_path, train=True, transform=transform),
                train_indices
            )
            test_subset = torch.utils.data.Subset(
                torchvision.datasets.MNIST(
                    self.base_path, train=False, transform=transform),
                test_indices
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(
                test_subset, batch_size=test_batch_size, shuffle=False)
            
            data_sequence.append((train_loader, test_loader))
        
        return data_sequence