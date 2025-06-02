# This file contains adaptations from IBM Qiskit CVaR Optimization Tutorial
# https://qiskit-community.github.io/qiskit-optimization/tutorials/08_cvar_optimization.html
# Original copyright (c) IBM 2017, 2024.
# Modified by Colin Farley (2025) for research use at HARP Research.
import numpy as np


def load_default_portfolio():
    '''
        Instance Described in Barkoutsos et al., 2020
    '''
    mu = np.array([0.7313, 0.9893, 0.2725, 0.8750, 0.7667, 0.3622])
    sigma = np.array(
        [
            [0.7312, -0.6233, 0.4689, -0.5452, -0.0082, -0.3809],
            [-0.6233, 2.4732, -0.7538, 2.4659, -0.0733, 0.8945],
            [0.4689, -0.7538, 1.1543, -1.4095, 0.0007, -0.4301],
            [-0.5452, 2.4659, -1.4095, 3.5067, 0.2012, 1.0922],
            [-0.0082, -0.0733, 0.0007, 0.2012, 0.6231, 0.1509],
            [-0.3809, 0.8945, -0.4301, 1.0922, 0.1509, 0.8992],
        ]
    )
    return mu, sigma

def load_random_portfolio(n=6, seed=123):
    """
    Generate a random portfolio with symmetric, positive semi-definite risk matrix.
    
    Args:
        n: Number of assets
        seed: Random seed for reproducibility

    Returns:
        mu: Expected returns vector
        sigma: Risk covariance matrix
    """
    np.random.seed(seed)
    
    # Random expected returns between 0 and 1
    mu = np.random.rand(n)
    
    # Generate symmetric positive semi-definite covariance matrix
    A = np.random.randn(n, n)
    sigma = np.dot(A, A.T)  # ensure the matrix is symmetric and PSD
    
    return mu, sigma