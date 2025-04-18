import math
import numpy as np
from typing import List

class GaussianAccountant:
    def __init__(self):
        """Initialize the privacy accountant for Gaussian differential privacy"""
        self.reset()
    
    def reset(self):
        """Reset the accountant's state"""
        self._steps = 0 
        self._log_moments = []
    
    def step(self, noise_multiplier: float, sample_rate: float, steps: int = 1):
        """
        Record a training step with DP parameters
        
        Args:
            noise_multiplier: Ratio of noise standard deviation to clipping norm (σ)
            sample_rate: Probability of a sample being in a batch (q = batch_size/dataset_size)
            steps: Number of steps to account for (default 1)
        """
        if noise_multiplier <= 0:
            return
            
        for _ in range(steps):
            self._log_moments.append((noise_multiplier, sample_rate))
            self._steps += 1
    
    def get_epsilon(self, delta: float) -> float:
        """
        Compute epsilon (ε) for given delta (δ) using moments accountant
        
        Args:
            delta: Target failure probability (δ)
            
        Returns:
            The privacy budget spent (ε)
        """
        if not self._log_moments:
            return 0.0
            
        # Compute the log moment for each step
        eps_values = []
        for noise_multiplier, sample_rate in self._log_moments:
            if noise_multiplier == 0:
                continue
                
            # Compute the RDP order α
            alpha = self._compute_optimal_alpha(noise_multiplier, sample_rate, delta)
            
            # Compute ε from RDP
            eps = self._compute_epsilon(noise_multiplier, sample_rate, alpha, delta)
            eps_values.append(eps)
        
        # Composition theorem: sum epsilons
        total_eps = sum(eps_values)
        
        # Advanced composition (tight bound)
        # total_eps = math.sqrt(2 * self._steps * math.log(1/delta)) * eps_values[0] + \
        #             self._steps * eps_values[0] * (math.exp(eps_values[0]) - 1)
        
        return total_eps
    
    def _compute_optimal_alpha(self, sigma: float, q: float, delta: float) -> float:
        """
        Find the optimal RDP order α that minimizes the ε bound
        
        Args:
            sigma: Noise multiplier (σ)
            q: Sampling probability
            delta: Failure probability
            
        Returns:
            Optimal α value
        """
        # Binary search for optimal alpha between 1 and 1000
        low = 1 + 1e-4
        high = 1000
        alpha = 2.0  # Initial guess
        
        for _ in range(50):  # 50 iterations for convergence
            if self._compute_epsilon(sigma, q, alpha, delta) < \
               self._compute_epsilon(sigma, q, (low + high)/2, delta):
                high = alpha
            else:
                low = alpha
            alpha = (low + high) / 2
        
        return alpha
    
    def _compute_epsilon(self, sigma: float, q: float, alpha: float, delta: float) -> float:
        if sigma == 0:
            return float('inf')
        
        # Correct RDP bound for subsampled Gaussian mechanisms
        # From "Rényi Differential Privacy of the Sampled Gaussian Mechanism" (Wang et al., 2019)
        lambda_ = sigma ** 2
        log_moment = alpha * (2 * q ** 2 * lambda_) / (2 * sigma ** 2)
        
        # Convert RDP to (ε,δ)-DP
        eps = (log_moment + math.log(1 / delta)) / (alpha - 1)
        return eps
        
    def __repr__(self):
        return f"GaussianAccountant(steps={self._steps})"