import numpy as np
from scipy.stats import beta
from core import ind_max  # Import from your core utilities

class ThompsonSampling:
    def __init__(self, counts, values):
        self.counts = counts  # Number of pulls per arm
        self.values = values  # Observed success rates
        self.alpha = [1] * len(counts)  # Beta prior (successes + 1)
        self.beta = [1] * len(counts)   # Beta prior (failures + 1)
    
    def initialize(self, n_arms):
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.alpha = [1] * n_arms  # Uniform Beta(1,1) prior
        self.beta = [1] * n_arms
    
    def select_arm(self):
        # Sample from Beta distributions
        theta_samples = [beta.rvs(a, b) for a, b in zip(self.alpha, self.beta)]
        return ind_max(theta_samples)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        
        # Update Beta parameters
        if reward == 1:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1
        
        # Update empirical mean
        n = self.counts[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * self.values[chosen_arm] + (1 / n) * reward