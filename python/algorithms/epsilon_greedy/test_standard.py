import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Go up 3 levels
import numpy as np
from core import *
import random

def run_standard_epsilon(seed, num_sim, horizon, epsilon):
    """Run epsilon-greedy bandit experiment for a given epsilon and return average regret percentage.
    
    Args:
        seed (int): Random seed for reproducibility
        epsilon (float): Exploration rate (0.0 to 1.0)
        num_sim (int): Number of simulations
        horizon (int): Time horizon for each simulation
    Returns:
        float: Average regret percentage across simulations
    """
    random.seed(seed)
    probs = [0.1, 0.1, 0.1, 0.1, 0.9] 
    n_arms = len(probs)
    random.shuffle(probs)
    arms = [BernoulliArm(p) for p in probs]

    # Get best arm's probability
    best_arm_index = ind_max(probs)
    best_arm_prob = probs[best_arm_index]

    # Initialize algorithm
    algo = EpsilonGreedy(epsilon, [], [])
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, num_sim, horizon)

    # Track total regret across simulations
    total_regret_pct = 0.0

    # Open files for writing (optional)
    with open("algorithms/epsilon_greedy/standard_results.tsv", "w") as f_detail, \
         open("algorithms/epsilon_greedy/standard_regret.tsv", "w") as f_summary:

        # Write headers
        f_detail.write(f"Best arm is {best_arm_index} (p={best_arm_prob})\n")
        f_detail.write("e\tsim\tt\tarm\treward\tcum_reward\n")
        f_summary.write("e\tsim\treg_pct\n")

        for sim in range(1, num_sim + 1):
            # Filter results for current simulation
            sim_mask = [s == sim for s in results[0]]
            optimal_reward = best_arm_prob * sum(sim_mask)
            actual_reward = sum(r for r, m in zip(results[3], sim_mask) if m)
            regret_pct = max(0, (optimal_reward - actual_reward) / optimal_reward * 100)
            total_regret_pct += regret_pct

            # Write to files (optional)
            f_summary.write(f"{epsilon}\t{sim}\t{regret_pct:.2f}\n")
            for i in range(len(results[0])):
                if results[0][i] == sim:
                    f_detail.write(f"{epsilon}\t")
                    f_detail.write("\t".join([str(results[j][i]) for j in range(len(results))]))
                    f_detail.write("\n")

    # Calculate average regret percentage
    avg_regret_pct = total_regret_pct / num_sim
    return avg_regret_pct