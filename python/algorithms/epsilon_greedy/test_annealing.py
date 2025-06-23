import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from core import *
import random
import os

def run_annealing_epsilon(seed=1, num_sim=10, horizon=500):
    """
    Run annealing epsilon-greedy experiment with file outputs and return average regret.
    
    Args:
        seed (int): Random seed
        num_sim (int): Number of simulations
        horizon (int): Time horizon
        
    Returns:
        float: Average regret percentage across simulations
    """
    random.seed(seed)
    probs = [0.1, 0.1, 0.1, 0.1, 0.9]
    n_arms = len(probs)
    random.shuffle(probs)
    arms = [BernoulliArm(p) for p in probs]

    # Get best arm
    best_arm_index = ind_max(probs)
    best_arm_prob = probs[best_arm_index]

    # Ensure output directory exists
    os.makedirs("algorithms/epsilon_greedy", exist_ok=True)

    total_regret = 0.0
    
    with open("algorithms/epsilon_greedy/annealing_regret.tsv", "w") as f_summary, \
         open("algorithms/epsilon_greedy/annealing_results.tsv", "w") as f_detail:

        # Write headers
        f_detail.write(f"Best arm is {best_arm_index} (p={best_arm_prob})\n")
        f_detail.write("sim\tt\tarm\treward\tcum_reward\n")
        f_summary.write("sim\treg_pct\n")

        # Run algorithm
        algo = AnnealingEpsilonGreedy([], [])
        algo.initialize(n_arms)
        results = test_algorithm(algo, arms, num_sim, horizon)
        
        # Process results
        for sim in range(1, num_sim + 1):
            sim_mask = [s == sim for s in results[0]]
            optimal = best_arm_prob * sum(sim_mask)
            actual = sum(r for r, m in zip(results[3], sim_mask) if m)
            regret_pct = max(0, (optimal - actual) / optimal * 100)
            total_regret += regret_pct
            
            # Write to summary file
            f_summary.write(f"{sim}\t{regret_pct:.2f}\n")

        # Write detailed results
        for i in range(len(results[0])):
            f_detail.write("\t".join([
                str(results[0][i]),  # sim_num
                str(results[1][i]),  # time
                str(results[2][i]),  # chosen_arm
                str(results[3][i]),  # reward
                str(results[4][i])   # cum_reward
            ]) + "\n")

    return total_regret / num_sim  # Average regret percentage
