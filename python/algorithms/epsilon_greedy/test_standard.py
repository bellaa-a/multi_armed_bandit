import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Go up 3 levels
import numpy as np
from core import *

import random

random.seed(1)
probs = [0.1, 0.1, 0.1, 0.1, 0.9] 
n_arms = len(probs)
random.shuffle(probs)
arms = [BernoulliArm(p) for p in probs]

# Get best arm's probability
best_arm_index = ind_max(probs)
best_arm_prob = probs[best_arm_index]  # Use the probability directly
print(f"Best arm is {best_arm_index} (p={best_arm_prob})")

num_sim = 2

# Open both files
with open("algorithms/epsilon_greedy/standard_results.tsv", "w") as f_detail, \
     open("algorithms/epsilon_greedy/standard_regret.tsv", "w") as f_summary:

    # Write headers
    f_detail.write("e\tsim\tt\tarm\treward\tcum_reward\n")
    f_summary.write("e\tsim\treg_pct\n")

    for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = EpsilonGreedy(epsilon, [], [])
        algo.initialize(n_arms)
        results = test_algorithm(algo, arms, num_sim, 5000)
        
        # Track regret per simulation
        sim_regrets = []
        
        for sim in range(1, num_sim+1):  # For each simulation
            sim_mask = [s == sim for s in results[0]]  # Filter current sim
            optimal_reward = best_arm_prob * sum(sim_mask)
            actual_reward = sum(r for r, m in zip(results[3], sim_mask) if m)
            regret_pct = max(0, (optimal_reward - actual_reward) / optimal_reward * 100)
            sim_regrets.append(regret_pct)
            
            # Write to summary file per simulation
            f_summary.write(f"{epsilon}\t{sim}\t{regret_pct:.2f}\n")

        # Write detailed results
        for i in range(len(results[0])):
            f_detail.write(f"{epsilon}\t")
            f_detail.write("\t".join([str(results[j][i]) for j in range(len(results))]))
            f_detail.write("\n")