# explores in phases; i.e. AAA BBBB CC 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from core import *
import random
import os

random.seed(1)
probs = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(probs)
random.shuffle(probs)
arms = [BernoulliArm(p) for p in probs] 

# Get best arm's probability
best_arm_index = ind_max(probs)
best_arm_prob = probs[best_arm_index]
print(f"Best arm is {best_arm_index} (p={best_arm_prob})")

num_sim = 10
horizon = 500

# Create output directory
os.makedirs("algorithms/ucb", exist_ok=True)

# Test different alpha values for UCB2
for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    with open(f"algorithms/ucb/ucb2_alpha{alpha}_results.tsv", "w") as f_detail, \
         open(f"algorithms/ucb/ucb2_alpha{alpha}_regret.tsv", "w") as f_summary:

        # Write headers
        f_detail.write("sim\tt\tarm\treward\tcum_reward\n")
        f_summary.write("sim\treg_pct\n")

        # Run UCB2 algorithm
        algo = UCB2(alpha, [], [])
        algo.initialize(n_arms)
        results = test_algorithm(algo, arms, num_sim, horizon)
        
        # Track regret per simulation
        for sim in range(1, num_sim + 1):
            sim_mask = [s == sim for s in results[0]]  # Filter current sim
            optimal_reward = best_arm_prob * sum(sim_mask)
            actual_reward = sum(r for r, m in zip(results[3], sim_mask) if m)
            regret_pct = max(0, (optimal_reward - actual_reward) / optimal_reward * 100)
            
            # Write regret for EVERY simulation
            f_summary.write(f"{sim}\t{regret_pct:.2f}\n")

        # Write detailed results
        for i in range(len(results[0])):
            f_detail.write("\t".join([str(results[j][i]) for j in range(len(results))]))
            f_detail.write("\n")