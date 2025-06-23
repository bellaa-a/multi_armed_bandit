import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from core import *
from algorithms.thompson.thompson import ThompsonSampling
import random
import os

random.seed(1)
probs = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(probs)
random.shuffle(probs)
arms = [BernoulliArm(p) for p in probs]

# Get best arm
best_arm_index = ind_max(probs)
best_arm_prob = probs[best_arm_index]

num_sim = 10
horizon = 500

with open("algorithms/thompson/thompson_results.tsv", "w") as f_detail, \
     open("algorithms/thompson/thompson_regret.tsv", "w") as f_summary:

    # Write headers
    f_detail.write(f"Best arm is {best_arm_index} (p={best_arm_prob})\n")
    f_detail.write("sim\tt\tarm\treward\tcum_reward\n")
    f_summary.write("sim\treg_pct\n")

    # Initialize and run algorithm
    algo = ThompsonSampling([], [])
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, num_sim, horizon)
    
    # Process results
    for sim in range(1, num_sim + 1):
        sim_mask = [s == sim for s in results[0]]
        optimal_reward = best_arm_prob * sum(sim_mask)
        actual_reward = sum(r for r, m in zip(results[3], sim_mask) if m)
        regret_pct = max(0, (optimal_reward - actual_reward) / optimal_reward * 100)
        f_summary.write(f"{sim}\t{regret_pct:.2f}\n")

    # Write detailed results
    for i in range(len(results[0])):
        f_detail.write("\t".join([str(results[j][i]) for j in range(len(results))]))
        f_detail.write("\n")