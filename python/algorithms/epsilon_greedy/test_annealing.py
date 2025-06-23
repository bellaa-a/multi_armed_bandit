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

# Get best arm
best_arm_index = ind_max(probs)
best_arm_prob = probs[best_arm_index]
print(f"Best arm is {best_arm_index} (p={best_arm_prob})")

num_sim = 10
horizon = 500

with open("algorithms/epsilon_greedy/annealing_regret.tsv", "w") as f_summary, \
     open("algorithms/epsilon_greedy/annealing_results.tsv", "w") as f_detail:

    # Write headers (no epsilon column)
    f_summary.write("sim\treg_pct\n")
    f_detail.write("sim\tt\tarm\treward\tcum_reward\n")

    # Run annealing algorithm
    algo = AnnealingEpsilonGreedy([], [])
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, num_sim, horizon)
    
    sim_regrets = []
    
    for sim in range(1, num_sim + 1):
        sim_mask = [s == sim for s in results[0]]
        optimal_reward = best_arm_prob * sum(sim_mask)
        actual_reward = sum(r for r, m in zip(results[3], sim_mask) if m)
        regret_pct = max(0, (optimal_reward - actual_reward) / optimal_reward * 100)
        sim_regrets.append(regret_pct)
        
        # Write regret summary 
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