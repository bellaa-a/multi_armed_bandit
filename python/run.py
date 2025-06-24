# for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]
# for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]
from algorithms.epsilon_greedy.test_standard import *
from algorithms.epsilon_greedy.test_annealing import *
from algorithms.ucb.test_ucb1 import *
from algorithms.ucb.test_ucb2 import *
from algorithms.thompson.test_thompson import *

#print(run_standard_epsilon(1, 10, 500, 0.1))
#print(run_annealing_epsilon(1, 10, 500))
#print(run_ucb1(1, 10, 500))
#print(run_ucb2(1, 10, 500, 0.5))
#print(run_thompson(1, 10, 500))

import pandas as pd
import matplotlib.pyplot as plt

# Test configurations
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
alphas_ucb2 = [0.1, 0.3, 0.5, 0.7, 0.9]
num_sim = 1000
horizon = 500

# Store results
results = []

# Test ε-Greedy (Standard)
for eps in epsilons:
    regret = run_standard_epsilon(seed=1, num_sim=num_sim, horizon=horizon, epsilon=eps)
    results.append({"Algorithm": "ε-Greedy (Standard)", "Param": f"ε={eps}", "Regret (%)": regret})

# Test ε-Greedy (Annealing)
regret = run_annealing_epsilon(seed=1, num_sim=num_sim, horizon=horizon)
results.append({"Algorithm": "ε-Greedy (Annealing)", "Param": "N/A", "Regret (%)": regret})

# Test UCB1
regret = run_ucb1(seed=1, num_sim=num_sim, horizon=horizon)
results.append({"Algorithm": "UCB1", "Param": "N/A", "Regret (%)": regret})

# Test UCB2
for alpha in alphas_ucb2:
    regret = run_ucb2(seed=1, num_sim=num_sim, horizon=horizon, alpha=alpha)
    results.append({"Algorithm": "UCB2", "Param": f"α={alpha}", "Regret (%)": regret})

# Test Thompson Sampling
regret = run_thompson(seed=1, num_sim=num_sim, horizon=horizon)
results.append({"Algorithm": "Thompson Sampling", "Param": "Beta(1,1)", "Regret (%)": regret})

# Create DataFrame
df = pd.DataFrame(results)
print(df.to_markdown(index=False))