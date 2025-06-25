# baseline (can compare with ab testing) 2:200
# small problem 5:500
# few amrs with long horizon 5:10,000 
# medium problem 10:2,000
# lots of arms with short horizon 50:1,000
from algorithms.epsilon_greedy.test_standard import *
from algorithms.epsilon_greedy.test_annealing import *
from algorithms.ucb.test_ucb1 import *
from algorithms.ucb.test_ucb2 import *
from algorithms.thompson.test_thompson import *
import pandas as pd

# Define all test configurations
configs = [
    #{'name': 'Baseline (2:200)', 'num_arms': 2, 'horizon': 200},
    #{'name': 'Small Problem (5:500)', 'num_arms': 5, 'horizon': 500},
    {'name': 'Few Arms/Long Horizon (5:10,000)', 'num_arms': 5, 'horizon': 10000},
    {'name': 'Medium Problem (10:2,000)', 'num_arms': 10, 'horizon': 2000},
    {'name': 'Many Arms/Short Horizon (50:1,000)', 'num_arms': 50, 'horizon': 1000}
]

# Algorithm parameters
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
alphas_ucb2 = [0.1, 0.3, 0.5, 0.7, 0.9]
num_sim = 10

for config in configs:
    results = []
    num_arms = config['num_arms']
    horizon = config['horizon']
    
    print(f"\n=== Running {config['name']} ===")
    
    # ε-Greedy (Standard)
    for eps in epsilons:
        regret = run_standard_epsilon(seed=1, num_sim=num_sim, 
                                    horizon=horizon, num_arms=num_arms, 
                                    epsilon=eps)
        results.append({
            'Algorithm': 'ε-Greedy (Standard)',
            'Param': f'ε={eps}',
            'Regret (%)': regret
        })
    
    # ε-Greedy (Annealing)
    regret = run_annealing_epsilon(seed=1, num_sim=num_sim, 
                                 horizon=horizon, num_arms=num_arms)
    results.append({
        'Algorithm': 'ε-Greedy (Annealing)',
        'Param': 'N/A',
        'Regret (%)': regret
    })
    
    # UCB1
    regret = run_ucb1(seed=1, num_sim=num_sim, 
                     horizon=horizon, num_arms=num_arms)
    results.append({
        'Algorithm': 'UCB1',
        'Param': 'N/A',
        'Regret (%)': regret
    })
    
    # UCB2
    for alpha in alphas_ucb2:
        regret = run_ucb2(seed=1, num_sim=num_sim, 
                         horizon=horizon, num_arms=num_arms, 
                         alpha=alpha)
        results.append({
            'Algorithm': 'UCB2',
            'Param': f'α={alpha}',
            'Regret (%)': regret
        })
    
    # Thompson Sampling
    regret = run_thompson(seed=1, num_sim=num_sim, 
                         horizon=horizon, num_arms=num_arms)
    results.append({
        'Algorithm': 'Thompson Sampling',
        'Param': 'Beta(1,1)',
        'Regret (%)': regret
    })
    
    # Create and display DataFrame for this configuration
    df = pd.DataFrame(results)
    print(f"\nResults for {config['name']} (arms={num_arms}, horizon={horizon}):")
    print(df.to_markdown(index=False))
    print("\n" + "="*80 + "\n")