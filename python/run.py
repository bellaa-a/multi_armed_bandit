# for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]
# for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]
from algorithms.epsilon_greedy.test_standard import *
from algorithms.epsilon_greedy.test_annealing import *
from algorithms.ucb.test_ucb1 import *
from algorithms.ucb.test_ucb2 import *

avg_reg_pct = run_standard_epsilon(1, 10, 500, 0.1)
#print(avg_reg_pct)

#print(run_annealing_epsilon(1, 10, 500))

#print(run_ucb1(1, 10, 500))

print(run_ucb2(1, 10, 500, 0.5))