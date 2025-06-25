# Convenience functions
def ind_max(x):
  m = max(x)
  return x.index(m)

def setup_bernoulli_arms(num_arms, horizon):
    # Baseline case (2 arms)
    if num_arms == 2 and horizon == 200:
        probs = [0.3, 0.7]
    
    # Small problem (5 arms)
    elif num_arms == 5 and horizon == 500:
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Few arms, long horizon (5 arms)
    elif num_arms == 5 and horizon == 10000:
        probs = [0.45, 0.48, 0.5, 0.52, 0.55]
    
    # Medium problem (10 arms)
    elif num_arms == 10 and horizon == 2000:
        probs = list(np.linspace(0.05, 0.45, 9)) + [0.9]
    
    # Many arms, short horizon (50 arms)
    elif num_arms == 50 and horizon == 1000:
        probs = list(np.linspace(0.01, 0.49, 49)) + [0.5]
        probs[-1] = 0.9  # Replace last arm with optimal
    
    else:
        raise ValueError(f"Unsupported (num_arms, horizon) combination: ({num_arms}, {horizon})")
    
    random.shuffle(probs)
    
    return [BernoulliArm(p) for p in probs], probs

# Need access to random numbers
import random
import numpy as np

# Definitions of bandit arms
from arms.adversarial import *
from arms.bernoulli import *
from arms.normal import *

# Definitions of bandit algorithms
from algorithms.epsilon_greedy.standard import *
from algorithms.epsilon_greedy.annealing import *
from algorithms.softmax.standard import *
from algorithms.softmax.annealing import *
from algorithms.ucb.ucb1 import *
from algorithms.ucb.ucb2 import *
from algorithms.exp3.exp3 import *
from algorithms.hedge.hedge import *

# # Testing framework
from testing_framework.tests import *
