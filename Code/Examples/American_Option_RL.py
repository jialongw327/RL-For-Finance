import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Processes\\Markov')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Examples')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Algorithms')

from MDP_Refined import MDPRefined
from American_Option_MDP import AmericanOptionMDP
from TD0_Control import TD0_Control

import numpy as np


T = 30
s0 = 100
u = 0.05
d = -0.03
p = 0.399
K = 105
    

test = AmericanOptionMDP(T,p,u,d,K,s0)
state, tr = test.build_graph()

gamma_val = 0.9
mdp_ref_obj1 = MDPRefined(tr, gamma_val)
mdp_rep_obj = mdp_ref_obj1.get_mdp_rep_for_rl_tabular()
epsilon_val = 0.1
epsilon_half_life_val = 100
learning_rate_val = 0.1
learning_rate_decay_val = 1e6
episodes_limit = 5000
max_steps_val = 1000

sarsa = TD0_Control(
        mdp_rep_obj,
        epsilon_val,
        epsilon_half_life_val,
        learning_rate_val,
        learning_rate_decay_val,
        episodes_limit,
        max_steps_val,
        'Sarsa'
    )
print("Value-action function estimates with Sarsa")
qv_sarsa = sarsa.get_qv_func_dict()
qv_list = [v for k,v in qv_sarsa.items()]

print(max(list(qv_list[0].values()))*p + max(list(qv_list[1].values()))*(1-p))
print("Best Action for Upper branch is: ", list(qv_list[0].keys())[np.argmax(list(qv_list[0].values()))], '\n')
print("Best Action for Lower branch is: ", list(qv_list[1].keys())[np.argmax(list(qv_list[1].values()))], '\n')

qlearning = TD0_Control(
        mdp_rep_obj,
        epsilon_val,
        epsilon_half_life_val,
        learning_rate_val,
        learning_rate_decay_val,
        episodes_limit,
        max_steps_val,
        'Q-learning'
    )

print("Value-action function estimates with Q-learning")
qv_qlearn = qlearning.get_qv_func_dict()
qv_list = [v for k,v in qv_qlearn.items()]

print(max(list(qv_list[0].values()))*p + max(list(qv_list[1].values()))*(1-p))
print("Best Action for Upper branch is: ", list(qv_list[0].keys())[np.argmax(list(qv_list[0].values()))], '\n')
print("Best Action for Lower branch is: ", list(qv_list[1].keys())[np.argmax(list(qv_list[1].values()))], '\n')
