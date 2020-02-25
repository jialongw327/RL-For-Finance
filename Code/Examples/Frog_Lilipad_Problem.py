import sys 
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Processes\\Markov')
from MDP import MDP

def get_lily_pads_mdp(n: int) -> MDP:
    data = {
        i: {'A': ({i - 1:  i / n, i + 1:  1. - i / n}, 1 / n if i == n - 1 else 0.),
            'B': ({j : 1 / n for j in range(n + 1) if j != i}, 1 / n )
        } for i in range(1, n)
    }
    data[0] = {'A': ({0: 1.}, 0.), 'B': ({0: 1.}, 0.)}
    data[n] = {'A': ({n: 1.}, 0.), 'B': ({n: 1.}, 0.)}

    gamma = 1.0
    return MDP(data, gamma)



if __name__ == '__main__':
    pads: int = 10
    mdp: MDP = get_lily_pads_mdp(pads)
    pol = mdp.policy_iteration(1e-8)
    print(pol.policy_data)
    print(mdp.find_value_func_dict(pol))