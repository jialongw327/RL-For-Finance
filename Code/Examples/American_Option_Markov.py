import sys 
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Processes\\Markov')
from MDP_Refined import MDPRefined

class AmericanOptionMDP():
    
     def __init__(self, T: int, p:float, u: float, d: float, K: float, s0: float)\
    -> None:
        self.T = T
        self.p = p
        self.u = u
        self.d = d
        self.K = K
        self.s0 = s0
        
    
     def build_graph(self):
        u = self.u
        d = self.d
        T = self.T
        K = self.K
        prob = self.p
        state = [(((1.0 + u)**i)*((1+d)**(t-i))*s0, t) for t in range(1, T+1) for i in range(t+1)]
        tr = {s:{'Exercise':{'Terminal': (1.0, max(K - s[0],0))}, \
                             'Stay':{(round(s[0]*(1+u),10), s[1]+1):(prob, 0.0), \
                             (round(s[0]*(1+d),10), s[1]+1):(1 - prob, 0.0)} } \
                            if s[1] <= T-1 else \
     {'Exercise': {'Terminal': (1.0,  max(K - s[0],0))}} for s in state}
    
        state = [(round(s[0],10),s[1]) for s in state]
        tr = {(round(s[0],10), s[1]): v for s,v in tr.items()}
        
        state.append('Terminal')
        tr['Terminal'] = {'Exercise': {'Terminal': (1.0, 0.0)}, 'Stay': {'Terminal': (1.0, 0.0)}}

        return state, tr
        

if __name__ == '__main__':
    T = 2
    s0 = 100
    u = 0.05
    d = -0.03
    p = 0.56
    K = 105
    
    gamma = 1 
    test = AmericanOptionMDP(T,p,u,d,K,s0)
    state, tr = test.build_graph()
    test_mdp = MDPRefined(tr, gamma)

    #Generate Guess Policy
    # Generate policy data
    policy_data = {s: {'Exercise': 0.6, 'Stay': 0.4} if s[0] < 10 else {'Stay': 0.6, 'Exercise': 0.4} \
                   for s in test_mdp.nt_states if s[1]<T}
    policy_data['Terminal'] = {'Exercise': 0.5, 'Stay': 0.5}
    for s in state:
        if s[1] == T:
            policy_data[s] = {'Exercise': 1.0}


    pol_opt= test_mdp.policy_iteration()
    val_opt = test_mdp.find_value_func_dict(pol_opt)
    val_list = [(k, v) for k,v in val_opt.items()]
    print('Value of an American Put by policy iteration: ')
    print(val_list[0][1]*p+ val_list[1][1]*(1-p))
















if __name__ == '__main__':
    # Generate mdp
    T = 20
    s0 = 100
    u = 0.05
    d = -0.03
    p = 0.56
    K = 105
    
    test_mdp = AmericanOptionMDP(T,p,u,d,K,s0)
    state = test_mdp.build_graph()
    

