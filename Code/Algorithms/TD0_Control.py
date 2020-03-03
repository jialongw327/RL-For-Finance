import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Processes\\Markov')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Algorithms')

from TD0 import TD0
from MDP_Rep_For_RL_Tabular import MDPRepForRLTabular
from Markov_Functions import SAf, find_expected_action_value,\
     find_epsilon_greedy_action
     
class TD0_Control(TD0):
    
    def __init__(
            self,
            mdp_rep_for_rl: MDPRepForRLTabular,
            epsilon: float,
            epsilon_half_life: float,
            learning_rate: float,
            learning_rate_decay: float,
            num_episodes: int,
            max_steps: int,
            algorithm: str
    ) -> None:
        super().__init__(
                mdp_rep_for_rl,
                epsilon,
                epsilon_half_life,
                learning_rate,
                learning_rate_decay,
                num_episodes,
                max_steps)
        self.algorithm = algorithm
        

    def get_qv_func_dict(self) -> SAf:
        sa_dict = self.mdp_rep.state_action_dict
        qf_dict = {s: {a: 0.0 for a in v} for s, v in sa_dict.items()}
        episodes = 0
        updates = 0
        
        while episodes < self.num_episodes:
            state = self.mdp_rep.init_state_gen()
            action = find_epsilon_greedy_action(qf_dict[state], self.epsilon)
            steps = 0
            terminate = False
            
            while not terminate:
                next_state, reward = \
                    self.mdp_rep.state_reward_gen_dict[state][action]()

                if self.algorithm == 'Q-learning':
                    next_qv = max(qf_dict[next_state][a] for a in
                                  qf_dict[next_state])
                    next_action, next_qv = max(qf_dict[next_state].items(), \
                                          key = lambda l:l[1])
                elif self.algorithm == 'Sarsa':
                    next_qv = find_expected_action_value(qf_dict[next_state], \
                                                        self.epsilon)
                    next_action = find_epsilon_greedy_action(qf_dict[next_state], \
                                                        self.epsilon)
                else:
                    raise RuntimeError('Algorithm wrong, use Q-learning or Sarsa')
                qf_dict[state][action] += self.learning_rate *\
                    (updates / self.learning_rate_decay + 1) ** -0.5 *\
                    (reward + self.mdp_rep.gamma * next_qv -
                     qf_dict[state][action])
                updates += 1
                steps += 1
                terminate = steps >= self.max_steps or \
                    state in self.mdp_rep.terminal_states
                state = next_state
                action = next_action
                
            episodes += 1
             
        return qf_dict