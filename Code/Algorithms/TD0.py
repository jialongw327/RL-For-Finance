import sys
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Processes\\Markov')
sys.path.append('C:\\Users\\ThinkPad\\Desktop\\CME241\\Push\\Code\\Algorithms')

from Policy import Policy

from MDP_Rep_For_RL_Tabular import MDPRepForRLTabular
from Markov_Functions import get_rv_gen_func_single

class TD0():

    def __init__(
            self,
            mdp_rep_for_rl: MDPRepForRLTabular,
            epsilon: float,
            epsilon_half_life: float,
            learning_rate: float,
            learning_rate_decay: float,
            num_episodes: int,
            max_steps: int
    ) -> None:
        self.mdp_rep: MDPRepForRLTabular = mdp_rep_for_rl
        self.epsilon: float = epsilon
        self.epsilon_half_life: float = epsilon_half_life
        self.learning_rate: float = learning_rate
        self.learning_rate_decay: float = learning_rate_decay
        self.num_episodes: int = num_episodes
        self.max_steps: int = max_steps
        
    def get_value_func_dict(self, pol: Policy):
        sa_dict = self.mdp_rep.state_action_dict
        vf_dict = {s: 0.0 for s in sa_dict.keys()}
        act_gen_dict = {s: get_rv_gen_func_single(pol.get_state_probabilities(s))
                        for s in sa_dict.keys()}
        episodes = 0
        updates = 0

        while episodes < self.num_episodes:
            state = self.mdp_rep.init_state_gen()
            steps = 0
            terminate = False

            while not terminate:
                action = act_gen_dict[state]()
                next_state, reward = \
                    self.mdp_rep.state_reward_gen_dict[state][action]()
                vf_dict[state] += self.learning_rate *\
                    (updates / self.learning_rate_decay + 1) ** -0.5 *\
                    (reward + self.mdp_rep.gamma * vf_dict[next_state] -
                     vf_dict[state])
                updates += 1
                steps += 1
                terminate = steps >= self.max_steps or \
                    state in self.mdp_rep.terminal_states
                state = next_state

            episodes += 1

        return vf_dict
    