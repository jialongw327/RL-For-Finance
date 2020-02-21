import matplotlib.pyplot as plt
import numpy as np


class AmericanOptionDP():
    
    def __init__(self, T: int, p:float, u: float, d: float, K: float, s0: float) -> None:
        self.T = T
        self.p = p
        self.u = u
        self.d = d
        self.K = K
        self.s0 = s0
        
        
    def find_matrices(self) -> np.array:
        price_matrix =np.zeros((self.T+1, self.T+1))
        cash_flow_matrix = np.zeros((self.T+1, self.T+1))
        for i in range(T+1):
            for j in range(i,T+1):
                price_matrix[i, j] = ((1+u)**(j-i))*((1+d)**(i))
                if (price_matrix[i, j]) != 0:
                    cash_flow_matrix[i,j] = (self.K - self.s0*price_matrix[i, j])*((self.K - self.s0*price_matrix[i, j])>0)
        #price_matrix = self.s0*price_matrix.T
        #cash_flow_matrix =  cash_flow_matrix.T
        self.price_matrix = price_matrix*self.s0
        self.cash_flow_matrix = cash_flow_matrix
        return price_matrix*self.s0, cash_flow_matrix
    
    def find_value_matrix(self):
        value_matrix = self.cash_flow_matrix.copy()
        action_matrix = np.zeros(value_matrix.shape)
        for i in reversed(np.arange(self.T)):
            next_values = value_matrix[:(i+2),i+1].copy()
            current_values = value_matrix[:(i+1),i].copy()
            expected_returns = (1 - self.p)*next_values[:-1] + self.p*next_values[1:]
            
            cur_exp_compare = np.concatenate((expected_returns.reshape(len(expected_returns),1), 
                                  current_values.reshape(len(current_values),1)),1 )
            action_matrix[:(i+1),i] = np.apply_along_axis(np.argmax, 1,  cur_exp_compare)
            
            value_matrix[:(i+1),i] = np.apply_along_axis(max, 1,  cur_exp_compare)
            
            self.value_matrix = value_matrix
            self.action_matrix = action_matrix
        return action_matrix, value_matrix
    
    def find_max_price_sequence(self):
        max_price_seq = np.zeros(T+1)
        max_price_loc = sum(self.action_matrix) - 1
        for i in np.arange(T+1):
            if max_price_loc[i] == -1:
                max_price_seq[i] = -1
            else:
                max_price_seq[i] = self.price_matrix[i - int(max_price_loc[i]),i]
            
        return max_price_seq, self.value_matrix[0,0]
        
    def build_mdp_graph(self):
        return ValueError
        
if __name__ == '__main__':
    T = 2
    s0 = 100
    u = 0.05
    d = -0.03
    p = 0.56
    K = 105
    
    test = AmericanOptionDP(T,p,u,d,K,s0)
    price_matrix, cash_flow_matrix = test.find_matrices()
    action_matrix, value_matrix = test.find_value_matrix()
    max_price_seq, temp = test.find_max_price_sequence()
    #print(cash_flow_matrix)
    
    start = len(max_price_seq[:-1][max_price_seq[:-1]==-1])
    end = 1
    x_axis = np.linspace(start, end, end - start +1)
    price = max_price_seq[max_price_seq>0]
    plt.plot(x_axis, price)
    plt.title('The maximum price to execuate the put option')
    plt.ylabel('max price')
    plt.xlabel('time')
    plt.show()