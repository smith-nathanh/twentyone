import numpy as np


def initialize_agent(environment, args):
    if args.algorithm == 'MCC':
        agent = MonteCarloControl(environment, args.gamma, args.epsilon)
    elif args.algorithm == 'Q':
        agent = QLearning(environment, args.alpha, args.gamma, args.epsilon)
    elif args.algorithm == 'MCC':
        agent = DeepQLearning(environment, args.alpha, args.gamma, args.epsilon) # just placeholder args for now
    else:
        raise ValueError('Algorithm must be one of MCC, Q, DQ.')
    return agent


class MonteCarloControl:
    def __init__(self, env, gamma=1, epsilon=0.2):
        self.env = env
        self.num_states = env.get_number_of_states()
        self.num_actions = env.get_number_of_actions()
        self.q_table = np.zeros((self.num_states, self.num_actions)) # initialize q_table with zeros
        self.n_table = np.zeros((self.num_states, self.num_actions)) # initialize n_table with zeros
        self.gamma = gamma
        self.epsilon = epsilon
        self.trajectory = []
    
    def get_number_of_states(self):
        return self.num_states

    def get_number_of_actions(self):
        return self.num_actions

    def e_greedy(self, actions):
        """
        Epsilon-greedy decision policy using epsilon threshold and uniform distribution 
        to select between exploration and exploitation. 
        """
        # if all q values are the same, break ties randomly and exit early
        if len(set(actions)) == 1: 
            return np.random.randint(0, self.num_actions)
        b = np.random.default_rng().uniform(low=0, high=1, size=1)
        if b > self.epsilon: # exploit
            return np.argmax(actions)
        else: # explore
            return np.random.randint(0, self.num_actions) 

    def select_action(self, state):
        actions = self.q_table[state, ]
        action = self.e_greedy(actions)
        return action

    def update(self, state, action, reward, new_state=None):
        """
        For MCC, which performs its table updates after the episode, 
        the update method only appends to the trajectory.
        """
        self.trajectory.append((state, action, reward))

    def update_tables(self):
        """
        Update q and n tables based on the reversed trajectory if it's the first visit.
        """
        G = 0
        first_visit = []
        for state, action, reward in self.trajectory[::-1]:
            if (state, action) not in first_visit:
                first_visit.append((state, action))
                G += self.gamma*G + reward
                self.update_n(state, action)
                self.update_q(state, action, G)
            else:
                print(f"Already visited ({state}, {action})")
        self.trajectory = [] # reset trajectory

    def update_q(self, state, action, G):
        q = self.q_table[state, action]
        n = self.n_table[state, action]
        self.q_table[state, action] = q + (1/n)*(G - q)
    
    def update_n(self, state, action):
        self.n_table[state, action] += 1


class QLearning:
    def __init__(self, env, alpha=0.1, gamma=1, epsilon=0.2):
        self.env = env
        self.num_states = env.get_number_of_states()
        self.num_actions = env.get_number_of_actions()
        self.q_table = np.zeros((self.num_states, self.num_actions)) # initialize q_table with zeros
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount rate
        self.epsilon = epsilon # exploration probability threshold
        self.action = None 

    def e_greedy(self, actions):
        """
        Epsilon-greedy decision policy using epsilon threshold and uniform distribution 
        to select between exploration and exploitation.
        """
        # if all q values are the same, break ties randomly and exit early
        if len(set(actions)) == 1: 
            return np.random.randint(0, self.num_actions)
        b = np.random.default_rng().uniform(low=0, high=1, size=1)
        if b > self.epsilon: # exploit
            return np.argmax(actions)
        else: # explore
            return np.random.randint(0, self.num_actions) 

    def select_action(self, state):
        actions = self.q_table[state, ]
        action = self.e_greedy(actions)
        return action

    def update(self, state, action, reward, new_state):
        q = self.q_table[state, action]
        self.q_table[state, action] = q + self.alpha*(reward + self.gamma*np.max(self.q_table[new_state, ]) - q)




class DeepQLearning:
    def __init__(self, env, alpha=0.1, gamma=1, epsilon=0.2):
        self.action = None

