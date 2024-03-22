import numpy as np
from copy import deepcopy


def initialize_agent(environment, args):
    if args.algorithm == 'MCC':
        agent = MonteCarloControl(environment, args.hilo)
    elif args.algorithm == 'Q':
        agent = QLearning(environment, args.hilo, args.alpha, args.gamma, args.epsilon)
    elif args.algorithm == 'DQ':
        agent = DeepQLearning(environment, args.hilo, args.alpha, args.gamma, args.epsilon) # just placeholder args for now
    else:
        raise ValueError('Algorithm must be one of MCC, Q, DQ.')
    return agent


class BaseAgent:
    """
    BaseAgent comes with the ability to count cards using the hi-lo method.
    """
    def __init__(self, env, hilo=False, bet_sizes=[1,5,10]):
        self.env = env
        self.bet_sizes = bet_sizes
        self.num_states = env.get_number_of_states()
        self.num_actions = env.get_number_of_actions()
        self.hilo = hilo
        self.hilo_count = 0
        self.count_state = 0
        self.num_count_states = 30 if self.hilo else 1
        self.bet_shape = (self.num_count_states, len(self.bet_sizes))
        self.action_shape = (self.num_count_states, self.num_states, self.num_actions)
    
    def get_true_count(self):
        if self.env.deck.fresh_deck: # reset hi-lo count if we have a new deck
            self.hilo_count = 0
        for card in self.env.open_cards: # update the counts
            if card == 1 or card == 10:
                self.hilo_count -= 1
            elif 2 <= card <= 6:
                self.hilo_count += 1
        true_count = round(self.hilo_count/(len(self.env.deck.cards)/52)) + 15
        true_count = min(max(true_count, 0), 29)
        return true_count
    
    def get_new_true_count(self):
        temp_env = deepcopy(self.env)
        hilo_count = deepcopy(self.hilo_count)
        if temp_env.deck.fresh_deck: # reset hi-lo count if we have a new deck
            hilo_count = 0
        for card in temp_env.open_cards: # update the counts
            if card == 1 or card == 10:
                hilo_count -= 1
            elif 2 <= card <= 6:
                hilo_count += 1
        true_count = round(hilo_count/(len(temp_env.deck.cards)/52)) + 15
        true_count = min(max(true_count, 0), 29)
        return true_count


class MonteCarloControl(BaseAgent):
    def __init__(self, env, hilo, bet_sizes=[1,5,10], gamma=1, epsilon=0.2):
        super().__init__(env, hilo, bet_sizes)
        self.q_table, self.n_table = np.zeros(self.action_shape), np.zeros(self.action_shape)
        self.bet_q_table, self.bet_n_table = np.zeros(self.bet_shape), np.zeros(self.bet_shape)
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
            return np.random.randint(0, len(actions))
        b = np.random.default_rng().uniform(low=0, high=1, size=1)
        if b > self.epsilon: # exploit
            return np.argmax(actions)
        else: # explore
            return np.random.randint(0, len(actions))
    
    def select_bet_size(self):
        self.count_state = self.get_true_count() if self.hilo else self.count_state
        actions = self.bet_q_table[self.count_state, ]
        action = self.e_greedy(actions)
        return self.bet_sizes[action]

    def select_action(self, state):
        self.count_state = self.get_true_count() if self.hilo else self.count_state
        actions = self.q_table[self.count_state, state, ]
        action = self.e_greedy(actions)
        return action

    def update(self, state, action, reward, new_state=None):
        """
        For MCC, which performs its table updates after the episode, 
        the update method only appends to the trajectory.
        """
        self.trajectory.append((self.count_state, state, action, reward))

    def update_tables(self):
        """
        Update q and n tables based on the reversed trajectory if it's the first visit.
        """
        G = 0
        first_visit = []
        for count_state, state, action, reward in self.trajectory[::-1]:
            if (count_state, state, action) not in first_visit:
                first_visit.append((count_state, state, action))
                G += self.gamma*G + reward
                self.update_n(count_state, state, action)
                self.update_q(count_state, state, action, G)
            else:
                print(f"Already visited ({state}, {action})")
        self.trajectory = [] # reset trajectory

    def update_q(self, count_state, state, action, G):
        q = self.q_table[count_state, state, action]
        n = self.n_table[count_state, state, action]
        self.q_table[count_state, state, action] = q + (1/n)*(G - q)
    
    def update_n(self, count_state, state, action):
        self.n_table[count_state, state, action] += 1


class QLearning(BaseAgent):
    def __init__(self, env, hilo, bet_sizes, alpha=0.1, gamma=1, epsilon=0.2):
        super().__init__(env, hilo, bet_sizes=[1,5,10])
        self.q_table = np.zeros(self.action_shape)
        self.bet_q_table = np.zeros(self.bet_shape)
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount rate
        self.epsilon = epsilon # exploration probability threshold
        self.action = None
        self.count_state = 0
        self.new_count_state = 0

    def e_greedy(self, actions):
        """
        Epsilon-greedy decision policy using epsilon threshold and uniform distribution 
        to select between exploration and exploitation.
        """
        # if all q values are the same, break ties randomly and exit early
        if len(set(actions)) == 1: 
            return np.random.randint(0, len(actions))
        b = np.random.default_rng().uniform(low=0, high=1, size=1)
        if b > self.epsilon: # exploit
            return np.argmax(actions)
        else: # explore
            return np.random.randint(0, len(actions))
    
    def select_bet_size(self):
        self.count_state = self.get_true_count() if self.hilo else self.count_state
        actions = self.bet_q_table[self.count_state, ]
        action = self.e_greedy(actions)
        return self.bet_sizes[action]

    def select_action(self, state):
        self.count_state = self.get_true_count() if self.hilo else self.count_state
        actions = self.q_table[self.count_state, state, ]
        action = self.e_greedy(actions)
        return action

    def update(self, state, action, reward, new_state):
        self.count_state = self.get_true_count() if self.hilo else self.count_state
        new_count_state = self.get_new_true_count() if self.hilo else self.new_count_state
        q = self.q_table[self.count_state, state, action]
        self.q_table[self.count_state, state, action] = q + self.alpha*(reward + self.gamma*np.max(self.q_table[new_count_state, new_state, ]) - q)


class DeepQLearning(BaseAgent):
    def __init__(self, env, alpha=0.1, gamma=1, epsilon=0.2):
        self.action = None

