"""
Filename:

Project:

Description:

Author:

Date:
"""

import numpy as np


class RLAgent:
    """RL agent for the blackjack"""

    def __init__(self, hi_lo):
        self.hi_lo = hi_lo
        if self.hi_lo:
            self.q = [np.zeros((204, 3), dtype="float64") for _ in range(30)]  # the q value for each state-action
            self.q_count = [np.zeros((201, 3), dtype="float64") for _ in range(30)]  # for tracking how many visits the agent has made to each state-action
        else:
            self.q = [np.zeros((204, 3), dtype="float64")]  # the q value for each state-action
            self.q_count = [np.zeros((201, 3), dtype="float64")]  # for tracking how many visits the agent has made to each state-action
        self.count_state = 0
        self.hand_state = 0
        self.reward = 0
        self.action = 0
        self.epsilon = 0.2
        self.gamma = 1
        self.current_policy = []
        self.bet_choice = [1, 5, 10]

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def reset_policy(self, count_state):
        """
        Reset the agent's policy and select a bet size
        :return: the chosen bet size
        """
        self.current_policy = []

        if self.hi_lo:
            self.count_state = count_state

        # select a bet size for this hand with epsilon greedy
        actions = self.q[self.count_state][200, ]
        action = self.e_greedy(actions)
        self.store_policy(200, action, 0)

        return self.bet_choice[action]

    def e_greedy(self, actions):
        """
        Identify the epsilon-greedy action
        :param actions: an array whose index values are actions and values are those values associated with each action
        :return: the chosen action
        """
        # identify the action with the largest value and return that action with probability 1 - epsilon,
        # else return an action at random.  If more than one action have the max value, choose between
        # the optimal actions at random
        max_value = actions.max()
        max_indices = np.where(actions == max_value)[0]
        max_index = np.random.choice(max_indices)

        rng = np.random.default_rng()
        if self.epsilon <= rng.random():
            return max_index
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx

    def select_action(self, hand_state):
        """
        Given a particular state, pick a particular action
        :param state: the state of the game
        :return: the selected action
        """
        self.hand_state = hand_state

        # identify possible actions for this state and select one with epsilon-greedy
        # note that we focus on only the first two columns, as the third column applies only to bet-size
        actions = self.q[self.count_state][hand_state, ][:2]
        action = self.e_greedy(actions)
        self.action = action
        return action

    def store_policy(self, hand_state, action, reward):
        """
        Append the latest state-action-reward combo to the agent's policy
        :param state: the current state of the game
        :param action: the action taken from this state
        :param reward: the reward from transitioning to this state
        :return: n/a
        """
        self.current_policy.append([hand_state, action, reward])

    def update_q(self, reward):
        """
        Update the q-values (values for each state-action pair) to account for the latest episode
        :param reward: the final reward associated with transitioning to the terminal state of the episode
        :return: n/a
        """
        g_val = 0

        # reverse the policy
        policy_reversed = self.current_policy[::-1]

        # for each state-action-reward combination stored for this policy, working in reverse
        for idx, state_action_reward in enumerate(policy_reversed):
            # update g_val to account for the next reward; note that for the first g_val,
            # reward comes from an input parameter, not from the stored policy information
            g_val = g_val * self.gamma + reward
            state, action, reward = state_action_reward

            # increment q_count to track the number of visits to this state-action pair
            self.q_count[self.count_state][state, action] += 1

            # update the q value at this state-action pair
            self.q[self.count_state][state, action] += (1/self.q_count[self.count_state][state, action])*(g_val - self.q[self.count_state][state, action])



