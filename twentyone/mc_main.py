"""
Filename:

Project:

Description:

Author:

Date:
"""

import math
import numpy as np
import agent_mc as ag2
import environment as env2


def get_agent_hand(state):
    """
    Convert state to player hand.
    :param state: the game state
    :return: the player hand, and True if there is a usable ace
    """
    return state % 10 + 12, state - 100 >= 0


def get_dealer_hand(state):
    """
    Convert state to dealer face-up card.
    :param state: the game state
    :return: the dealer face-up card
    """
    return ((state - state % 10) / 10) % 10 + 1


def adjust_count(cards):
    count = 0
    for card in cards:
        if card == 1 or card == 10:
            count -= 1
        elif 2 <= card <= 6:
            count += 1
    return count


def play_blackjack(num_episodes, epsilon, decay_epsilon, hi_lo):
    """
    Play the game of blackjack
    :param num_episodes: the number of episodes of the game to play
    :return: three 1 dimensional np arrays of size num_episodes,
    the first holding the percent of episode wins by episode
    the second holding the cumulative return by episode
    the third holding the percent of states visited by episode
    as well as the agent two-dimensional q-table and q_count table
    """
    # load the environment and agent
    environment = env2.Blackjack()
    agent = ag2.RLAgent(hi_lo)

    # initialize variables
    total_return = 0 # return over a set of decks
    hand_count = 0 # how many hands played over a set of decks
    episode_return = np.zeros(num_episodes, dtype="float64")
    decay_factor = 1
    cur_episode = 0
    hi_lo_count = 0

    agent.set_epsilon(epsilon)

    # define the decay factor as that which linearly decays epsilon to 0.01 over the course of the episodes
    if decay_epsilon:
        decay_factor = math.exp(math.log(0.01/epsilon)/num_episodes)

    # each episode
    while True:
        # reset the environment and observe the current state and new_deck; the latter is True
        # if we are on to a new set of decks
        current_state, open_cards = environment.reset()

        hi_lo_count += adjust_count(open_cards)

        # at the start of each hand, we reset the agent's policy and retrieve the bet size
        true_count = round(hi_lo_count/(len(environment.deck.cards)/52)) + 15
        true_count = max(true_count, 0)
        true_count = min(true_count, 29)
        # print(true_count)

        bet_size = agent.reset_policy(true_count)

        reward = 0
        if current_state == 203: # occurs only from natural blackjack, which pays 1.5x
            reward = 1.5 * bet_size

        # Do until the game ends:
        while current_state < 200:

            # select an action
            action = agent.select_action(current_state)

            # store this state, action, and reward.  Note that reward refers to
            # that which was earned on the transition to this state, and thus starts at 0
            agent.store_policy(current_state, action, reward)

            # execute the action and identify the new state and the reward for transitioning from current_state to new_state
            new_state, reward, open_cards = environment.execute_action(action)

            hi_lo_count += adjust_count(open_cards)

            reward *= bet_size

            # transition current_state to new_state and increment total_return with the reward received
            current_state = new_state
            total_return += reward

        hand_count += 1

        # update the agent's q table
        agent.update_q(reward)

        agent.set_epsilon(agent.epsilon*decay_factor)

        hidden_card = environment.dealer_card

        hi_lo_count += adjust_count([hidden_card])

        # reset the environment if we've exceeded 60% of cards, and
        # break out of the script if we've exceeded num_episodes
        if len(environment.deck.cards) < 125:
            episode_return[cur_episode] = total_return/hand_count
            cur_episode += 1

            if cur_episode >= num_episodes:
                break

            environment = env2.Blackjack()
            total_return = 0
            hand_count = 0
            hi_lo_count = 0

    return episode_return, agent.q, agent.q_count


# if this script is being run as the main program
if __name__ == "__main__":

    # we test epsilon in three ways; the first feature is the initial epsilon value, the second is whether to decay epsilon
    epsilon_choice = [[0.3, True]] # [0.2, False], [0.1, False],
    
    hi_lo = True

    with open('output.txt', 'wt') as f:
        for choice in epsilon_choice:

            # initialize variables
            num_episodes = 100000
            num_agents = 1
            all_player_return = np.zeros(num_episodes, dtype="float64")
            all_player_q_values = np.zeros((204, 3), dtype="float64")
            all_player_q_count = np.zeros((201, 3), dtype="float64")

            # for each agent, play blackjack and increment variables that track win rate, cumulative return, and state visit rate
            # at each episode
            for i in range(num_agents):
                cur_return, q_values, q_count = play_blackjack(num_episodes, choice[0], choice[1], hi_lo)
                all_player_return += cur_return
                all_player_q_values += q_values[0]
                all_player_q_count += q_count[0]

            # print the variables at each episode
            print(f'\nEpsilon = {choice[0]}, Decay epsilon = {choice[1]}', file=f)
            print('Episode, Avg % wins, Avg % returns, Avg % states visited:', file=f)
            for episode in range(num_episodes):
                if episode % 1000 == 0:
                    print(f'{episode}, {all_player_return[episode]/num_agents:.2f}', file=f)

            if not hi_lo:
                print(f'Agent hand, usable ace, dealer hand, stick-value, hit-value, stick-count, hit-count', file=f)
                for idx, row in enumerate(all_player_q_values):
                    if idx < 200:
                        agent_hand, usable_ace = get_agent_hand(idx)
                        dealer_hand = get_dealer_hand(idx)
                        print(agent_hand, usable_ace, dealer_hand, row[0]/num_agents, row[1]/num_agents, sep=',', file=f)
                    if idx == 200:
                        print(f'\nbet_1 value, bet_5 value, bet_10 value', file=f)
                        print(row[0]/num_agents, row[1]/num_agents, row[2]/num_agents, sep=',', file=f)
            
            for idx, row in enumerate(q_values):
                print(f'\nbet_1 value, bet_5 value, bet_10 value', file=f)
                print(row[200][0]/num_agents, row[200][1]/num_agents, row[200][2]/num_agents, sep=',', file=f)


