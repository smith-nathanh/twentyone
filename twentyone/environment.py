import numpy as np
import random


class CardDeck:
    """For shuffling and dealing cards"""

    def __init__(self, num_decks=6):
        self.cards = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                      6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10,
                      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        self.num_decks = num_decks
        self.cards = [card for card in self.cards for _ in range(self.num_decks)]
        self.deal_seq = []

    def shuffle_cards(self):
        self.deal_seq = random.sample(self.cards, len(self.cards))

    def deal_card(self):
        return self.deal_seq.pop(0)


class Blackjack:
    def __init__(self, bet_sizes=[1, 5, 10]):

        self.deck = CardDeck()
        self.agent_total = 0
        self.usable_ace = 0
        self.dealer_card = 0
        self.dealer_total = 0
        self.dealer_ace = 0
        self.current_state = 0
        self.bet_sizes = bet_sizes
        self.num_states = 204
        self.num_actions = 2
        self.open_cards = []
        self.agent_natural = False
        
    def get_number_of_states(self):
        return self.num_states

    def get_number_of_actions(self):
        return self.num_actions

    def get_state(self):
        return self.current_state

    def get_state_index(self):
        a_idx = self.agent_total - 12
        d_idx = 10 * (self.dealer_card - 1)
        u_idx = 100 * self.usable_ace
        return a_idx + d_idx + u_idx

    def get_next_state(self):
        new_card = self.deck.deal_card()
        self.agent_total += new_card
        if self.agent_total > 21 and self.usable_ace == 1:
            self.usable_ace = 0
            self.agent_total -= 10
        if self.agent_total > 21:
            new_state = 201      # 201 is the losing state
        else:
            new_state = self.get_state_index()
        return new_state

    def reset(self):
        self.deck.shuffle_cards()
        self.agent_total = 0
        self.usable_ace = 0
        self.dealer_card = 0
        self.dealer_total = 0
        self.dealer_ace = 0
        self.current_state = 0

        # deal a face up card and a second card to the dealer
        self.dealer_card = self.deck.deal_card()
        d_card_2 = self.deck.deal_card()
        self.dealer_total = self.dealer_card + d_card_2
        if self.dealer_card == 1 or d_card_2 == 1:
            self.dealer_ace = 1
            self.dealer_total += 10

        # deal two cards to the agent
        card_1 = self.deck.deal_card()
        card_2 = self.deck.deal_card()
        self.open_cards = [d_card_2, card_1, card_2]
        self.agent_total = card_1 + card_2
        if card_1 == 1 or card_2 == 1:
            self.usable_ace = 1
            self.agent_total += 10

        # check to see if the agent has a natural (ace + face card)
        if self.agent_total == 21:
            if self.dealer_total == 21:
                self.current_state = 202    # tie game
            else:
                self.current_state = 203    # agent wins
                self.agent_natural = True

        # otherwise, deal enough cards to the agent so that the total is >11
        else:
            while self.agent_total < 12:
                new_card = self.deck.deal_card()
                self.agent_total += new_card
                if new_card == 1 and self.usable_ace == 0 and self.agent_total < 12:
                    self.usable_ace = 1
                    self.agent_total += 10
            # now determine the initial state
            self.current_state = self.get_state_index()

        # reset complete; return the initial state
        return self.current_state

    # Use the agent's action to determine the next state and reward
    def execute_action(self, action, bet_size):

        self.open_cards = []

        # action is 'stick'
        if action == 0:
            # dealer's turn
            while self.dealer_total < 17:
                new_card = self.deck.deal_card()
                self.open_cards.append(new_card)
                self.dealer_total += new_card
                if new_card == 1 and self.dealer_ace == 0 and self.dealer_total < 12:
                    self.dealer_ace = 1
                    self.agent_total += 10
                if self.dealer_total > 21 and self.dealer_ace == 1:
                    self.dealer_ace = 0
                    self.agent_total -= 10
            if self.dealer_total > 21:
                # dealer busted; agent wins
                new_state = 203
                reward = 1*bet_size
                game_end = True
            else:
                if self.dealer_total > self.agent_total:
                    # dealer wins
                    new_state = 201
                    reward = -1*bet_size
                    game_end = True
                elif self.dealer_total < self.agent_total:
                    # agent wins
                    new_state = 203
                    reward = 1*bet_size
                    game_end = True
                else:
                    # tie
                    new_state = 202
                    reward = 0
                    game_end = True

        # action is 'hit'
        elif action == 1:
            new_state = self.get_next_state()
            if new_state == 201:
                reward = -1*bet_size
                game_end = True
            else:
                reward = 0
                game_end = False

        self.current_state = new_state
        return new_state, reward, game_end