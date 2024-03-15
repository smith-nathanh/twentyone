# twentyone
Toolbox for training reinforcement learning agents to play blackjack (otherwise known as twenty-one). 


## Summary of the Blackjack game

### A standard "Bridge" set of cards:
- numbered cards worth their value
- face cards (K, Q, J) worth 10
- Ace can be worth 1 or 11
- Suits don’t matter
- Value of a hand is the sum of the values of the cards held
### 2 players (competitive, zero sum):
- Player (agent)
- Dealer (in environment) 
### Number of decks:
- 6 decks of cards that gradually deplete until 2.4 decks (40% of original) remain, then original 6 decks are shuffled together and play resumes
### Goal of players:
- Draw cards as necessary to obtain hand with largest value 𝑃 without exceeding 21
- Size the bet in order to maximize expected return based on the state of the deck

## Sequence of play:
- The player chooses how much to bet (bet sizes: 1 or 10)
- Two cards are dealt to both the Player and the Dealer.  The second card dealt to the Dealer is placed face up (so that its value is known to both players).
- If the Player has a value of 21 (i.e., an ace and a face card or 10), which is called a natural, and the Dealer does not have a natural, the Player wins 1.5x the bet and the episode is complete.  If both players have a natural, the episode is a tie.
- If the value of the Player’s hand is less than 21, they can decide to select additional cards, one by one (called ‘hit’) until they decide to stop (‘stick’) or exceed a value of 21 (‘goes bust’).  If the Player goes bust, the Player loses its bet.  Otherwise, it becomes the Dealer’s turn.
- The Dealer follows a fixed strategy: they hit on a value less than 17 or stick for a value of 17 or greater.  If the Dealer goes bust, the Player wins its bet. 
- If both the Player and the Dealer have stuck without going bust, the value of the Player’s hand is compared to the value of the Dealer’s hand.  If the values of the hands are the same, it is a tie.  If the value of the Player’s hand is larger than that of the Dealer, the Player wins its bet; if the Dealer’s is larger, the Player loses its bet.

## Hi-Lo Card Counting Strategy
In one condition, player’s state will be informed according the Hi-Lo card counting strategy
The Hi-Lo count starts at 0 and accumulates into a running count as cards are played in the following way:
- 2,3,4,5,6: +1
- 7,8,9: 0
- 10, A: -1
A “True Count” is then computed by dividing the running count by the number of decks remaining
Hi-Lo count resets to 0 when the 6 decks are reshuffled
Strategy reference: https://wizardofodds.com/games/blackjack/card-counting/high-low/ 


## Algorithms in the Toolbox
- Monte Carlo Control
- Q-learning
- Deep Q-learning


## Getting Started:

```{python}
from twentyone.environment import BlackJack
```
