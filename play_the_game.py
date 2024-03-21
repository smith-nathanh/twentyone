
import argparse
import json
import math
from collections import defaultdict
from twentyone.environment import Blackjack
from twentyone.agents import initialize_agent

    

def main():
    """
    Entry point to train agents to play blackjack using Monte Carlo Control, Q-learning, or Deep Q-learning.

    Takes the following arguments via the command line:
        --algorithm: The algorithm to use: MCC, Q, or DQ
        --num_agents: Number of agents to train
        --num_episodes: Number of episodes to play
        --alpha: Learning rate
        --gamma: Discount factor 
        --epsilon: Exploration probability threshold
        --decay_epsilon: Flag to decay epsilon or not
        --output_path: Output path to save results

    Raises
    ------
    ValueError
        If environment and agent parameters do not match
    
    Example
    -------
    $ python play_the_game.py --algorithm MCC --num_agents 10 --num_episodes 2000 --gamma 0.9 --epsilon 0.2
    """
    parser = argparse.ArgumentParser(description="Blackjack RL Program")
    parser.add_argument("--algorithm", type=str, default="MCC", help="The algorithm to use: MCC, Q, or DQ")
    parser.add_argument("--num_agents", type=int, default=10, help="Number of agents to train")
    parser.add_argument("--num_episodes", type=int, default=2000, help="Number of episodes to play")
    parser.add_argument("--alpha", type=float, required=False, default=None, help="Learning rate")
    parser.add_argument("--gamma", type=float, required=False, default=0.9, help="Discount factor")
    parser.add_argument("--epsilon", type=float, required=False, default=0.2, help="Exploration probability threshold")
    parser.add_argument("--decay_epsilon", action='store_true', default=False, required=False, help="Flag to decay epsilon or not")
    parser.add_argument("--output_path", type=str, required=False, default='results/', help="Output path to save results")
    args = parser.parse_args()
    
    # iterate over the agents
    for i in range(args.num_agents):

        # initialize the environment and agent
        environment = Blackjack()
        agent = initialize_agent(environment, args)
    
        # play the episodes
        wins = 0
        cumulative_reward = 0
        metrics = defaultdict(list)
        for episode in range(args.num_episodes):
            current_state = environment.reset()
            game_end = False
            bet_size = agent.select_bet_size()
            while not game_end:
                action = agent.select_action(current_state)
                new_state, reward, game_end = environment.execute_action(action, bet_size)
                agent.update(current_state, action, reward, new_state)
                current_state = new_state

            # MCC performs its updates after the episode
            if args.algorithm == 'MCC':
                agent.update_tables()
            
            # decay epsilon if requested
            agent.epsilon = agent.epsilon*math.exp(math.log(0.01/agent.epsilon)/args.num_episodes) if args.decay_epsilon else agent.epsilon

            # record a win if episode ended with a reward
            wins += 1 if reward > 1 else 0
            cumulative_reward += reward

            # record the metrics
            metrics['Win_Percentage'].append(wins/(episode+1))
            metrics['Cumulative_Reward'].append(cumulative_reward)

            # print metrics
            if episode % 100 == 0 or episode == args.num_episodes - 1:
                print((f"Episode {episode} --- Win Percentage: {metrics['Win_Percentage'][-1]:.3f}, " 
                       f"Cumulative Reward: {metrics['Cumulative_Reward'][-1]}, "))
        
        # write results to files
        agent_args = f"{args.algorithm}_{args.num_episodes}_{args.alpha}_{args.gamma}_{args.epsilon}"
        with open(f'results/Agent_{i}_{agent_args}.json', 'wt') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Agent {i} trained successfully.\n")

    print("\nProgram completed successfully.\n")
        

if __name__ == "__main__":
    main()