from twentyone.environment import Blackjack
from twentyone.agents import MonteCarloControl, QLearning, DeepQLearning
import argparse
from collections import defaultdict
import json


def initialize_agent(args, environment):
    if args.algorithm == 'MCC':
        agent = MonteCarloControl(environment, args.gamma, args.epsilon)
    elif args.algorithm == 'Q':
        agent = QLearning(environment, args.alpha, args.gamma, args.epsilon)
    elif args.algorithm == 'MCC':
        agent = DeepQLearning(environment, args.alpha, args.gamma, args.epsilon) # just placeholder args for now
    else:
        raise ValueError('Algorithm must be one of MCC, Q, DQ.')
    return agent
    

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

    Raises
    ------
    ValueError
        If environment and agent parameters do not match
    """
    parser = argparse.ArgumentParser(description="Blackjack RL Program")
    parser.add_argument("--algorithm", type=str, default="MCC", help="The algorithm to use: MCC, Q, or DQ")
    parser.add_argument("--num_agents", type=int, default=10, help="Number of agents to train")
    parser.add_argument("--num_episodes", type=int, default=2000, help="Number of episodes to play")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, required=False, default=0.9, help="Discount factor")
    parser.add_argument("--epsilon", type=float, required=False, default=0.2, help="Exploration probability threshold")
    parser.add_argument("--output_path", type=str, required=True, default='results/', help="Output path to save results")
    args = parser.parse_args()
    
    # iterate over the agents
    for i in range(args.num_agents):

        # initialize the environment and agent
        environment = Blackjack()
        agent = initialize_agent(args, environment)
    
        # play the episodes
        wins = 0
        cumulative_return = 0
        metrics = defaultdict(list)
        for e in range(args.num_episodes):
            current_state = environment.reset()
            game_end = False
            while not game_end:
                action = agent.select_action(current_state)
                new_state, reward, game_end = environment.execute_action(action)
                agent.update((current_state, action, reward, new_state))
                current_state = new_state

            # MCC performs its updates after the episode
            if args.algorithm == 'MCC':
                agent.update_tables()

            # record a win if episode ended with a reward
            wins += 1 if reward == 1 else 0

            # record the metrics
            metrics['Win_Percentage'].append(wins/(e+1))
            metrics['Cumulative_Return'].append(cumulative_return)

            # print metrics
            if e % 100 == 0 or e == args.num_episodes - 1:
                print((f"Episode {e} --- Win Percentage: {metrics['Win_Percentage'][-1]:.3f}, " 
                       f"Cumulative Return: {metrics['Cumulative_Return'][-1]}, "))
        
        # write results to files
        filename = f'{args.output_path}/Agent_{i}_{args.epsilon}_Metrics.json'
        with open(filename, 'wt') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Agent {i} trained successfully.\n")

    print("\nProgram completed successfully.\n")
        

if __name__ == "__main__":
    main()