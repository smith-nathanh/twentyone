import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def get_agent_metrics(agent_num, args):
    """
    Retrieve the metrics json file
    """
    agent_args = f"{args.algorithm}_{args.num_episodes}_{args.alpha}_{args.gamma}_{args.epsilon}"
    filename = f'results/Agent_{agent_num}_{agent_args}.json'
    print(filename)
    with open(filename, 'r') as f:
        metrics = json.load(f)
    return metrics


def plot_metrics():
    """
    Plot all the metrics for the agents and the average

    Example
    ------
    $ python plot_the_results.py --algorithm MCC --num_agents 10 --num_episodes 2000 --gamma 0.9 --epsilon 0.2
    """
    parser = argparse.ArgumentParser(description="Gather results from training agents and plot the metrics")
    parser.add_argument("--algorithm", type=str, default="S", help="The algorithm to use ('S': Sarsa or 'Q': Q-learning)")
    parser.add_argument("--num_agents", type=int, default=10, help="Number of agents to train")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to run for each agent")
    parser.add_argument("--alpha", type=float, required=False, default=None, help="Learning rate")
    parser.add_argument("--gamma", type=float, required=False, default=0.9, help="Discount factor")
    parser.add_argument("--epsilon", type=float, required=False, default=0.2, help="Exploration probability threshold")
    parser.add_argument("--get_all_results", action='store_true', default=False, required=False, help="Flag to get all results")
    args = parser.parse_args()

    agent_args = f"{args.algorithm}_{args.num_episodes}_{args.alpha}_{args.gamma}_{args.epsilon}"
    print((f"\nGathering results for {agent_args}\n"))

    win = pd.DataFrame()
    rewards = pd.DataFrame()
    for i in range(args.num_agents):
        metrics = get_agent_metrics(i, args)
        win[f'Agent_{i}'] = metrics['Win_Percentage']
        rewards[f'Agent_{i}'] = metrics['Cumulative_Reward']

    win['Average'] = win.mean(axis=1)
    win_max = win.loc[49:, 'Average'].max()
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for col in win.columns:
        if col == 'Average':
            plt.plot(win[col], label=col, linewidth=2, color='black')
        else:
            plt.plot(win[col], label=col, linewidth=1, alpha=0.4)
    ax.set_title('Win Percentage')
    ax.xaxis.set_label_text('Episodes')
    ax.yaxis.set_label_text('Win Percentage')
    ax.legend(loc='upper right', bbox_to_anchor=(.99, .99),
              ncol=2, fancybox=True, shadow=True)
    ax.text(1000, 0.2, f"Max after 50 episodes: {win_max:0.2f}", fontsize=8, color='black', bbox=dict(facecolor='white', edgecolor='black'))
    plt.savefig(f'results/Wins_{agent_args}.png')

    rewards['Average'] = rewards.mean(axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for col in rewards.columns:
        if col == 'Average':
            plt.plot(rewards[col], label=col, linewidth=2, color='black')
        else:
            plt.plot(rewards[col], label=col, linewidth=1, alpha=0.4)
    ax.set_title('Cumulative Rewards')
    ax.xaxis.set_label_text('Episodes')
    ax.yaxis.set_label_text('Cumulative Rewards')
    ax.legend()
    plt.savefig(f'results/Rewards_{agent_args}.png')


if __name__ == "__main__":
    plot_metrics()