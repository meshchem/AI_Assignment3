# ----------------------------------------------------------------------
# train.py
# CS7IS2 Assignment 3 – Train RL agents for Tic Tac Toe
#
# Usage:
#   python train_ttt.py --agent qlearn --episodes 50000
#   python train_ttt.py --agent dqn    --episodes 50000
#   python train_ttt.py --agent both   --episodes 50000
#
# Saved models:
#   results/qlearn_ttt.pkl
#   results/dqn_ttt.pt
# ----------------------------------------------------------------------

import argparse
import os

from tictactoe import (
    QLearningAgent, DQNAgent,
    DefaultOpponent, RandomOpponent,
    PLAYER_X, PLAYER_O
)

os.makedirs("results", exist_ok=True)

parser = argparse.ArgumentParser(description="Train TTT RL agents")
parser.add_argument("--agent",    choices=["qlearn", "dqn", "both"],
                    default="both")
parser.add_argument("--episodes", type=int, default=50_000)
args = parser.parse_args()

ai_id    = PLAYER_O
human_id = PLAYER_X


# ----------------------------------------------------------------------
# Q-Learning
# ----------------------------------------------------------------------

# def train_qlearn(episodes):
#     print(f"\n{'-'*55}")
#     print(f"Tic Tac Toe: Training Tabular Q-Learning ({episodes} episodes)")
#     print(f"{'-'*55}")

#     # Train against the default opponent, it wins and blocks,
#     # which forces Q-Learning to learn proper counter-strategies
#     opponent = DefaultOpponent(player1=human_id, player2=ai_id)
#     agent = QLearningAgent(player1=ai_id, opp_player=human_id)
#     agent.train(episodes, opponent)
#     agent.save("results/qlearn_ttt.pkl")
#     print(f"  Q-table states learned: {len(agent.q)}")

# Mixed Opponent
def train_qlearn(episodes):
    print(f"\n{'-'*55}")
    print(f"Tic Tac Toe: Training QLearn ({episodes} episodes)")
    print(f"{'-'*55}")

    agent        = DQNAgent(player1=ai_id, opp_player=human_id)
    random_opp   = RandomOpponent()
    default_opp  = DefaultOpponent(player1=human_id, player2=ai_id)

    # Phase 1 train against random opponent first
    # Gives the network easy wins to learn basic patterns from
    phase1 = int(episodes * 0.3)   # 60% of episodes
    print(f"Phase 1: vs Random     ({phase1} episodes)")
    agent.train(phase1, random_opp)

    # Phase 2 — switch to default opponent
    # Network now knows basic strategy, ready to face a smarter opponent
    phase2 = int(episodes * 0.2)      # remaining 40%
    print(f"Phase 2: vs Default    ({phase2} episodes)")
    agent.train(phase2, default_opp)

    phase3 = int(episodes * 0.3)       # remaining 40%
    print(f"Phase 2: vs Default    ({phase3} episodes)")
    agent.train(phase3, random_opp)

    phase4 = int(episodes * 0.1)      # remaining 40%
    print(f"Phase 2: vs Default    ({phase4} episodes)")
    agent.train(phase4, default_opp)

    phase5 = int(episodes * 0.1)       # remaining 40%
    print(f"Phase 2: vs Default    ({phase5} episodes)")
    agent.train(phase5, random_opp)

    agent.save("results/qlearn_mixed_ttt.pkl")


# ----------------------------------------------------------------------
# DQN
# ----------------------------------------------------------------------

# Random Opponent
# def train_dqn(episodes):
#     print(f"\n{'-'*55}")
#     print(f"Tic Tact Toe: Training DQN ({episodes} episodes)")
#     print(f"{'-'*55}")

#     # Train against a random opponent, gives the network a positive
#     # reward signal early in training before it learns anything useful.
#     # A smart opponent at this stage causes the replay buffer to fill
#     # with losses and the network collapses.
#     opponent = RandomOpponent()
#     agent = DQNAgent(player1=ai_id, opp_player=human_id)
#     agent.train(episodes, opponent)
#     agent.save("results/dqn_ttt.pt")

# Mixed Opponent
# def train_dqn(episodes):
#     print(f"\n{'-'*55}")
#     print(f"Tic Tac Toe: Training DQN ({episodes} episodes)")
#     print(f"{'-'*55}")

#     agent        = DQNAgent(player1=ai_id, opp_player=human_id)
#     random_opp   = RandomOpponent()
#     default_opp  = DefaultOpponent(player1=human_id, player2=ai_id)

#     # Phase 1 train against random opponent first
#     # Gives the network easy wins to learn basic patterns from
#     phase1 = int(episodes * 0.3)   # 60% of episodes
#     print(f"Phase 1: vs Random     ({phase1} episodes)")
#     agent.train(phase1, random_opp)

#     # Phase 2 — switch to default opponent
#     # Network now knows basic strategy, ready to face a smarter opponent
#     phase2 = int(episodes * 0.2)      # remaining 40%
#     print(f"Phase 2: vs Default    ({phase2} episodes)")
#     agent.train(phase2, default_opp)

#     phase3 = int(episodes * 0.3)       # remaining 40%
#     print(f"Phase 2: vs Default    ({phase3} episodes)")
#     agent.train(phase3, random_opp)

#     phase4 = int(episodes * 0.1)      # remaining 40%
#     print(f"Phase 2: vs Default    ({phase4} episodes)")
#     agent.train(phase4, default_opp)

#     phase5 = int(episodes * 0.1)       # remaining 40%
#     print(f"Phase 2: vs Default    ({phase5} episodes)")
#     agent.train(phase5, random_opp)

#     agent.save("results/dqn_mixed_ttt.pt")

# Default Opponent
def train_dqn(episodes):
    print(f"\n{'-'*55}")
    print(f"Tic Tact Toe: Training DQN against rule based opponent({episodes} episodes)")
    print(f"{'-'*55}")

    # Train against a random opponent, gives the network a positive
    # reward signal early in training before it learns anything useful.
    # A smart opponent at this stage causes the replay buffer to fill
    # with losses and the network collapses.
    opponent = DefaultOpponent(player1=human_id, player2=ai_id)
    agent = DQNAgent(player1=ai_id, opp_player=human_id)
    agent.train(episodes, opponent)
    agent.save("results/dqn_rule_ttt.pt")

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

if args.agent in ("qlearn", "both"):
    train_qlearn(args.episodes)

if args.agent in ("dqn", "both"):
    train_dqn(args.episodes)

print("\nDone. Models saved to ./results/")