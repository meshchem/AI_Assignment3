import argparse
import os

from tictactoe import (
    QLearningAgent, DQNAgent,
    DefaultOpponent, RandomOpponent,
    PLAYER_X, PLAYER_O
)

os.makedirs("results", exist_ok=True)

player2    = PLAYER_O
player1 = PLAYER_X
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
#     agent = DQNAgent(player1=player2, opp_player=player1)
#     agent.train(episodes, opponent)
#     agent.save("results/dqn_ttt.pt")

# Mixed Opponent
def train_dqn(episodes):
    print(f"\n{'-'*55}")
    print(f"Tic Tac Toe: Training DQN ({episodes} episodes)")
    print(f"{'-'*55}")

    agent        = DQNAgent(player1=player2, opp_player=player1)
    random_opp   = RandomOpponent()
    default_opp  = DefaultOpponent(player1=player1, player2=player2)

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

    agent.save("results/dqn_mixed_ttt.pt")

# Default Opponent
# def train_dqn(episodes):
#     print(f"\n{'-'*55}")
#     print(f"Tic Tact Toe: Training DQN against rule based opponent({episodes} episodes)")
#     print(f"{'-'*55}")

#     # Train against a random opponent, gives the network a positive
#     # reward signal early in training before it learns anything useful.
#     # A smart opponent at this stage causes the replay buffer to fill
#     # with losses and the network collapses.
#     opponent = DefaultOpponent(player1=player1, player2=player2)
#     agent = DQNAgent(player1=player2, opp_player=player1)
#     agent.train(episodes, opponent)
#     agent.save("results/dqn_rule_ttt.pt")

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------


train_dqn(200000)

