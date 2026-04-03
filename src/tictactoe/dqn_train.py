import argparse
import os

from tictactoe import (
    DQNAgent,
    DefaultOpponent, RandomOpponent,
    PLAYER_X, PLAYER_O
)

os.makedirs("results", exist_ok=True)

player2 = PLAYER_O
player1 = PLAYER_X


def train_random(episodes, agent):
    print(f"Strategy: vs Random ({episodes} episodes)")
    opponent = RandomOpponent()
    agent.train(episodes, opponent)
    agent.save("results/dqn_ttt.pt")


def train_default(episodes, agent):
    print(f"Strategy: vs Default ({episodes} episodes)")
    opponent = DefaultOpponent(player1=player1, player2=player2)
    agent.train(episodes, opponent)
    agent.save("results/dqn_rule_ttt.pt")


def train_mixed(episodes, agent):
    print(f"Strategy: Mixed")
    random_opp  = RandomOpponent()
    default_opp = DefaultOpponent(player1=player1, player2=player2)

    phase1 = int(episodes * 0.3)
    print(f"Phase 1: vs Random     ({phase1} episodes)")
    agent.train(phase1, random_opp)

    phase2 = int(episodes * 0.2)
    print(f"Phase 2: vs Default    ({phase2} episodes)")
    agent.train(phase2, default_opp)

    phase3 = int(episodes * 0.3)
    print(f"Phase 3: vs Random     ({phase3} episodes)")
    agent.train(phase3, random_opp)

    phase4 = int(episodes * 0.1)
    print(f"Phase 4: vs Default    ({phase4} episodes)")
    agent.train(phase4, default_opp)

    phase5 = int(episodes * 0.1)
    print(f"Phase 5: vs Random     ({phase5} episodes)")
    agent.train(phase5, random_opp)

    agent.save("results/dqn_mixed_ttt.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for Tic Tac Toe")
    parser.add_argument("--episodes", type=int, default=200000,
                        help="Number of training episodes (default: 200000)")
    parser.add_argument("--strategy", type=str, default="mixed",
                        choices=["random", "default", "mixed"],
                        help="Training strategy (default: mixed)")
    args = parser.parse_args()

    print(f"\nTic Tac Toe: Training DQN ({args.episodes} episodes)")
    agent = DQNAgent(player1=player2, opp_player=player1)

    if args.strategy == "random":
        train_random(args.episodes, agent)
    elif args.strategy == "default":
        train_default(args.episodes, agent)
    else:
        train_mixed(args.episodes, agent)