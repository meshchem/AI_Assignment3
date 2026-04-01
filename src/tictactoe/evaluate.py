import argparse
import os
import csv

from tictactoe import (
    create_board, make_move, check_winner, is_terminal,
    QLearningAgent, DQNAgent, MinimaxAgent,
    DefaultOpponent, RandomOpponent,
    PLAYER_X, PLAYER_O
)

# OUTPUT_DIR = "results/random_dqn/"
OUTPUT_DIR = "results/mixed_dqn/"
# OUTPUT_DIR = "results/rule_dqn/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Evaluate Tic Tac Toe agents")
parser.add_argument("--games", type=int, default=1000,
                    help="Games per matchup")

args = parser.parse_args()

NUM_GAMES = args.games
player2_id = PLAYER_O
player1_id = PLAYER_X

# ----------------------------------------------------------------------
# Load agents
# ----------------------------------------------------------------------

minimax = MinimaxAgent(player1=player2_id, opp_player=player1_id, use_alpha_beta=False)
minimax_ab = MinimaxAgent(player1=player2_id, opp_player=player1_id, use_alpha_beta=True)

qlearn = QLearningAgent(player1=player2_id, opp_player=player1_id)
qlearn.load("results/qlearn_ttt.pkl")

dqn = DQNAgent(player1=player2_id, opp_player=player1_id)
dqn.load("results/dqn_rule_ttt.pt")

default = DefaultOpponent(player1=player1_id, player2=player2_id)
algo_names = ["Minimax", "Minimax-AB", "Q-Learning", "DQN"]
agents = [("Minimax", minimax), ("Minimax-AB", minimax_ab), ("Q-Learning", qlearn), ("DQN", dqn)]

print("Agents loaded")


# ----------------------------------------------------------------------
# Play num_games between agent1 and agent2.
# Alternates who goes first each game to remove first-mover bias.
# agent1 is always the target agent for win/loss counting.
# Also tracks how often agent1 wins when playing as P1 vs P2.
# Returns dict with win, draw, loss pct and P1/P2 breakdown.
# ----------------------------------------------------------------------

def play_games(agent1, agent2, num_games=NUM_GAMES, label=""):
   
    wins = draws = losses = 0

    # P1/P2 win breakdown for agent1
    wins_as_p1  = wins_as_p2  = 0
    games_as_p1 = games_as_p2 = 0

    for i in range(num_games):
        board = create_board()
        if i % 2 == 0:
            # agent1 goes first as PLAYER_X
            agents = [(agent1, PLAYER_X), (agent2, PLAYER_O)]
            a1_piece= PLAYER_X
            a1_role = "p1"
        else:
            # agent1 goes second as PLAYER_O
            agents = [(agent2, PLAYER_X), (agent1, PLAYER_O)]
            a1_piece = PLAYER_O
            a1_role = "p2"

        turn = 0
        while not is_terminal(board):
            ag, pid = agents[turn % 2]
            move = ag.choose_move(board)
            if move:
                make_move(board, *move, pid)
            turn += 1

        winner = check_winner(board)
        if winner == a1_piece:   
            wins += 1
        elif winner is None:     
            draws += 1
        else:                    
            losses += 1

        # P1/P2 breakdown
        if a1_role == "p1":
            games_as_p1 += 1
            if winner == a1_piece: 
                wins_as_p1 += 1
        else:
            games_as_p2 += 1
            if winner == a1_piece: 
                wins_as_p2 += 1

    total  = num_games
    result = {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win": wins/total * 100,
        "draw": draws/total * 100,
        "loss" : losses/total * 100,
        # how often agent1 won when going first vs second
        "wins_as_p1": wins_as_p1,
        "wins_as_p2": wins_as_p2,
        "games_as_p1": games_as_p1,
        "games_as_p2": games_as_p2,
        "win_pct_as_p1": wins_as_p1 / games_as_p1 * 100 if games_as_p1 else 0,
        "win_pct_as_p2": wins_as_p2 / games_as_p2 * 100 if games_as_p2 else 0,
    }
    if label:
        print(f"  {label:35s}  "
              f"W={result['win']:5.1f}%  "
              f"D={result['draw']:5.1f}%  "
              f"L={result['loss']:5.1f}%  "
              f"| as P1={result['win_pct_as_p1']:5.1f}%  "
              f"as P2={result['win_pct_as_p2']:5.1f}%")
    return result


# ----------------------------------------------------------------------
# 1. Each algorithm vs Default Opponent
# ----------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"  Tic Tac Toe: Algorithms vs Default Opponent  (n={NUM_GAMES})")
print(f"{'='*60}")

vs_default = {}
for name, agent in agents:
    vs_default[name] = play_games(agent, default, label=f"{name} vs Default")


# ----------------------------------------------------------------------
# 2. Algorithm vs Algorithm 
# ----------------------------------------------------------------------

print(f"\n{'-'*60}")
print(f"  Tic Tac Toe: Algorithm vs Algorithm  (n={NUM_GAMES})")
print(f"{'-'*60}")

aVa = {}
for i in range(len(agents)):
    for j in range(i + 1, len(agents)):
        n1, a1 = agents[i]
        n2, a2 = agents[j]
        label  = f"{n1} vs {n2}"
        r = play_games(a1, a2, label=label)
        aVa[label] = {"name1": n1, "name2": n2, **r}


# ----------------------------------------------------------------------
# Save to csv
# ----------------------------------------------------------------------

# CSV 1: vs Default Opponent
with open(f"{OUTPUT_DIR}ttt_vs_default.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "Win%", "Draw%", "Loss%", "WinAsP1%", "WinAsP2%"])
    for n in algo_names:
        r = vs_default[n]
        writer.writerow([n, f"{r['win']:.1f}", f"{r['draw']:.1f}",
                         f"{r['loss']:.1f}", f"{r['win_pct_as_p1']:.1f}",
                         f"{r['win_pct_as_p2']:.1f}"])
print(f"Saved to {OUTPUT_DIR}ttt_vs_default.csv")

# CSV 2: Head to Head
with open(f"{OUTPUT_DIR}ttt_aVa.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Matchup", "Agent1", "Agent1Win%", "Draw%",
                     "Agent2Win%", "WinAsP1%", "WinAsP2%"])
    for label, r in aVa.items():
        writer.writerow([label, r["name1"], f"{r['win']:.1f}", f"{r['draw']:.1f}",
                         f"{r['loss']:.1f}", f"{r['win_pct_as_p1']:.1f}",
                         f"{r['win_pct_as_p2']:.1f}"])
print(f"Saved to {OUTPUT_DIR}ttt_aVa.csv")