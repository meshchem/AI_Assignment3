import argparse
import os
import csv
import time

from connect4 import (
    Connect4Game,
    QLearningAgent, DQNAgent, MinimaxAgent,
    DefaultOpponent, RandomAgent,
    PLAYER1_PIECE, PLAYER2_PIECE
)

os.makedirs("results/csv", exist_ok=True)

parser = argparse.ArgumentParser(description="Evaluate Connect 4 agents")
parser.add_argument("--games", type=int, default=500,
                    help="Number of games per matchup (default 500)")
args = parser.parse_args()

NUM_GAMES = args.games


# ----------------------------------------------------------------------
# Two agents playing against each other.
# Returns 
#   win/draw/loss stats 
#   avg_game_length   : average number of moves per game
#   avg_move_time_ms  : average time in ms per move for agent 1
#   win_rate_variance : variance of win outcome (1/0) across all games,
#                       measures consistency of the agent
# ----------------------------------------------------------------------

def play_games(agent1_p1, agent1_p2, agent2_p1, agent2_p2, num_games=NUM_GAMES, label=""):
    wins = draws = losses = 0
    wins_as_p1 = wins_as_p2 = 0
    games_as_p1 = games_as_p2 = 0

    total_moves      = 0
    total_a1_time_ms = 0
    total_a1_moves   = 0
    win_outcomes     = []   # 1 if agent1 wins, 0 otherwise, used for variance

    for i in range(num_games):
        game = Connect4Game()
        if i % 2 == 0:
            # agent1 goes first as PLAYER1_PIECE
            agents   = {PLAYER1_PIECE: agent1_p1, PLAYER2_PIECE: agent2_p2}
            a1_piece = PLAYER1_PIECE
            a1_role  = "p1"
        else:
            # agent1 goes second as PLAYER2_PIECE
            agents   = {PLAYER1_PIECE: agent2_p1, PLAYER2_PIECE: agent1_p2}
            a1_piece = PLAYER2_PIECE
            a1_role  = "p2"

        move_count = 0

        while not game.is_terminal():
            current = game.current_player
            agent   = agents[current]

            if current == a1_piece:
                t0  = time.perf_counter()
                col = agent.choose_move(game)
                total_a1_time_ms += (time.perf_counter() - t0) * 1000
                total_a1_moves   += 1
            else:
                col = agent.choose_move(game)

            if col is not None:
                game.make_move(col)
                move_count += 1

        total_moves += move_count
        print(f"  Game {i}: winner={game.winner}, a1_piece={a1_piece}, a1_role={a1_role}, moves={move_count}")

        if game.winner == a1_piece:
            wins += 1
            win_outcomes.append(1)
        elif game.winner is None:
            draws += 1
            win_outcomes.append(0)
        else:
            losses += 1
            win_outcomes.append(0)

        if a1_role == "p1":
            games_as_p1 += 1
            if game.winner == a1_piece: wins_as_p1 += 1
        else:
            games_as_p2 += 1
            if game.winner == a1_piece: wins_as_p2 += 1

    total = num_games

    # Win rate variance: measures how consistent the agent is across games.
    # A low variance means the agent wins or loses predictably.
    # A high variance means unpredictable performance.
    mean_outcome = sum(win_outcomes) / total
    variance     = sum((x - mean_outcome) ** 2 for x in win_outcomes) / total

    avg_game_length  = total_moves / total
    avg_move_time_ms = total_a1_time_ms / total_a1_moves if total_a1_moves > 0 else 0.0

    result = {
        "wins"             : wins,
        "draws"            : draws,
        "losses"           : losses,
        "win_pct"          : wins   / total * 100,
        "draw_pct"         : draws  / total * 100,
        "loss_pct"         : losses / total * 100,
        "win_pct_as_p1"    : wins_as_p1 / games_as_p1 * 100 if games_as_p1 else 0,
        "win_pct_as_p2"    : wins_as_p2 / games_as_p2 * 100 if games_as_p2 else 0,
        "avg_game_length"  : avg_game_length,
        "avg_move_time_ms" : avg_move_time_ms,
        "win_rate_variance": variance,
    }

    if label:
        print(f"  {label:35s}  "
              f"W={result['win_pct']:5.1f}%  "
              f"D={result['draw_pct']:5.1f}%  "
              f"L={result['loss_pct']:5.1f}%  "
              f"| as P1={result['win_pct_as_p1']:5.1f}%  "
              f"as P2={result['win_pct_as_p2']:5.1f}%  "
              f"| AvgLen={result['avg_game_length']:.1f}  "
              f"MoveTime={result['avg_move_time_ms']:.2f}ms  "
              f"Var={result['win_rate_variance']:.4f}")
    return result


# ----------------------------------------------------------------------
# Build agents, two versions of each, one per side
# ----------------------------------------------------------------------

def make_agents(piece, opp):
    mm    = MinimaxAgent(piece=piece, depth=5, use_alpha_beta=False)
    mm_ab = MinimaxAgent(piece=piece, depth=5, use_alpha_beta=True)
    ql    = QLearningAgent(piece=piece, opp_piece=opp)
    ql.load("results/qlearn_c4.pkl")
    dqn_a = DQNAgent(piece=piece, opp_piece=opp)
    dqn_a.load("results/dqn_c4.pt")
    return mm, mm_ab, ql, dqn_a

mm_p1,  mm_ab_p1, ql_p1,  dqn_p1 = make_agents(PLAYER1_PIECE, PLAYER2_PIECE)
mm_p2,  mm_ab_p2, ql_p2,  dqn_p2 = make_agents(PLAYER2_PIECE, PLAYER1_PIECE)

default_p1 = DefaultOpponent(player=PLAYER1_PIECE)
default_p2 = DefaultOpponent(player=PLAYER2_PIECE)

agents = [
    ("Minimax",    mm_p1,    mm_p2),
    ("Minimax-AB", mm_ab_p1, mm_ab_p2),
    ("Q-Learning", ql_p1,    ql_p2),
    ("DQN",        dqn_p1,   dqn_p2),
]

algo_names = [n for n, _, _ in agents]
results    = {}

print("Agents loaded")


# ----------------------------------------------------------------------
# Each algorithm vs Default Opponent
# ----------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"Connect 4: Each Algorithm vs Default Opponent  (n={NUM_GAMES})")
print(f"{'='*60}")

vs_default = {}
for name, a_p1, a_p2 in agents:
    r = play_games(a_p1, a_p2, default_p1, default_p2,
                   label=f"{name} vs Default")
    vs_default[name] = r

results["vs_default"] = vs_default


# ----------------------------------------------------------------------
# Algorithm vs Algorithm (head to head)
# ----------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"Connect 4: Algorithm vs Algorithm  (n={NUM_GAMES})")
print(f"{'='*60}")

aVa = {}
for i in range(len(agents)):
    for j in range(i + 1, len(agents)):
        n1, a1_p1, a1_p2 = agents[i]
        n2, a2_p1, a2_p2 = agents[j]
        label = f"{n1} vs {n2}"
        r = play_games(a1_p1, a1_p2, a2_p1, a2_p2, label=label)
        aVa[label] = {"name1": n1, "name2": n2, **r}

results["aVa"] = aVa

# ----------------------------------------------------------------------
# Save to csv
# ----------------------------------------------------------------------

# CSV 1: vs Default Opponent
with open("results/csv/c4_vs_default.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Algorithm", "Win%", "Draw%", "Loss%",
        "WinAsP1%", "WinAsP2%",
        "AvgGameLength", "AvgMoveTimeMs", "WinRateVariance"
    ])
    for n in algo_names:
        r = vs_default[n]
        writer.writerow([
            n,
            f"{r['win_pct']:.1f}",
            f"{r['draw_pct']:.1f}",
            f"{r['loss_pct']:.1f}",
            f"{r['win_pct_as_p1']:.1f}",
            f"{r['win_pct_as_p2']:.1f}",
            f"{r['avg_game_length']:.2f}",
            f"{r['avg_move_time_ms']:.3f}",
            f"{r['win_rate_variance']:.4f}",
        ])

# CSV 2: Algorithm vs Algorithm
with open("results/csv/c4_aVa.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Matchup", "Agent1", "Agent1Win%", "Draw%", "Agent2Win%",
        "WinAsP1%", "WinAsP2%",
        "AvgGameLength", "AvgMoveTimeMs", "WinRateVariance"
    ])
    for label, r in aVa.items():
        writer.writerow([
            label,
            r["name1"],
            f"{r['win_pct']:.1f}",
            f"{r['draw_pct']:.1f}",
            f"{r['loss_pct']:.1f}",
            f"{r['win_pct_as_p1']:.1f}",
            f"{r['win_pct_as_p2']:.1f}",
            f"{r['avg_game_length']:.2f}",
            f"{r['avg_move_time_ms']:.3f}",
            f"{r['win_rate_variance']:.4f}",
        ])

print("Saved results to results/csv/")