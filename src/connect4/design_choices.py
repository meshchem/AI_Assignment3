import os
import csv
import random
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from connect4 import (
    Connect4Game,
    QLearningAgent, DQNAgent,
    RandomAgent,
    PLAYER1_PIECE, AI_PIECE
)

os.makedirs("results/experiments/", exist_ok=True)

EPISODES = 200_000
WINDOW   = 1_000   # rolling window for training curve
EVAL_GAMES = 200   # games to evaluate each trained agent


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def moving_average(history, window=WINDOW):
    rates = []
    for i in range(len(history)):
        start = max(0, i - window + 1)
        chunk = history[start: i + 1]
        rates.append(chunk.count(1) / len(chunk))
    return rates


# Play num_games against a random opponent, return win %.
def evaluate(agent, num_games=EVAL_GAMES):
    wins = 0
    opp  = RandomAgent()
    for i in range(num_games):
        game = Connect4Game()
        if i % 2 == 0:
            players  = {PLAYER1_PIECE: agent, AI_PIECE: opp}
            a_piece  = PLAYER1_PIECE
        else:
            players  = {PLAYER1_PIECE: opp, AI_PIECE: agent}
            a_piece  = AI_PIECE

        while not game.is_terminal():
            col = players[game.current_player].choose_move(game)
            if col is not None:
                game.make_move(col)

        if game.winner == a_piece:
            wins += 1

    return wins / num_games * 100

# Plot multiple training curves on the same axes.
def plot_curves(curves, labels, colours, title, filename, window=WINDOW):
    fig, ax = plt.subplots(figsize=(10, 5))
    for rates, label, colour in zip(curves, labels, colours):
        ax.plot(range(1, len(rates) + 1), rates,
                color=colour, linewidth=1.2, label=label)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9,
               label="50% baseline")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Win rate (rolling {window} eps)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/experiments/{filename}", dpi=150)
    plt.close()
    print(f"  Saved results/experiments/{filename}")


# ----------------------------------------------------------------------
# Summary table — collects all results for CSV
# ----------------------------------------------------------------------

summary = []   # list of dicts


# ----------------------------------------------------------------------
# Q-Learning alpha comparison
# ----------------------------------------------------------------------

# print(f"\n{'-'*60}")
# print(f" Q-Learning Alpha Comparison")
# print(f"{'-'*60}")

# alphas  = [0.1, 0.2, 0.5]
# colours = ["blue", "orange", "green"]
# curves  = []
# labels  = []

# for alpha in alphas:
#     print(f"  Training alpha={alpha}")
#     agent = QLearningAgent(piece=AI_PIECE, opp_piece=PLAYER1_PIECE,
#                            alpha=alpha)
#     agent.train(EPISODES, RandomAgent())
#     win_pct = evaluate(agent)
#     rates   = moving_average(agent.win_history)
#     curves.append(rates)
#     labels.append(f"α={alpha}  (eval {win_pct:.1f}%)")
#     print(f"    Final eval win %: {win_pct:.1f}%")
#     summary.append({"experiment": "QL Alpha", "variant": f"alpha={alpha}",
#                     "eval_win_pct": f"{win_pct:.1f}"})

# plot_curves(curves, labels, colours,
#             title=f"Connect 4: Q-Learning Alpha Comparison (n={EPISODES})",
#             filename="c4_alpha_comparison.png")


# # ----------------------------------------------------------------------
# # Q-Learning gamma comparison
# # ----------------------------------------------------------------------

# print(f"\n{'-'*60}")
# print(f" Q-Learning Gamma Comparison")
# print(f"{'-'*60}")

# gammas  = [0.9, 0.95, 0.99]
# colours = ["blue", "orange", "green"]
# curves  = []
# labels  = []

# for gamma in gammas:
#     print(f"  Training gamma={gamma}")
#     agent = QLearningAgent(piece=AI_PIECE, opp_piece=PLAYER1_PIECE,
#                            gamma=gamma)
#     agent.train(EPISODES, RandomAgent())
#     win_pct = evaluate(agent)
#     rates   = moving_average(agent.win_history)
#     curves.append(rates)
#     labels.append(f"γ={gamma}  (eval {win_pct:.1f}%)")
#     print(f"    Final eval win %: {win_pct:.1f}%")
#     summary.append({"experiment": "QL Gamma", "variant": f"gamma={gamma}",
#                     "eval_win_pct": f"{win_pct:.1f}"})

# plot_curves(curves, labels, colours,
#             title=f"Connect 4: Q-Learning Gamma Comparison (n={EPISODES})",
#             filename="c4_gamma_comparison.png")


# # ----------------------------------------------------------------------
# # DQN target update frequency
# # ----------------------------------------------------------------------

# print(f"\n{'-'*60}")
# print(f" DQN Target Update Frequency")
# print(f"{'-'*60}")

# target_updates = [200, 500, 1000]
# colours        = ["blue", "orange", "green"]
# curves         = []
# labels         = []

# for tu in target_updates:
#     print(f"  Training target_update={tu}")
#     agent = DQNAgent(piece=AI_PIECE, opp_piece=PLAYER1_PIECE)
#     agent.TARGET_UPDATE = tu
#     agent.train(EPISODES, RandomAgent())
#     win_pct = evaluate(agent)
#     rates   = moving_average(agent.win_history)
#     curves.append(rates)
#     labels.append(f"target_update={tu}  (eval {win_pct:.1f}%)")
#     print(f"    Final eval win %: {win_pct:.1f}%")
#     summary.append({"experiment": "DQN Target Update", "variant": f"target_update={tu}",
#                     "eval_win_pct": f"{win_pct:.1f}"})

# plot_curves(curves, labels, colours,
#             title=f"Connect 4: DQN Target Update Frequency (n={EPISODES})",
#             filename="c4_dqn_target_update.png")


# # ----------------------------------------------------------------------
# # Experiment 4: DQN training curve (episode count)
# # Shows how performance evolves over 200,000 episodes
# # ----------------------------------------------------------------------

# print(f"\n{'-'*60}")
# print(f" DQN Training Curve")
# print(f"{'-'*60}")

# print(f"  Training DQN for {EPISODES} episodes")
# agent = DQNAgent(piece=AI_PIECE, opp_piece=PLAYER1_PIECE)
# agent.train(EPISODES, RandomAgent())
# win_pct = evaluate(agent)
# print(f"  Final eval win %: {win_pct:.1f}%")

# rates      = moving_average(agent.win_history)
# episodes   = list(range(1, len(rates) + 1))
# final      = rates[-1]

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(episodes, rates, color="orange", linewidth=1.2,
#         label=f"Win rate (rolling {WINDOW} eps)")
# ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9,
#            label="50% baseline")
# offset = 15 if final < 0.85 else -25
# ax.annotate(f"Final: {final*100:.1f}%",
#             xy=(episodes[-1], final),
#             xytext=(-110, offset), textcoords="offset points",
#             arrowprops=dict(arrowstyle="->", color="black"),
#             fontsize=11)
# ax.set_xlabel("Episode", fontsize=12)
# ax.set_ylabel(f"Win rate (rolling {WINDOW} eps)", fontsize=12)
# ax.set_title(f"Connect 4: DQN Training Curve (n={EPISODES})", fontsize=13)
# ax.set_ylim(0, 1.05)
# ax.legend(fontsize=11)
# ax.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("results/experiments/c4_dqn_training_curve.png", dpi=150)
# plt.close()
# print("  Saved results/experiments/c4_dqn_training_curve.png")

# summary.append({"experiment": "DQN Training Curve", "variant": f"episodes={EPISODES}",
#                 "eval_win_pct": f"{win_pct:.1f}"})


# ----------------------------------------------------------------------
# Q-Learning epsilon decay comparison
# ----------------------------------------------------------------------

decays  = [0.9998, 0.9999, 0.99995]
colours = ["blue", "orange", "green"]
curves  = []
labels  = []

for decay in decays:
    print(f"  Training eps_decay={decay}")
    agent = QLearningAgent(piece=AI_PIECE, opp_piece=PLAYER1_PIECE,
                           eps_decay=decay)
    agent.train(EPISODES, RandomAgent())
    win_pct = evaluate(agent)
    rates   = moving_average(agent.win_history)
    curves.append(rates)
    labels.append(f"decay={decay}  (eval {win_pct:.1f}%)")
    print(f"    Final eval win %: {win_pct:.1f}%")
    summary.append({"experiment": "QL Epsilon Decay", "variant": f"decay={decay}",
                    "eval_win_pct": f"{win_pct:.1f}"})

plot_curves(curves, labels, colours,
            title=f"Connect 4: Q-Learning Epsilon Decay Comparison (n={EPISODES})",
            filename="c4_epsilon_decay_comparison.png")


# ----------------------------------------------------------------------
# Save summary CSV
# ----------------------------------------------------------------------

# with open("results/experiments/c4_experiments_summary.csv", "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=["experiment", "variant", "eval_win_pct"])
#     writer.writeheader()
#     writer.writerows(summary)
# print("\nSaved results/experiments/c4_experiments_summary.csv")

print("\nAll experiments done.")