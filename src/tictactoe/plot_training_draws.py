import os
import pickle
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/training_fig/wins_draws", exist_ok=True)

WINDOW = 500

def moving_average_wins(history, window=WINDOW):
    rates = []
    for i in range(len(history)):
        start = max(0, i - window + 1)
        chunk = history[start: i + 1]
        rates.append(chunk.count(1) / len(chunk))
    return rates

def moving_average_draws(history, window=WINDOW):
    rates = []
    for i in range(len(history)):
        start = max(0, i - window + 1)
        chunk = history[start: i + 1]
        rates.append(chunk.count(0) / len(chunk))
    return rates

def plot_curve(rates, draw_rates, title, filename, colour):
    episodes = list(range(1, len(rates) + 1))
    fig, ax  = plt.subplots(figsize=(10, 5))

    ax.plot(episodes, rates, color=colour, linewidth=1.2,
            label=f"Win rate (rolling {WINDOW} eps)")
    ax.plot(episodes, draw_rates, color=colour, linewidth=1.2,
            linestyle="--", alpha=0.6,
            label=f"Draw rate (rolling {WINDOW} eps)")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9,
               label="50% baseline")

    final      = rates[-1]
    final_draw = draw_rates[-1]
    offset     = 15 if final < 0.85 else -25
    ax.annotate(f"Final win: {final*100:.1f}%",
                xy=(episodes[-1], final),
                xytext=(-110, offset), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=11)
    ax.annotate(f"Final draw: {final_draw*100:.1f}%",
                xy=(episodes[-1], final_draw),
                xytext=(-120, -offset), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=11)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Rate (rolling {WINDOW} eps)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/training_fig/wins_draws{filename}", dpi=150)
    plt.close()
    print(f"  Saved results/training_fig/wins_draws{filename}")


# ----------------------------------------------------------------------
# Q-Learning
# ----------------------------------------------------------------------

ql_history = None
QL_PATH    = "results/qlearn_mixed_ttt.pt"

if not os.path.exists(QL_PATH):
    print(f"{QL_PATH} not found, run train.py --agent qlearn first")
else:
    with open(QL_PATH, "rb") as f:
        data = pickle.load(f)
    ql_history = data.get("history", [])
    print(f"Tabular Q-Learning: {len(ql_history)} episodes loaded")

    rates      = moving_average_wins(ql_history)
    draw_rates = moving_average_draws(ql_history)
    plot_curve(rates, draw_rates,
               title=f"Tic Tac Toe: Tabular Q-Learning Training Curve (n={len(ql_history)})",
               filename="ttt_qlearn_mixed_training.png",
               colour="blue")


# ----------------------------------------------------------------------
# DQN
# ----------------------------------------------------------------------

dqn_history = None
DQN_PATH    = "results/dqn_mixed_ttt.pkl"

if not os.path.exists(DQN_PATH):
    print(f"{DQN_PATH} not found, run train.py --agent dqn first")
else:
    ck = torch.load(DQN_PATH, map_location="cpu")
    dqn_history = ck.get("history", [])
    print(f"DQN: {len(dqn_history)} episodes loaded")

    rates      = moving_average_wins(dqn_history)
    draw_rates = moving_average_draws(dqn_history)
    plot_curve(rates, draw_rates,
               title=f"Tic Tac Toe: DQN Training Curve (n={len(dqn_history)})",
               filename="ttt_dqn_mixed_training.png",
               colour="orange")


# ----------------------------------------------------------------------
# Combined
# ----------------------------------------------------------------------

if ql_history and dqn_history:
    ql_wins   = moving_average_wins(ql_history)
    ql_draws  = moving_average_draws(ql_history)
    dqn_wins  = moving_average_wins(dqn_history)
    dqn_draws = moving_average_draws(dqn_history)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(ql_wins)  + 1), ql_wins,
            color="blue",   linewidth=1.2, label="Q-Learning win rate")
    ax.plot(range(1, len(ql_draws) + 1), ql_draws,
            color="blue",   linewidth=1.2, linestyle="--", alpha=0.6, label="Q-Learning draw rate")
    ax.plot(range(1, len(dqn_wins)  + 1), dqn_wins,
            color="orange", linewidth=1.2, label="DQN win rate")
    ax.plot(range(1, len(dqn_draws) + 1), dqn_draws,
            color="orange", linewidth=1.2, linestyle="--", alpha=0.6, label="DQN draw rate")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9, label="50% baseline")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Rate (rolling {WINDOW} eps)", fontsize=12)
    ax.set_title("Tic Tac Toe: Q-Learning vs DQN Training Curves", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/training_draws/ttt_training_comparison.png", dpi=150)
    plt.close()
    print("  Saved results/training_draws/ttt_training_comparison.png")

print("\nDone.")