# --------------------------------------------------------------------------------
# plot_training.py
# Plot Connect 4 RL training curves
#
# Run AFTER train.py:
#   python plot_training.py
#
# Outputs saved to ./results/
#   c4_qlearn_training.png
#   c4_dqn_training.png
# --------------------------------------------------------------------------------

import os
import pickle
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/training", exist_ok=True)

WINDOW = 500   # rolling average window size


#  Convert a list of episode outcomes (1=win, 0=draw, -1=loss) into a smoothed rolling win rate.
def rolling_win_rate(history, window=WINDOW):
    rates = []
    for i in range(len(history)):
        start = max(0, i - window + 1)
        chunk = history[start: i + 1]
        rates.append(chunk.count(1) / len(chunk))
    return rates


# Plot a training curve with final win rate annotation and 50% baseline.
def plot_curve(rates, title, filename, colour):
    episodes = list(range(1, len(rates) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rates, color=colour, linewidth=1.2,
            label=f"Win rate (rolling {WINDOW} eps)")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.9,
               label="50% baseline")

    final = rates[-1]
    ax.annotate(f"Final: {final*100:.1f}%",
                xy=(episodes[-1], final),
                xytext=(-90, 15 if final < 0.85 else -25),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=11)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Win rate (rolling {WINDOW} eps)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()
    print(f"Saved results/{filename}")

# Plot Q-table coverage history: number of unique states stored over training.
def plot_coverage(coverage, eps_decay=0.99995, eps_start=1.0, eps_min=0.05):
    if not coverage:
        print("No coverage history found")
        return

    episodes, sizes = zip(*coverage)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episodes, sizes, color="blue", linewidth=1.5,
            label="States visited")

    # exploration-stop estimate
    import math
    end_ep = math.log(eps_min / eps_start) / math.log(eps_decay)
    ax.axvline(end_ep, color="grey", linestyle=":", linewidth=1,
               label=f"Exploration ends (~ep {int(end_ep):,})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Unique states in Q-table", fontsize=12)
    ax.set_title("Connect 4: Q-Table State Coverage Over Training", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training/c4_qtable_coverage.png", dpi=150)
    plt.close()
    print("Saved results/training/c4_qtable_coverage.png")


# --------------------------------------------------------------------------------
# Tabular Q-Learning
# --------------------------------------------------------------------------------
    
ql_history = None
QL_PATH = "results/qlearn_c4.pkl"

if not os.path.exists(QL_PATH):
    print(f"{QL_PATH} not found. Run train.py --agent qlearn first")
else:
    with open(QL_PATH, "rb") as f:
        data = pickle.load(f)
    ql_history = data.get("history", [])
    print(f"Q-Learning: {len(ql_history)} episodes loaded")

    rates = rolling_win_rate(ql_history)
    coverage   = data.get("coverage", [])
    plot_curve(rates,
               title=f"Connect 4: Q-Learning Training Curve (n={len(ql_history)})",
               filename="training/c4_qlearn_training.png",
               colour="pink")
    plot_coverage(coverage)
    





# --------------------------------------------------------------------------------
# DQN
# --------------------------------------------------------------------------------

dqn_history = None
DQN_PATH = "results/dqn_c4.pt"

if not os.path.exists(DQN_PATH):
    print(f"{DQN_PATH} not found. Run train.py --agent dqn first")
else:
    ck = torch.load(DQN_PATH, map_location="cpu")
    dqn_history = ck.get("history", [])
    print(f"DQN: {len(dqn_history)} episodes loaded")

    rates = rolling_win_rate(dqn_history)
    plot_curve(rates,
               title=f"Connect 4: DQN Training Curve  (n={len(dqn_history)})",
               filename="training/c4_dqn_training.png",
               colour="green")



print("\nDone.")