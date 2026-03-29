import os
import pickle
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

WINDOW = 500   # caclculates average win rate over this many episodes

# Convert per-episode outcomes (1=win, 0=draw, -1=loss) into a moving average of the win rate.
def moving_average(history, window=WINDOW):
    rates = []
    for i in range(len(history)):
        start = max(0, i - window + 1)
        chunk = history[start: i + 1]
        rates.append(chunk.count(1) / len(chunk))
    return rates


def plot_curve(rates, title, filename, colour):
    episodes = list(range(1, len(rates) + 1))
    fig, ax  = plt.subplots(figsize=(10, 5))

    ax.plot(episodes, rates, color=colour, linewidth=1.2,
            label=f"Win rate (rolling {WINDOW} eps)")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.9,
               label="50% baseline")

    # Annotate final win rate
    final = rates[-1]
    offset = 15 if final < 0.85 else -25
    ax.annotate(f"Final: {final*100:.1f}%",
                xy=(episodes[-1], final),
                xytext=(-90, offset), textcoords="offset points",
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
    print(f"  Saved results/{filename}")


# ----------------------------------------------------------------------
# Tabular Q-Learning
# ----------------------------------------------------------------------

ql_history  = None
QL_PATH     = "results/qlearn_ttt.pkl"

if not os.path.exists(QL_PATH):
    print(f"{QL_PATH} not found, run train.py --agent qlearn first")
else:
    with open(QL_PATH, "rb") as f:
        data = pickle.load(f)
    ql_history = data.get("history", [])
    print(f"Tabular Q-Learning {len(ql_history)} episodes loaded")

    rates = moving_average(ql_history)
    plot_curve(rates,
               title=f"Tic Tac Toe: Tabular Q-Learning Training Curve (n={len(ql_history)})",
               filename="ttt_qlearn_training.png",
               colour="blue")


# ----------------------------------------------------------------------
# DQN
# ----------------------------------------------------------------------

dqn_history = None
DQN_PATH    = "results/dqn_rule_ttt.pt"

if not os.path.exists(DQN_PATH):
    print(f"{DQN_PATH} not found, run train.py --agent dqn first")
else:
    ck = torch.load(DQN_PATH, map_location="cpu")
    dqn_history = ck.get("history", [])
    print(f"DQN {len(dqn_history)} episodes loaded")

    rates = moving_average(dqn_history)
    plot_curve(rates,
               title=f"Tic Tac Toe: DQN Training Curve (n={len(dqn_history)})",
               filename="ttt_dqn_rule_training.png",
               colour="orange")


# ----------------------------------------------------------------------
# Combined comparison
# ----------------------------------------------------------------------

if ql_history and dqn_history:
    ql_rates  = moving_average(ql_history)
    dqn_rates = moving_average(dqn_history)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(ql_rates)  + 1), ql_rates,
            color="blue", linewidth=1.2, label="Tabular Q-Learning")
    ax.plot(range(1, len(dqn_rates) + 1), dqn_rates,
            color="orange", linewidth=1.2, label="DQN")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.9,
               label="50% baseline")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Win rate (rolling {WINDOW} eps)", fontsize=12)
    ax.set_title("Tic Tac Toe: Tabular Q-Learning vs DQN Training Curves", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/ttt_training_comparison.png", dpi=150)
    plt.close()
    print("  Saved results/ttt_training_comparison.png")

print("\nDone.")