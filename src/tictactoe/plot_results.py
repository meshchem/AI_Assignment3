import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# INPUT_DIR = "results/random_dqn/"
INPUT_DIR = "results/mixed_dqn/"
# INPUT_DIR = "results/rule_dqn/"

# OUTPUT_DIR = "results/random_dqn_fig/"
# OUTPUT_DIR = "results/mixed_dqn_fig/"
# OUTPUT_DIR = "results/rule_dqn_fig/"
OUTPUT_DIR = "results/bar_charts/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# check all CSVs exist
for f in [f"{INPUT_DIR}ttt_vs_default.csv", f"{INPUT_DIR}ttt_aVa.csv"]:
    if not os.path.exists(f):
        print(f"[ERROR] {f} not found. Run evaluate_ttt.py first.")
        exit(1)


# ----------------------------------------------------------------------
# Parse CSVs
# ----------------------------------------------------------------------

algo_names = ["Minimax", "Minimax-AB", "Q-Learning", "DQN"]

vs_default = {}
with open(f"{INPUT_DIR}ttt_vs_default.csv") as f:
    for row in csv.DictReader(f):
        vs_default[row["Algorithm"]] = {
            "win_pct"       : float(row["Win%"]),
            "draw_pct"      : float(row["Draw%"]),
            "loss_pct"      : float(row["Loss%"]),
            "win_pct_as_p1" : float(row["WinAsP1%"]),
            "win_pct_as_p2" : float(row["WinAsP2%"]),
        }

aVa = {}
with open(f"{INPUT_DIR}ttt_aVa.csv") as f:
    for row in csv.DictReader(f):
        aVa[row["Matchup"]] = {
            "name1"         : row["Agent1"],
            "win_pct"       : float(row["Agent1Win%"]),
            "draw_pct"      : float(row["Draw%"]),
            "loss_pct"      : float(row["Agent2Win%"]),
            "win_pct_as_p1" : float(row["WinAsP1%"]),
            "win_pct_as_p2" : float(row["WinAsP2%"]),
        }


print(f"Parsed: {len(vs_default)} algorithms, {len(aVa)} matchups")

WIN_COL  = "salmon"
DRAW_COL = "plum"
LOSS_COL = "indigo"
P1_COL   = "navy"
P2_COL   = "crimson"


def add_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=8)


# ----------------------------------------------------------------------
# Graph 1: vs Default Opponent
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(algo_names))
w = 0.25

add_labels(ax, ax.bar(x - w, [vs_default[n]["win_pct"]  for n in algo_names],
                      w, label="Win %",  color=WIN_COL))
add_labels(ax, ax.bar(x,     [vs_default[n]["draw_pct"] for n in algo_names],
                      w, label="Draw %", color=DRAW_COL))
add_labels(ax, ax.bar(x + w, [vs_default[n]["loss_pct"] for n in algo_names],
                      w, label="Loss %", color=LOSS_COL))

ax.set_xticks(x)
ax.set_xticklabels(algo_names, fontsize=12)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_title("Tic Tac Toe: Algorithms vs Default Opponent", fontsize=13)
ax.set_ylim(0, 115)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}ttt_vs_default.png", dpi=150)
plt.close()
print(f"  Saved {OUTPUT_DIR}ttt_vs_default.png")


# ----------------------------------------------------------------------
# Graph 2: vs Each Other
# ----------------------------------------------------------------------

labels = list(aVa.keys())
fig, ax = plt.subplots(figsize=(13, 5))
x2 = np.arange(len(labels))

add_labels(ax, ax.bar(x2 - w, [aVa[l]["win_pct"]  for l in labels],
                      w, label="Agent 1 Win %", color=WIN_COL))
add_labels(ax, ax.bar(x2,     [aVa[l]["draw_pct"] for l in labels],
                      w, label="Draw %",        color=DRAW_COL))
add_labels(ax, ax.bar(x2 + w, [aVa[l]["loss_pct"] for l in labels],
                      w, label="Agent 2 Win %", color=LOSS_COL))

ax.set_xticks(x2)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_title("Tic Tac Toe: Algorithms vs Algorithms", fontsize=13)
ax.set_ylim(0, 115)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}ttt_aVa.png", dpi=150)
plt.close()
print(f"  Saved {OUTPUT_DIR}ttt_aVa.png")

# ----------------------------------------------------------------------
# Graph 3: P1 vs P2 win rate (vs Default Opponent)
# Shows first-mover advantage — win rate when going first vs second.
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
x4 = np.arange(len(algo_names))
w2 = 0.35

add_labels(ax, ax.bar(x4 - w2/2,
                      [vs_default[n]["win_pct_as_p1"] for n in algo_names],
                      w2, label="Win % as Player 1 (goes first)", color=P1_COL))
add_labels(ax, ax.bar(x4 + w2/2,
                      [vs_default[n]["win_pct_as_p2"] for n in algo_names],
                      w2, label="Win % as Player 2 (goes second)", color=P2_COL))

ax.set_xticks(x4)
ax.set_xticklabels(algo_names, fontsize=12)
ax.set_ylabel("Win Rate (%)", fontsize=12)
ax.set_title("Tic Tac Toe: First-Mover Advantage", fontsize=13)
ax.set_ylim(0, 115)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}ttt_p1_p2.png", dpi=150)
plt.close()
print(f"Saved {OUTPUT_DIR}ttt_p1_p2.png")


# ----------------------------------------------------------------------
# Graph 4: P1 vs P2 win rate (Head-to-Head)
# Shows first-mover advantage when algorithms play each other.
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 5))
x5 = np.arange(len(labels))
w3 = 0.35

add_labels(ax, ax.bar(x5 - w3/2,
                      [aVa[l]["win_pct_as_p1"] for l in labels],
                      w3, label="Agent 1 Win % as Player 1 (goes first)", color=P1_COL))
add_labels(ax, ax.bar(x5 + w3/2,
                      [aVa[l]["win_pct_as_p2"] for l in labels],
                      w3, label="Agent 1 Win % as Player 2 (goes second)", color=P2_COL))

ax.set_xticks(x5)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
ax.set_ylabel("Win Rate (%)", fontsize=12)
ax.set_title("Tic Tac Toe: First-Mover Advantage (Head-to-Head)", fontsize=13)
ax.set_ylim(0, 115)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}ttt_aVa_p1_p2.png", dpi=150)
plt.close()
print(f"Saved {OUTPUT_DIR}ttt_aVa_p1_p2.png")

print("\nAll graphs saved in ./results/")