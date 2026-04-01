import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOTAL_STATES = 5478   # known reachable states in Tic Tac Toe
QL_PATH      = "results/qlearn_ttt.pkl"

with open(QL_PATH, "rb") as f:
    data = pickle.load(f)

coverage = data.get("coverage", [])
episodes, sizes = zip(*coverage)

fig, ax1 = plt.subplots(figsize=(10, 5))

# left axis: raw state count
ax1.plot(episodes, sizes, color="blue", linewidth=1.5,
         label="States visited")
ax1.axhline(TOTAL_STATES, color="red", linestyle="--", linewidth=1,
            label=f"Total reachable states ({TOTAL_STATES:,})")
ax1.set_xlabel("Episode", fontsize=12)
ax1.set_ylabel("Unique states in Q-table", fontsize=12, color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# right axis: percentage coverage
ax2 = ax1.twinx()
ax2.plot(episodes, [s / TOTAL_STATES * 100 for s in sizes],
         color="blue", linewidth=1.5, alpha=0)   # invisible, just for axis scale
ax2.set_ylabel("Coverage (%)", fontsize=12, color="blue")
ax2.tick_params(axis="y", labelcolor="blue")

# mark where exploration effectively stops (eps ~= eps_min)
# eps decays by 0.9995 per episode from 1.0 to 0.05
# solve: 1.0 * 0.9995^n = 0.05  =>  n = log(0.05)/log(0.9995) ~ 5990
ax1.axvline(5990, color="grey", linestyle=":", linewidth=1,
            label="Exploration ends (~ep 6,000)")

ax1.set_title("Tic Tac Toe: Q-Table State Coverage Over Training", fontsize=13)
ax1.legend(loc="center right", fontsize=10)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/training/ttt_qtable_coverage.png", dpi=150)
plt.close()
print("Saved results/training/ttt_qtable_coverage.png")