========================================================================
CS7IS2 Artificial Intelligence  Assignment 3
Connect 4
========================================================================

------------------------------------------------------------------------
REQUIREMENTS
------------------------------------------------------------------------

Python 3.9 or higher is required.

Install all dependencies with:

pip install numpy pygame torch matplotlib


------------------------------------------------------------------------
PROJECT FILES
------------------------------------------------------------------------

    connect4.py         Game logic, all agents, Pygame UI, entry point
    evaluate.py         Evaluation across all algorithm matchups
    design_choices.py   Hyperparameter experiments for Q-Learning and DQN
    scalability.py      Minimax scalability experiments (full depth vs depth 5)
    plot_results.py     Generate bar charts from evaluation CSV results
    plot_training.py    Generate training curve plots from saved model files


------------------------------------------------------------------------
RESULTS DIRECTORY
------------------------------------------------------------------------

All output files are written to a results/ folder which is created
automatically on first run. The expected structure is:

    results/
        qlearn_c4.pkl               Saved Q-Learning model
        dqn_c4.pt                   Saved DQN model
        csv/
            c4_vs_default.csv       Evaluation results vs default opponent
            c4_aVa.csv              Head-to-head evaluation results
        c4_vs_default.png
        c4_aVa.png
        c4_overall.png
        c4_p1_p2.png
        c4_aVa_p1_p2.png
        c4_avg_game_length.png
        c4_avg_move_time.png
        c4_win_rate_variance.png
        training/
            c4_qlearn_training.png
            c4_dqn_training.png
            c4_training_comparison.png
        experiments/
            c4_epsilon_decay_comparison.png


------------------------------------------------------------------------
STEP 1  TRAIN THE RL AGENTS
------------------------------------------------------------------------

Train Q-Learning (1,000,000 episodes, saves to results/qlearn_c4.pkl):

    python connect4.py --train qlearn --episodes 1000000

Train DQN (300,000 episodes, saves to results/dqn_c4.pt):

    python connect4.py --train dqn --episodes 300000

Both commands exit automatically once training is complete.


------------------------------------------------------------------------
STEP 2  PLAY THE GAME (Pygame UI)
------------------------------------------------------------------------

Play against an algorithm as the human player (Red goes first):

    python connect4.py --agent minimax_ab
    python connect4.py --agent minimax
    python connect4.py --agent qlearn
    python connect4.py --agent dqn
    python connect4.py --agent default
    python connect4.py --agent random

Watch two algorithms play against each other (watch mode):

    python connect4.py --agent minimax_ab --watch default
    python connect4.py --agent minimax_ab --watch minimax
    python connect4.py --agent qlearn     --watch dqn
    python connect4.py --agent dqn        --watch random

Optional flags:

    --depth N       Set Minimax search depth (default 5)
    --load PATH     Load a model from a custom file path


------------------------------------------------------------------------
STEP 3  EVALUATE ALL ALGORITHMS
------------------------------------------------------------------------

Run headless evaluation across all matchups (500 games each by default):

    python evaluate.py

Use a custom game count:

    python evaluate.py --games 200

This produces three CSV files in results/csv/ covering:
    each algorithm vs the default opponent
    all head-to-head algorithm matchups
    an overall summary

Metrics recorded per matchup:
    win / draw / loss percentage
    win percentage as Player 1 and Player 2 separately
    average game length (number of moves)
    average move time in milliseconds for the primary agent
    win rate variance (measures consistency across games)


------------------------------------------------------------------------
STEP 4  PLOT EVALUATION RESULTS
------------------------------------------------------------------------

Generate all bar charts from the evaluation CSVs:

    python plot_results.py

Must be run after evaluate.py. Produces 8 PNG files in results/.


------------------------------------------------------------------------
STEP 5  PLOT TRAINING CURVES
------------------------------------------------------------------------

Generate training curve plots from saved model files:

    python plot_training.py

Must be run after training both RL agents. Produces 3 PNG files
in results/training/.


------------------------------------------------------------------------
STEP 6  SCALABILITY EXPERIMENTS
------------------------------------------------------------------------

Run all scalability experiments (30 minute limit for full depth):

    python scalability.py --experiment all

Run a specific experiment:

    python scalability.py --experiment minimax
    python scalability.py --experiment minimax_ab
    python scalability.py --experiment depth5
    python scalability.py --experiment depth5_plain

Set a custom time limit in seconds:

    python scalability.py --experiment minimax --duration 600


------------------------------------------------------------------------
STEP 7  HYPERPARAMETER EXPERIMENTS
------------------------------------------------------------------------

Run the Q-Learning epsilon decay comparison experiment:

    python design_choices.py

Saves results to results/experiments/.
To run other experiments (alpha, gamma, DQN target update), uncomment
the relevant sections in design_choices.py before running.


------------------------------------------------------------------------
QUICK REFERENCE  ALL COMMANDS
------------------------------------------------------------------------

    pip install numpy pygame torch matplotlib

    python connect4.py --train qlearn --episodes 1000000
    python connect4.py --train dqn    --episodes 300000

    python connect4.py --agent minimax_ab
    python connect4.py --agent minimax_ab --watch default

    python evaluate.py --games 500
    python plot_results.py
    python plot_training.py
    python scalability.py --experiment all
    python design_choices.py


------------------------------------------------------------------------
CREDITS
------------------------------------------------------------------------

Connect 4 game logic adapted from:
    Keith Galli, Connect4-Python
    https://github.com/KeithGalli/Connect4-Python

Pygame library:
    https://github.com/pygame/pygame

========================================================================