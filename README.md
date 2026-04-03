# CS7IS2 Artificial Intelligence Assignment 3

Implement Minimax and Reinforcement Learning algorithms for playing: Tic Tac Toe and Connect 4, and comparing their performance.

## Setup

python3 3.9

Create and activate a virtual environment:

    python3 -m venv venv
    source venv/bin/activate        (Mac/Linux)
    venv\Scripts\activate           (Windows)

Install dependencies:

    pip install -r requirements.txt

## Project Structure

    src/
        tictactoe/
            tictactoe.py            Game logic, all agents, Pygame UI, entry point
            dqn_train.py            Train DQN agent
            evaluate.py             Headless evaluation across all algorithm matchups
            plot_results.py         Generate bar charts from evaluation CSV results
            plot_training_draws.py  Generate training curve plots from saved model files
            plot_coverage.py        Generate Q-table state coverage plot
        connect4/
            connect4.py             Game logic, all agents, Pygame UI, entry point
            evaluate.py             Evaluation across all algorithm matchups
            design_choices.py       Hyperparameter experiments for Q-Learning and DQN
            scalability.py          Minimax scalability experiments
            plot_results.py         Generate bar charts from evaluation CSV results
            plot_training.py        Generate training curve plots from saved model files
    results/                        All output files (created automatically)

## Tic Tac Toe

All commands should be run from `src/tictactoe/`.

Train Q-Learning agent:

    python3 tictactoe.py --train qlearn --episodes 100000

Train DQN agent:

    python3 dqn_train.py --episodes 100000 --strategy mixed
    python3 dqn_train.py --episodes 100000 --strategy random
    python3 dqn_train.py --episodes 100000 --strategy default

Play the game:

    Run to play against any algorithm:
    python3 tictactoe.py --agent minimax-ab (or minimax, qlearn, dqn)                     
    Run to watch algorithms play eachother:
    python3 tictactoe.py --agent minimax_ab --watch default     

Evaluate:

    python3 evaluate.py --games 1000
    python3 plot_results.py
    python3 plot_training_draws.py
    python3 plot_coverage.py

## Connect 4

All commands should be run from `src/connect4/`.

Train agents:

    python3 connect4.py --train qlearn --episodes 1000000
    python3 connect4.py --train dqn    --episodes 300000

Play the game:

    Run to play against any algorithm:
    python3 connect4.py --agent minimax_ab                     
    Run to watch algorithms play eachother:
    python3 connect4.py --agent dqn --watch qlearn              

Evaluate:

    python3 evaluate.py --games 500
    python3 plot_results.py
    python3 plot_training.py
    python3 scalability.py --experiment all
    python3 design_choices.py

## Credits

Pygame UI for ttt and c4: [pygame/pygame](https://github.com/pygame/pygame)

Minimax algorithm reference: [GeeksForGeeks](https://www.geeksforgeeks.org/finding-optimal-move-in-tic-tac-toe-using-minimax-algorithm-in-game-theory/)

Connect 4 algorithm reference: Keith Galli's [Connect4-python3](https://github.com/KeithGalli/Connect4-python3)


