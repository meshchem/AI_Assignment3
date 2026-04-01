# ----------------------------------------------------------------------
# scalability.py
# CS7IS2 Assignment 3 - Scalability experiments for Connect 4 Minimax
#
# Usage:
#   python3 scalability.py --experiment minimax
#   python3 scalability.py --experiment minimax_ab
#   python3 scalability.py --experiment depth5
#   python3 scalability.py --experiment depth5_plain
#   python3 scalability.py --experiment all
#
#   python3 scalability.py --experiment minimax --duration 1800
# ----------------------------------------------------------------------

import argparse
import math
import time

from connect4 import (
    Connect4Game, minimax,
    PLAYER1_PIECE, PLAYER2_PIECE,
    ROWS, COLS
)
import connect4  
# ----------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------

minimax_node_count  = 0
_scalability_start  = None
_scalability_limit  = None


# ----------------------------------------------------------------------
# Setup / teardown
# ----------------------------------------------------------------------

def _setup_scalability(duration_seconds):
    global minimax_node_count, _scalability_start, _scalability_limit
    minimax_node_count = 0
    _scalability_start = time.time()
    _scalability_limit = _scalability_start + duration_seconds

    # sync with connect4.py globals so the ticker fires
    connect4._scalability_start = _scalability_start
    connect4._scalability_limit = _scalability_limit
    connect4._last_tick_time    = _scalability_start
    connect4.minimax_node_count = 0

def _teardown_scalability():
    global _scalability_start, _scalability_limit
    _scalability_start = None
    _scalability_limit = None

    # reset connect4.py globals
    connect4._scalability_start = None
    connect4._scalability_limit = None

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

def _print_summary(label, nodes, elapsed, moves, time_per_move):
    MAX_MOVES = ROWS * COLS
    print(f"\n  {label} Summary:")
    print(f"    Total nodes visited : {nodes:,}")
    print(f"    Total time          : {elapsed:.1f}s")
    print(f"    Moves completed     : {moves} / {MAX_MOVES}")
    if moves > 0:
        avg_t = sum(time_per_move) / moves
        print(f"    Avg nodes / move    : {nodes // moves:,}")
        print(f"    Avg time  / move    : {avg_t:.2f}s")
        projected = avg_t * MAX_MOVES
        print(f"    Projected full game : ~{projected:.0f}s  (~{projected/3600:.1f} hrs)")
    else:
        rate = nodes / elapsed if elapsed > 0 else 0
        projected = (4.5e12 / rate) if rate > 0 else float('inf')
        print(f"    Nodes / second      : {rate:,.0f}")
        print(f"    Projected full game : ~{projected:.0f}s  (~{projected/3600:.1f} hrs)")
    print()


# ----------------------------------------------------------------------
# Plain Minimax
# ----------------------------------------------------------------------

def run_scalability_minimax(duration_seconds=1800):
    global minimax_node_count

    print(f"\n{'-'*65}")
    print(f"  Scalability: Minimax, {duration_seconds}s limit")
    print(f"{'-'*65}")

    _setup_scalability(duration_seconds)
    start          = _scalability_start
    game           = Connect4Game()
    moves          = 0
    nodes_per_move = []
    time_per_move  = []

    while not game.is_terminal():
        if time.time() > _scalability_limit:
            print(f"\nTimeout after {time.time()-start:.0f}s", flush=True)
            break

        minimax_node_count = 0
        move_start = time.time()

        col, _ = minimax(game, depth=99,
                         alpha=-math.inf, beta=math.inf,
                         maximising_player=True,
                         agent_piece=PLAYER1_PIECE,
                         use_ab=False)

        move_time = time.time() - move_start
        if col is None:
            break

        nodes_per_move.append(connect4.minimax_node_count)
        time_per_move.append(move_time)
        moves += 1
        print(f"  {'Move':>5}  {'Nodes (this move)':>20}  {'Time (s)':>10}  {'Cumulative':>14}")
        print(f"  {moves:>5}  {minimax_node_count:>20,}  "
              f"{move_time:>10.2f}  {sum(nodes_per_move):>14,}", flush=True)
        game.make_move(col)

    _teardown_scalability()
    _print_summary("Minimax Algorithm",
                   sum(nodes_per_move) + minimax_node_count,
                   time.time() - start, moves, time_per_move)


# ----------------------------------------------------------------------
# Minimax with alpha-beta pruning
# ----------------------------------------------------------------------

def run_scalability_ab(duration_seconds=1800):
    global minimax_node_count

    print(f"\n{'-'*65}")
    print(f"  Scalability: Minimax + Alpha-Beta, {duration_seconds}s limit")
    print(f"{'-'*65}")

    _setup_scalability(duration_seconds)
    start          = _scalability_start
    game           = Connect4Game()
    moves          = 0
    nodes_per_move = []
    time_per_move  = []

    while not game.is_terminal():
        if time.time() > _scalability_limit:
            print(f"\nTimeout after {time.time()-start:.0f}s", flush=True)
            break

        minimax_node_count = 0
        move_start = time.time()

        col, _ = minimax(game, depth=99,
                         alpha=-math.inf, beta=math.inf,
                         maximising_player=True,
                         agent_piece=PLAYER1_PIECE,
                         use_ab=True)

        move_time = time.time() - move_start
        if col is None:
            break

        nodes_per_move.append(connect4.minimax_node_count)
        time_per_move.append(move_time)
        moves += 1
        print(f"  {'Move':>5}  {'Nodes (this move)':>20}  {'Time (s)':>10}  {'Cumulative':>14}")
        print(f"  {moves:>5}  {minimax_node_count:>20,}  "
              f"{move_time:>10.2f}  {sum(nodes_per_move):>14,}", flush=True)
        game.make_move(col)

    _teardown_scalability()
    _print_summary("Minimax Alpha-Beta",
                   sum(nodes_per_move) + minimax_node_count,
                   time.time() - start, moves, time_per_move)


# ----------------------------------------------------------------------
# Depth-limited minimax (depth=5)
# ----------------------------------------------------------------------

def run_scalability_depth5(use_alpha_beta=True):
    global minimax_node_count
    label = "Depth-5 + Alpha-Beta" if use_alpha_beta else "Depth-5 (plain Minimax)"

    print(f"\n{'-'*65}")
    print(f"  Scalability: {label}")
    print(f"{'-'*65}")
    print(f"  {'Move':>5}  {'Nodes (this move)':>20}  {'Time (s)':>10}  {'Cumulative':>14}")
    print(f"  {'-'*55}", flush=True)

    _teardown_scalability()
    game           = Connect4Game()
    start          = time.time()
    moves          = 0
    nodes_per_move = []
    time_per_move  = []

    while not game.is_terminal():
        minimax_node_count = 0
        move_start = time.time()

        col, _ = minimax(game, depth=5,
                         alpha=-math.inf, beta=math.inf,
                         maximising_player=True,
                         agent_piece=PLAYER1_PIECE,
                         use_ab=use_alpha_beta)

        move_time = time.time() - move_start
        if col is None:
            break

        nodes_per_move.append(connect4.minimax_node_count)
        time_per_move.append(move_time)
        moves += 1
        print(f"  {moves:>5}  {minimax_node_count:>20,}  "
              f"{move_time:>10.4f}  {sum(nodes_per_move):>14,}", flush=True)
        game.make_move(col)

    _print_summary(label, sum(nodes_per_move),
                   time.time() - start, moves, time_per_move)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Connect 4 Scalability Experiments")
    
    # argument to select which experiment(s) to run
    parser.add_argument("--experiment", default="all",
                        choices=["minimax", "minimax_ab", "depth5", "depth5_plain", "all"],
                        help="Which experiment to run")
    parser.add_argument("--duration", type=int, default=1800,
                        help="Time limit in seconds for full-depth experiments (default 1800)")
    
    args = parser.parse_args()

    if args.experiment == "minimax":
        run_scalability_minimax(args.duration)

    elif args.experiment == "minimax_ab":
        run_scalability_ab(args.duration)

    elif args.experiment == "depth5":
        run_scalability_depth5(use_alpha_beta=True)

    elif args.experiment == "depth5_plain":
        run_scalability_depth5(use_alpha_beta=False)

    elif args.experiment == "all":
        run_scalability_minimax(args.duration)
        run_scalability_ab(args.duration)
        run_scalability_depth5(use_alpha_beta=True)
        run_scalability_depth5(use_alpha_beta=False)