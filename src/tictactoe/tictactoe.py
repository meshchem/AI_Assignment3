import pygame
import sys
import math
import random
import argparse
import pickle
import os
import numpy as np
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Global Variables and Pygame Constants
# ----------------------------------------------------------------------

# Players 
PLAYER_X = 1   
PLAYER_O = 2  

# Pygame constants (colours, dimensions, drawing sizes)
WIDTH = 600    # board width in pixels
HEIGHT = 700    # total window height (600 board + 100 status bar)
ROWS = 3
COLS = 3
CELL = WIDTH // COLS   # 200 px per cell

# Colours
WHITE = (255, 255, 255)
BLACK = (0,   0,   0)
BG_COL = pygame.Color("floralwhite")
LINE_COL = pygame.Color("plum")
X_COL = pygame.Color("lightcoral")
O_COL = pygame.Color("cornflowerblue")
WIN_COL = pygame.Color("palegreen")

# Drawing sizes
LINE_W = 15   # grid line width
CIRCLE_R = 60   # O circle radius
CIRCLE_W = 15   # O circle line width
CROSS_W = 25   # X line width
SPACE = 55   # inset from cell edge for X lines


# ----------------------------------------------------------------------
# Tic Tac Toe Game Logic 
# ----------------------------------------------------------------------

def create_board():
    return np.zeros((3, 3), dtype=int)

# Return a list of (row, col) tuples for every empty cell.
def get_valid_moves(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]


# Place player's mark at (row, col). 
# Returns True if legal.
def make_move(board, row, col, player):
    if board[row, col] == 0:
        board[row, col] = player
        return True
    return False
    
# Clear the cell at (row, col). 
# Used by Minimax to backtrack.
def undo_move(board, row, col):
    board[row][col] = 0


# Check all rows, columns, and diagonals.
# Returns the winning player (1 or 2) or None.
def check_winner(board):
    for i in range(3):
        # Row i: all same and non-zero
        if board[i, 0] != 0 and board[i, 0] == board[i, 1] == board[i, 2]:
            return board[i, 0]
        # Column i: all same and non-zero
        if board[0, i] != 0 and board[0, i] == board[1, i] == board[2, i]:
            return board[0, i]
    # Main diagonal
    if board[0, 0] != 0 and board[0, 0] == board[1, 1] == board[2, 2]:
        return board[0, 0]
    # Anti-diagonal
    if board[0, 2] != 0 and board[0, 2] == board[1, 1] == board[2, 0]:
        return board[0, 2]
    return None


# True when no empty cells remain.
def is_board_full(board):
    return len(get_valid_moves(board)) == 0

# True when the game is over (win or draw).
def is_terminal(board):
    return check_winner(board) is not None or is_board_full(board)

# Convert board to a hashable tuple.
# Used as Q-table key.
def board_to_tuple(board):
    return tuple(board[r][c] for r in range(ROWS) for c in range(COLS))

# Draws a strike through the winning line.
# Returns None if there is no winner yet.
def get_winning_line(board):
    # Rows
    for r in range(ROWS):
        if board[r][0] != 0 and board[r][0] == board[r][1] == board[r][2]:
            y = r * CELL + CELL // 2
            return (15, y), (WIDTH - 15, y)   # Return (start_pixel, end_pixel) for the line to draw
    
    # Columns
    for c in range(COLS):
        if board[0][c] != 0 and board[0][c] == board[1][c] == board[2][c]:
            x = c * CELL + CELL // 2
            return (x, 15), (x, WIDTH - 15)
    
    # Main diagonal
    if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
        return (15, 15), (WIDTH - 15, WIDTH - 15)
    
    # Anti-diagonal
    if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
        return (WIDTH - 15, 15), (15, WIDTH - 15)
    return None


# ----------------------------------------------------------------------
# DefaultOpponent: simple rule-based opponent that tries to win, block, then take good positions.
# ----------------------------------------------------------------------

class DefaultOpponent:
    def __init__(self, player1, player2):
        self.p1   = player1
        self.them = player2

    def choose_move(self, board):
        moves = get_valid_moves(board)

        # win immediately
        for (r, c) in moves:
            make_move(board, r, c, self.p1)
            if check_winner(board) == self.p1:
                undo_move(board, r, c)
                return (r, c)
            undo_move(board, r, c)

        # block opponent's win
        for (r, c) in moves:
            make_move(board, r, c, self.them)
            if check_winner(board) == self.them:
                undo_move(board, r, c)
                return (r, c)
            undo_move(board, r, c)

        # centre
        if (1, 1) in moves:
            return (1, 1)

        # corner
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        random.shuffle(corners)
        for corner in corners:
            if corner in moves:
                return corner

        # any cell
        return random.choice(moves)

# ----------------------------------------------------------------------
# Random Opponent: picks a completely random valid move, used as DQN training opponent.
# ----------------------------------------------------------------------
class RandomOpponent:
    def choose_move(self, board):
        moves = get_valid_moves(board)
        return random.choice(moves) if moves else None

# ----------------------------------------------------------------------
# Minimax algorithms
# ----------------------------------------------------------------------

# Global node counter so we can report it in experiments
minimax_node_count = 0

# Static board evaluator
def evaluate(board, agent_player, opp_player):
    winner = check_winner(board) 
    if winner == agent_player:
        return 10   # +10 if agent_player has won
    if winner == opp_player:
        return -10  # -10 if opp_player has won
    return 0        # draw or game not finished

# ----------------------------------------------------------------------
# Plain Minimax
# ----------------------------------------------------------------------
#   is_maximising = True, agent's (MAX) turn
#   is_maximising = False, the opponent's (MIN) turn
def minimax(board, depth, is_maximising, agent_player, opp_player):
    
    global minimax_node_count
    minimax_node_count += 1

    score = evaluate(board, agent_player, opp_player)

    # Agent won
    if score == 10:
        return score - depth

    # Opponent won
    if score == -10:
        return score + depth

    # Draw 
    if not get_valid_moves(board):
        return 0   # no moves left

    # MAX player's turn (agent)
    if is_maximising:
        best = -1000
        for (r, c) in get_valid_moves(board):
            make_move(board, r, c, agent_player)
            best = max(best, minimax(board, depth + 1, False,
                                     agent_player, opp_player))
            undo_move(board, r, c)
        return best

    # MIN player's turn (opponent)
    else:
        best = 1000
        for (r, c) in get_valid_moves(board):
            make_move(board, r, c, opp_player)
            best = min(best, minimax(board, depth + 1, True, agent_player, opp_player))
            undo_move(board, r, c)
        return best

# ----------------------------------------------------------------------
# Minimax with alpha-beta pruning
# ----------------------------------------------------------------------
#   alpha = -inf
#   beta  = +inf

def minimax_ab(board, depth, alpha, beta, is_maximising, agent_player, opp_player):
    
    global minimax_node_count
    minimax_node_count += 1

    score = evaluate(board, agent_player, opp_player)
    if score == 10:  return score - depth
    if score == -10: return score + depth
    if not get_valid_moves(board): return 0

    if is_maximising:
        best = -1000
        for (r, c) in get_valid_moves(board):
            make_move(board, r, c, agent_player)
            best  = max(best, minimax_ab(board, depth + 1, alpha, beta, False, agent_player, opp_player))
            undo_move(board, r, c)
            alpha = max(alpha, best)
            if beta <= alpha:
                break   # beta cut-off: MIN won't allow this branch
        return best
    else:
        best = 1000
        for (r, c) in get_valid_moves(board):
            make_move(board, r, c, opp_player)
            best = min(best, minimax_ab(board, depth + 1, alpha, beta, True, agent_player, opp_player))
            undo_move(board, r, c)
            beta = min(beta, best)
            if beta <= alpha:
                break   
        return best


# Iterate over all empty cells, score each with minimax,
# return the (row, col) with the highest score.
def find_best_move(board, agent_player, opp_player, use_alpha_beta=True):
    
    global minimax_node_count
    minimax_node_count = 0 

    best_val  = -1000
    best_move = None

    for (r, c) in get_valid_moves(board):
        make_move(board, r, c, agent_player)

        if use_alpha_beta:
            val = minimax_ab(board, 0, -1000, 1000, False, agent_player, opp_player)
        else:
            val = minimax(board, 0, False, agent_player, opp_player)

        undo_move(board, r, c)

        if val > best_val:
            best_val  = val
            best_move = (r, c)

    return best_move

class MinimaxAgent:
    def __init__(self, player1, opp_player, use_alpha_beta=True):
        self.p1    = player1
        self.p2    = opp_player
        self.use_ab = use_alpha_beta

    def choose_move(self, board):
        return find_best_move(board, self.p1, self.p2, use_alpha_beta=self.use_ab)


# ----------------------------------------------------------------------
# Tabular Q-Learning
# ----------------------------------------------------------------------

class QLearningAgent:

    # Tabular Q-Learning for Tic Tac Toe.
    def __init__(self, player1, opp_player, 
                 alpha=0.5, gamma=0.9, 
                 eps_start=1.0, eps_end=0.05, eps_decay=0.999995):

        self.p1 = player1
        self.p2 = opp_player
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.eps = eps_start  # current exploration rate (decays over training)
        self.eps_min = eps_end
        self.eps_decay = eps_decay

        # Q-table: state => {action => Q-value}, missing entries = 0
        self.q = defaultdict(lambda: defaultdict(float))

        # Track outcome of each training episode for later graphing
        self.win_history = []   # 1 = win, 0 = draw, -1 = loss

    # Return the move with the highest Q-value.
    # Ties broken randomly.
    def _best_move(self, state, moves):
        best_q = max(self.q[state][m] for m in moves)
        bests  = [m for m in moves if self.q[state][m] == best_q]
        return random.choice(bests)

    # Bellman update to Q(state, action).
    def _update(self, state, action, reward, board, terminal):
        old_q = self.q[state][action]

        if terminal:
            target = reward
        else:
            next_state = board_to_tuple(board)
            next_moves = get_valid_moves(board)
            max_next = max(self.q[next_state][m] for m in next_moves) \
                         if next_moves else 0.0
            target = reward + self.gamma * max_next

        # Q-learning update
        self.q[state][action] = old_q + self.alpha * (target - old_q)

    # Action selection (epsilon-greedy).
    # greedy=True: always pick argmax Q  (play / evaluation)
    # greedy=False: epsilon-greedy        (training)
    def choose_move(self, board, greedy=True):
        moves = get_valid_moves(board)
        if not moves:
            return None

        # Exploration: pick a random move
        if not greedy and random.random() < self.eps:
            return random.choice(moves)

        # Exploitation: pick the move with the highest Q-value
        state = board_to_tuple(board)
        return self._best_move(state, moves)

    # Training loop against a fixed opponent (DefaultOpponent or MinimaxAgent).
    def train(self, num_episodes, opponent):
        print(f"Q-Learning: Training for {num_episodes} episodes")

        # track Q-table size
        self.coverage_history = [] 

        for ep in range(num_episodes):
            board = create_board()
            result = 0
            last_state = None    
            last_action = None  

            if ep % 2 == 0:
                current = self.p1
            else:
                opp_action = opponent.choose_move(board)
                make_move(board, *opp_action, self.p2)
                current = self.p1

            # print(f"ep {ep}: agent goes {'first' if ep % 2 == 0 else 'second'}, "
            #     f"board empty: {(board == 0).all()}, board sum: {board.sum()}")

            while not is_terminal(board):

                # Agent's turn
                if current == self.p1:
                    state = board_to_tuple(board)
                    action = self.choose_move(board, greedy=False)
                    make_move(board, *action, self.p1)

                    winner = check_winner(board)
                    if winner == self.p1:
                        self._update(state, action, 1.0, board, terminal=True)
                        result = 1; break
                    elif is_board_full(board):
                        self._update(state, action, 0.5, board, terminal=True)
                        result = 0; break
                    else:
                        last_state = state
                        last_action = action
                        current = self.p2

                # Opponent's turn
                else:
                    opp_action = opponent.choose_move(board)
                    make_move(board, *opp_action, self.p2)

                    winner = check_winner(board)
                    if winner == self.p2:
                        if last_state is not None:
                            self._update(last_state, last_action,
                                         -1.0, board, terminal=True)
                        result = -1; break
                    elif is_board_full(board):
                        if last_state is not None:
                            self._update(last_state, last_action, 0.5, board, terminal=True)
                        result = 0; break
                    else:
                        if last_state is not None:
                            self._update(last_state, last_action,
                                          0.0, board, terminal=False)
                        current = self.p1

            self.win_history.append(result)
            # Decay exploration rate each episode
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

            if ep % 500 == 0:
                self.coverage_history.append((ep, len(self.q)))

            if (ep + 1) % 10_000 == 0:
                wins = self.win_history[-1_000:].count(1)
                print(f"  ep {ep+1:>6}/{num_episodes} | "
                      f"epsilon={self.eps:.4f} | "
                      f"win% (last 1k): {wins / 10:.1f}%")

        print("Q-Learning: Training complete.")

    # Save to file (Q-table and training history)
    def save(self, path="qlearn_ttt.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "q"        : dict(self.q),
                "history"  : self.win_history,
                "eps"  : self.eps,
                "coverage" : getattr(self, "coverage_history", []),
            }, f)

    def load(self, path="qlearn_ttt.pkl"):
        if not os.path.exists(path):
            print(f"Q-Learning: No saved file at {path}. Starting fresh.")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q = defaultdict(lambda: defaultdict(float),
                             {k: defaultdict(float, v)
                              for k, v in data["q"].items()})
        self.win_history = data.get("history", [])
        self.eps = data.get("eps", self.eps_min)
        print(f"Q-Learning: Loaded {len(self.q)} states from {path}")

# ----------------------------------------------------------------------
# Deep Q-Network (DQN)
# ----------------------------------------------------------------------

# Feed-forward network: 9 inputs => 256 => 128 => 9 Q-values
class QNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    BUFFER_SIZE   = 50_000
    BATCH_SIZE    = 128
    TARGET_UPDATE = 200

    def __init__(self, player1, opp_player, 
                 gamma=0.99, lr=5e-4, 
                 eps_start=1.0, eps_end=0.05, eps_decay=0.999995): 

        # hyperparameters
        self.p1 = player1
        self.p2 = opp_player
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay
        self.steps = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimiser = optim.Adam(self.online.parameters(), lr=lr)
        self.replay = deque(maxlen=self.BUFFER_SIZE)
        self.win_history = []

    # Encode board as float tensor: empty=0.0, p1=1.0, p2= -1.0
    def _encode(self, board):
        flat = [board[r][c] for r in range(ROWS) for c in range(COLS)]
        norm = [1.0 if v == self.p1 else (-1.0 if v == self.p2 else 0.0)
                for v in flat]
        return torch.tensor(norm, dtype=torch.float32, device=self.device)

    def choose_move(self, board, greedy=True):
        moves = get_valid_moves(board)
        if not moves:
            return None

        if not greedy and random.random() < self.eps:
            return random.choice(moves)

        self.online.eval()
        with torch.no_grad():
            q_vals = self.online(self._encode(board).unsqueeze(0)).squeeze()

        # Mask invalid moves with negative infinity so we never pick them
        mask = torch.full((9,), float('-inf'), device=self.device)
        for (r, c) in moves:
            mask[r * 3 + c] = q_vals[r * 3 + c]

        idx = int(mask.argmax())
        return (idx // 3, idx % 3)

    def _push(self, s, a, r, ns, done):
        self.replay.append((s, a, r, ns, done))

    def _learn(self):
        if len(self.replay) < self.BATCH_SIZE:
            return

        self.online.train()
        batch = random.sample(self.replay, self.BATCH_SIZE)
        states, actions, rewards, nexts, dones = zip(*batch)

        s = torch.stack(states)
        a = torch.tensor(actions, dtype=torch.long,    device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        n = torch.stack(nexts)
        d = torch.tensor(dones,   dtype=torch.float32, device=self.device)

        q_taken = self.online(s).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            q_next = self.target(n).max(1)[0]
        
        target_q = r + self.gamma * q_next * (1 - d)

        loss = F.smooth_l1_loss(q_taken, target_q)
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimiser.step()

        self.steps += 1
        if self.steps % self.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

    def train(self, num_episodes, opponent):
        print(f"DQN: Training for {num_episodes} episodes against {type(opponent).__name__}")

        for ep in range(num_episodes):
            board = create_board()
            result = 0
            game_memory = []   # store ALL transitions this episode

            # Play a full episode, recording every (s, a, s') transition.
            # Rewards are assigned after the episode ends so the outcome
            # (win/loss/draw) can be propagated back to every move made.
            current = self.p1

            while not is_terminal(board):
                if current == self.p1:
                    s = self._encode(board)
                    move = self.choose_move(board, greedy=False)
                    idx = move[0] * 3 + move[1]
                    make_move(board, *move, self.p1)
                    ns = self._encode(board)
                    done = is_terminal(board)

                    winner = check_winner(board)
                    if winner == self.p1:
                        # Agent wins, immediate positive reward
                        reward = 1.0; result = 1
                        self._push(s, idx, reward, ns, True)
                        self._learn()
                        break
                    elif done:
                        # Draw
                        reward = 0.5
                        self._push(s, idx, reward, ns, True)
                        self._learn()
                        break
                    else:
                        # Non-terminal agent move, store with 0 reward for now
                        game_memory.append((s, idx, ns))
                        current = self.p2

                else:
                    opp_move = opponent.choose_move(board)
                    make_move(board, *opp_move, self.p2)
                    done   = is_terminal(board)
                    winner = check_winner(board)

                    if winner == self.p2:
                        # Opponent wins, assign -1 to the agent's LAST move
                        # and a small penalty to all earlier moves
                        result = -1
                        for i, (s, idx, ns) in enumerate(game_memory):
                            # the last move that led to this loss gets -1,
                            # earlier moves get a smaller penalty
                            r = -1.0 if i == len(game_memory) - 1 else -0.1
                            self._push(s, idx, r, ns, i == len(game_memory) - 1)
                            self._learn()
                        break
                    elif done:
                        result = 0
                        for s, idx, ns in game_memory:
                            self._push(s, idx, 0.5, ns, True)
                            self._learn()
                        break
                    else:
                        current = self.p1

            self.win_history.append(result)
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

            if (ep + 1) % 5_000 == 0:
                wins = self.win_history[-500:].count(1)
                print(f"  ep {ep+1:>6}/{num_episodes} | "
                      f"eps={self.eps:.4f} | "
                      f"win% (last 500): {wins / 5:.1f}%")

        print("DQN: Training complete.")

    def save(self, path="dqn_ttt.pt"):
        torch.save({
            "online" : self.online.state_dict(),
            "target" : self.target.state_dict(),
            "history": self.win_history,
            "eps"    : self.eps,
            "steps"  : self.steps,
        }, path)
        print(f"DQN Saved weights to {path}")

    def load(self, path="dqn_ttt.pt"):
        if not os.path.exists(path):
            print(f"DQN: No checkpoint at {path}. Starting fresh.")
            return
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])
        self.win_history = ck.get("history", [])
        self.eps   = ck.get("eps",   self.eps_min)
        self.steps = ck.get("steps", 0)
        print(f"DQN Loaded checkpoint from {path}")


# Evaluation function: play two agents against each other without Pygame and report results.
# Alternates who goes first to reduce first-mover advantage bias.
def run_eval(agent, opponent, num_games=200):
    wins = draws = losses = 0

    for i in range(num_games):
        board = create_board()

        # Alternate sides
        if i % 2 == 0:
            first,  first_id  = agent,    PLAYER_X
            second, second_id = opponent, PLAYER_O
        else:
            first,  first_id  = opponent, PLAYER_X
            second, second_id = agent,    PLAYER_O

        agent_id = first_id if i % 2 == 0 else second_id
        order = [(first, first_id), (second, second_id)]
        turn = 0

        while not is_terminal(board):
            current_agent, pid = order[turn % 2]
            move = current_agent.choose_move(board)
            if move:
                make_move(board, *move, pid)
            turn += 1

        winner = check_winner(board)
        if winner == agent_id: 
            wins += 1
        elif winner is None: 
            draws += 1
        else:    
            losses += 1

    total = wins + draws + losses
    print(f"\n{'='*45}")
    print(f"  Results over {num_games} games:")
    print(f"  Wins   : {wins:5d}  ({wins   / total * 100:.1f}%)")
    print(f"  Draws  : {draws:5d}  ({draws  / total * 100:.1f}%)")
    print(f"  Losses : {losses:5d}  ({losses / total * 100:.1f}%)")
    print(f"{'='*45}\n")


# ----------------------------------------------------------------------
# Pygame Drawing Functions  
# ----------------------------------------------------------------------
    
def draw_lines(screen):
    # Draw the two horizontal and two vertical grid lines.
    pygame.draw.line(screen, LINE_COL, (0,CELL), (WIDTH, CELL), LINE_W)
    pygame.draw.line(screen, LINE_COL, (0,CELL * 2), (WIDTH, CELL * 2), LINE_W)
    pygame.draw.line(screen, LINE_COL, (CELL, 0), (CELL, WIDTH), LINE_W)
    pygame.draw.line(screen, LINE_COL, (CELL * 2, 0), (CELL * 2, WIDTH), LINE_W)


def draw_figures(screen, board):
    # Draw X lines and O circles for every filled cell.
    for r in range(ROWS):
        for c in range(COLS):
            cx = c * CELL + CELL // 2
            cy = r * CELL + CELL // 2

            if board[r][c] == PLAYER_O:
                pygame.draw.circle(screen, O_COL, (cx, cy), CIRCLE_R, CIRCLE_W)

            elif board[r][c] == PLAYER_X:
                pygame.draw.line(screen, X_COL,
                                 (c * CELL + SPACE, r * CELL + SPACE),
                                 ((c + 1) * CELL - SPACE, (r + 1) * CELL - SPACE),
                                 CROSS_W)
                pygame.draw.line(screen, X_COL,
                                 ((c + 1) * CELL - SPACE, r * CELL + SPACE),
                                 (c * CELL + SPACE, (r + 1) * CELL - SPACE), 
                                 CROSS_W)


def draw_status(screen, font, text):
    # Draw the black status bar below the board with centred white text.
    pygame.draw.rect(screen, BLACK, (0, WIDTH, WIDTH, HEIGHT - WIDTH))
    surf = font.render(text, True, WHITE)
    screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2,
                       WIDTH + (HEIGHT - WIDTH) // 2 - surf.get_height() // 2))


def draw_strike(screen, board):
    # Draw a gold strike-through line over the winning combination.
    line = get_winning_line(board)
    if line:
        pygame.draw.line(screen, WIN_COL, line[0], line[1], 8)


def redraw(screen, board, font, status_text, show_strike=False):
    # Full redraw of the board each frame.
    screen.fill(BG_COL)
    draw_lines(screen)
    draw_figures(screen, board)
    if show_strike:
        draw_strike(screen, board)
    draw_status(screen, font, status_text)
    pygame.display.update()


# ----------------------------------------------------------------------
# Main Pygame Game Loop
# ----------------------------------------------------------------------

def run_game(agent, agent_name, agent2=None, agent2_name=None):
    watch_mode = agent2 is not None
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    if watch_mode:
        pygame.display.set_caption(f"Tic Tac Toe:  {agent2_name} (X)  vs  {agent_name} (O)")
    else:
        pygame.display.set_caption(f"Tic Tac Toe:  Human (X)  vs  {agent_name} (O)")
    font  = pygame.font.SysFont("monospace", 20, bold=True)
    clock = pygame.time.Clock()

    board = create_board()
    player1_id = PLAYER_X
    player2_id = PLAYER_O
    game_over = False
    scores = {PLAYER_X: 0, PLAYER_O: 0, "draw": 0}

    # Restart button 
    btn_w, btn_h = 110, 34
    btn = pygame.Rect(WIDTH - btn_w - 12,
                      WIDTH + (HEIGHT - WIDTH) // 2 - btn_h // 2,
                      btn_w, btn_h)

    AI_DELAY = 500
    last_move_time = pygame.time.get_ticks()
    current = player1_id

    def do_redraw(status):
        screen.fill(BG_COL)
        for i in range(1, 3):
            pygame.draw.line(screen, LINE_COL, (i*CELL, 0), (i*CELL, WIDTH), LINE_W)
            pygame.draw.line(screen, LINE_COL, (0, i*CELL), (WIDTH, i*CELL), LINE_W)
        draw_figures(screen, board)
        draw_strike(screen, board)
        pygame.draw.rect(screen, BLACK, (0, WIDTH, WIDTH, HEIGHT - WIDTH))
        bar_cy  = WIDTH + (HEIGHT - WIDTH) // 2
        sc_surf = font.render(
            f"X:{scores[PLAYER_X]}  O:{scores[PLAYER_O]}  Draw:{scores['draw']}",
            True, WHITE)
        screen.blit(sc_surf, (12, bar_cy - sc_surf.get_height() // 2))
        st_surf = font.render(status, True, WHITE)
        screen.blit(st_surf, (WIDTH // 2 - st_surf.get_width() // 2,
                               bar_cy - st_surf.get_height() // 2))
        bc = pygame.Color("thistle") if btn.collidepoint(pygame.mouse.get_pos()) else pygame.Color("plum")
        pygame.draw.rect(screen, bc, btn, border_radius=8)
        bl = font.render("Restart", True, WHITE)
        screen.blit(bl, (btn.centerx - bl.get_width() // 2,
                          btn.centery - bl.get_height() // 2))
        pygame.display.update()

    if watch_mode:
        do_redraw(f"{agent2_name}'s Turn")
    else:
        do_redraw("Your turn")

    while True:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                board  = create_board()
                game_over = False
                current = player1_id
                last_move_time = pygame.time.get_ticks()
                do_redraw(f"{agent2_name} thinking..." if watch_mode else "Your turn")

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if btn.collidepoint(mx, my):
                    w = check_winner(board)
                    if game_over:
                        scores[w if w else "draw"] += 1
                    board = create_board()
                    game_over = False
                    current = player1_id
                    last_move_time = pygame.time.get_ticks()
                    do_redraw(f"{agent2_name}'s Turn" if watch_mode else "Your turn")
                    continue

                # Human move, only in play mode
                if (not watch_mode
                        and not game_over
                        and current == player1_id
                        and my < WIDTH):
                    row = my // CELL
                    col = mx // CELL
                    if make_move(board, row, col, player1_id):
                        last_move_time = pygame.time.get_ticks()
                        winner = check_winner(board)
                        if winner == player1_id:
                            game_over = True
                            do_redraw("You win!  (X)")
                        elif is_board_full(board):
                            game_over = True
                            do_redraw("Draw!")
                        else:
                            current = player2_id
                            do_redraw(f"{agent_name}'s Turn")

        # Agent 2 move (X, goes first), watch mode only
        if (watch_mode
                and not game_over
                and current == player1_id
                and pygame.time.get_ticks() - last_move_time >= AI_DELAY):
            move = agent2.choose_move(board)
            if move:
                make_move(board, *move, player1_id)
            last_move_time = pygame.time.get_ticks()
            winner = check_winner(board)
            if winner == player1_id:
                game_over = True
                do_redraw(f"{agent2_name} wins!")
            elif is_board_full(board):
                game_over = True
                do_redraw("Draw!")
            else:
                current = player2_id
                do_redraw(f"{agent_name}'s Turn")

        # Agent (O) move
        if (not game_over
                and current == player2_id
                and pygame.time.get_ticks() - last_move_time >= AI_DELAY):
            move = agent.choose_move(board)
            if move:
                make_move(board, *move, player2_id)
            last_move_time = pygame.time.get_ticks()
            winner = check_winner(board)
            if winner == player2_id:
                game_over = True
                do_redraw(f"{agent_name} wins!")
            elif is_board_full(board):
                game_over = True
                do_redraw("Draw!")
            else:
                current = player1_id
                do_redraw(f"{agent2_name}'s Turn" if watch_mode else "Your turn")


# main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tic Tac Toe")

    parser.add_argument("--agent", default="minimax_ab",
                        choices=["minimax", "minimax_ab", "qlearn", "dqn", "default"],
                        help="AI to play against (plays as O)")
    parser.add_argument("--watch", default=None,
                        choices=["minimax", "minimax_ab", "qlearn", "dqn", "default"],
                        help="Second agent to watch play as X against --agent")
    parser.add_argument("--train", default=None, choices=["qlearn", "dqn"])
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--load", type=str, default=None)
    
    args = parser.parse_args()

    player1_id = PLAYER_X
    player2_id = PLAYER_O
    default_opp = DefaultOpponent(player1=player2_id, player2=player1_id)

    # Build main agent (plays as O)
    if args.agent in ("minimax", "minimax_ab"):
        agent = MinimaxAgent(player1=player2_id, opp_player=player1_id,
                             use_alpha_beta=(args.agent == "minimax_ab"))
    elif args.agent == "qlearn":
        agent = QLearningAgent(player1=player2_id, opp_player=player1_id)
        agent.load(args.load or "qlearn_ttt.pkl")
    elif args.agent == "dqn":
        agent = DQNAgent(player1=player2_id, opp_player=player1_id)
        agent.load(args.load or "dqn_ttt.pt")
    else:
        agent = DefaultOpponent(player1=player2_id, player2=player1_id)

    # Build watch agent (plays as X) if --watch specified
    agent2 = None
    if args.watch:
        if args.watch in ("minimax", "minimax_ab"):
            agent2 = MinimaxAgent(player1=player1_id, opp_player=player2_id,
                                  use_alpha_beta=(args.watch == "minimax_ab"))
        elif args.watch == "qlearn":
            agent2 = QLearningAgent(player1=player1_id, opp_player=player2_id)
            agent2.load(args.load or "qlearn_ttt.pkl")
        elif args.watch == "dqn":
            agent2 = DQNAgent(player1=player1_id, opp_player=player2_id)
            agent2.load(args.load or "dqn_ttt.pt")
        else:
            agent2 = DefaultOpponent(player1=player1_id, player2=player2_id)

    # Training
    if args.train == "qlearn":
        train_opp = DefaultOpponent(player1=player1_id, player2=player2_id)
        agent = QLearningAgent(player1=player2_id, opp_player=player1_id)
        agent.train(args.episodes, train_opp)
        agent.save()
        sys.exit()
    elif args.train == "dqn":
        agent = DQNAgent(player1=player2_id, opp_player=player1_id)
        random_opp = RandomOpponent()
        agent.train(args.episodes, random_opp)
        agent.save()
        sys.exit()

    # Run
    if args.eval:
        run_eval(agent, default_opp, num_games=args.games)
    else:
        run_game(agent, args.agent, agent2=agent2, agent2_name=args.watch)