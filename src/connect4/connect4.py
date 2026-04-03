import pygame, sys, math, random, argparse, pickle, os, time
import numpy as np
from collections import defaultdict, deque
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pygame and UI constants
# ---------------------------------------------------------------------------

ROWS      = 6
COLS      = 7
CELL_SIZE = 100
RADIUS    = int(CELL_SIZE / 2 - 5)

width   = COLS * CELL_SIZE
PANEL_H = 80
height  = (ROWS + 1) * CELL_SIZE + PANEL_H
PANEL_Y = height - PANEL_H

BLUE   = pygame.Color("blue")
BLACK  = pygame.Color("black")
RED    = pygame.Color("red")
YELLOW = pygame.Color("yellow")
GREEN  = pygame.Color("lawngreen")
GREY   = pygame.Color("gray20")
GREY2  = pygame.Color("gray35")
GREY3  = pygame.Color("gray50")
WHITE  = pygame.Color("white")

# Algorithm player constants kept separate from UI player IDs
# The game uses 1 and -1 internally (doc-11 convention)
PLAYER1_PIECE  = 1    # human / agent 1
PLAYER2_PIECE      = -1   # AI opponent / agent 2
EMPTY         = 0
WINDOW_LENGTH = 4

# ----------------------------------------------------------------------
# Minimax scalability global variables
# ----------------------------------------------------------------------
minimax_node_count = 0
_scalability_start = None  
_last_tick_time    = 0     
_scalability_limit = None   


# ----------------------------------------------------------------------
# Connect 4 Game Logic Class
# ----------------------------------------------------------------------

class Connect4Game:

    def __init__(self):
        self.reset()

    def reset(self):
        self.board          = np.zeros((ROWS, COLS), dtype=float)
        self.current_player = PLAYER1_PIECE   # player 1 goes first
        self.winner         = None
        self.game_over      = False
        self.winning_cells  = []

    # Check position 

    def is_valid_location(self, col):
        # Column is valid if the TOP row (row ROWS-1) is still empty
        return self.board[ROWS - 1][col] == 0

    def get_valid_moves(self):
        return [c for c in range(COLS) if self.is_valid_location(c)]

    def get_next_open_row(self, col):
        # Scan from the bottom (row 0) upward, first empty row
        for r in range(ROWS):
            if self.board[r][col] == 0:
                return r

    # Moves 

    # places the piece in the board array 
    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def make_move(self, col):
        # Returns False if move is illegal (column full or game over)
        if not self.is_valid_location(col) or self.game_over:
            return False
        row = self.get_next_open_row(col)
        self.drop_piece(row, col, self.current_player)
        # Check for win/draw and update game state
        if self.winning_move(self.current_player):
            self.winner    = self.current_player
            self.game_over = True
        elif not self.get_valid_moves():
            self.game_over = True   # draw
        else:
            self.current_player *= -1   # swap turn
        return True

    # Winning move, checks all directions for a connect 4. 
    # If found, store the winning cells for UI highlighting.
    def winning_move(self, piece):
        # Horizontal
        for c in range(COLS - 3):
            for r in range(ROWS):
                if all(self.board[r][c+i] == piece for i in range(4)):
                    self.winning_cells = [(r, c+i) for i in range(4)]
                    return True
        # Vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                if all(self.board[r+i][c] == piece for i in range(4)):
                    self.winning_cells = [(r+i, c) for i in range(4)]
                    return True
        # Positive diagonal
        for c in range(COLS - 3):
            for r in range(ROWS - 3):
                if all(self.board[r+i][c+i] == piece for i in range(4)):
                    self.winning_cells = [(r+i, c+i) for i in range(4)]
                    return True
        # Negative diagonal
        for c in range(COLS - 3):
            for r in range(3, ROWS):
                if all(self.board[r-i][c+i] == piece for i in range(4)):
                    self.winning_cells = [(r-i, c+i) for i in range(4)]
                    return True
        return False

    def is_terminal(self):
        return self.game_over

    # For Minimax: 
    # create a deep copy of the game state to simulate moves without affecting the real game.
    def clone(self):
        g = Connect4Game()
        g.board          = self.board.copy()
        g.current_player = self.current_player
        g.winner         = self.winner
        g.game_over      = self.game_over
        g.winning_cells  = list(self.winning_cells)
        return g

    # For Q-Learning: 
    # convert the board to a hashable tuple state representation.
    def get_state(self):
        return tuple(self.board.flatten())

    def print_board(self):
        print(np.flip(self.board, 0))


# ---------------------------------------------------------------------------
# Scoring Function for minimax heuristic evaluation
# ---------------------------------------------------------------------------

# Evaluate a 4-cell window for the given piece, 
# returning a score based on how favourable it is.
def evaluate_window(window, piece):
    score     = 0
    opp_piece = -piece   # works because we use 1 / -1

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

# Score the whole board for the given piece by checking all possible 4-cell windows.
def score_position(game, piece):
    score = 0
    board = game.board

    # Centre column preference
    centre_col = [int(board[r][COLS // 2]) for r in range(ROWS)]
    score += centre_col.count(piece) * 3

    # Horizontal
    for r in range(ROWS):
        row_arr = [int(board[r][c]) for c in range(COLS)]
        for c in range(COLS - 3):
            score += evaluate_window(row_arr[c:c+WINDOW_LENGTH], piece)

    # Vertical
    for c in range(COLS):
        col_arr = [int(board[r][c]) for r in range(ROWS)]
        for r in range(ROWS - 3):
            score += evaluate_window(col_arr[r:r+WINDOW_LENGTH], piece)

    # Diagonal (positive)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [int(board[r+i][c+i]) for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Diagonal (negative)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [int(board[r-i][c+i]) for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


# ---------------------------------------------------------------------------
# Rule based default opponent (used for Minimax and RL training)
# ---------------------------------------------------------------------------

class DefaultOpponent:
    def __init__(self, player):
        self.player = player   # PLAYER1_PIECE or PLAYER2_PIECE

    def choose_move(self, game):
        moves = game.get_valid_moves()

        # Win immediately
        for col in moves:
            g = game.clone()
            g.make_move(col)
            if g.winner == self.player:
                return col

        # Block opponent, simulate opponent's move correctly
        opp = -self.player
        for col in moves:
            g = game.clone()
            g.current_player = opp  # temporarily set to opp to simulate their move
            g.make_move(col)
            if g.winner == opp:
                return col

        # Centre preference
        for col in [3, 2, 4, 1, 5, 0, 6]:
            if col in moves:
                return col

        return moves[0]


# -------------------------------------------------------------
# Plain Minimax
# -------------------------------------------------------------
#   is_maximising = True, agent's (MAX) turn
#   is_maximising = False, the opponent's (MIN) turn
def minimax(game, depth, alpha, beta, maximising_player, agent_piece, use_ab=True):
   
    global minimax_node_count, _last_tick_time
    minimax_node_count += 1

    # Scalability limit
    if _scalability_limit is not None and time.time() > _scalability_limit:
        return (None, 0)

    # Live ticker for scalability experiment, print progress every 5 seconds.
    if _scalability_start is not None:
        now = time.time()
        if now - _last_tick_time >= 5.0:
            elapsed = now - _scalability_start
            print(f"    Time elapsed: {elapsed:.0f}s,  "
                  f"Nodes searched: {minimax_node_count:,}",
                  flush=True)
            _last_tick_time = now

    opp_piece   = -agent_piece
    valid_moves = game.get_valid_moves()

    # Base cases
    if game.winner == agent_piece:
        return (None,  100_000_000)
    if game.winner == opp_piece:
        return (None, -100_000_000)
    if not valid_moves:
        return (None, 0)
    if depth == 0:
        return (None, score_position(game, agent_piece))

    if maximising_player:
        value  = -math.inf
        column = random.choice(valid_moves)
        for col in valid_moves:
            g = game.clone()
            g.make_move(col)
            new_score = minimax(g, depth-1, alpha, beta, False, agent_piece, use_ab)[1]
            if new_score > value:
                value = new_score; column = col
            if use_ab:
                alpha = max(alpha, value)
                if alpha >= beta: break
        return column, value
    else:
        value  = math.inf
        column = random.choice(valid_moves)
        for col in valid_moves:
            g = game.clone()
            g.make_move(col)
            new_score = minimax(g, depth-1, alpha, beta, True, agent_piece, use_ab)[1]
            if new_score < value:
                value = new_score; column = col
            if use_ab:
                beta = min(beta, value)
                if alpha >= beta: break
        return column, value


class MinimaxAgent:
    def __init__(self, piece, depth=5, use_alpha_beta=True):
        self.piece  = piece
        self.depth  = depth
        self.use_ab = use_alpha_beta

    def choose_move(self, game):
        global minimax_node_count
        minimax_node_count = 0
        # maximising_player is True only if it's actually the agent's turn
        is_max = (game.current_player == self.piece)
        col, _ = minimax(game, self.depth, -math.inf, math.inf,
                        is_max, self.piece, self.use_ab)
        return col


# -------------------------------------------------------------
# Tabular Q-Learning
# -------------------------------------------------------------
# Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',.) - Q(s,a)]
class QLearningAgent:
    # alpha = learning rate, gamma = discount factor, eps = exploration rate
    def __init__(self, piece, opp_piece,
                 alpha=0.5, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.99995):

        self.piece     = piece
        self.opp_piece = opp_piece
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps       = eps_start
        self.eps_min   = eps_end
        self.eps_decay = eps_decay
        self.q         = defaultdict(lambda: defaultdict(float))
        self.win_history = []

    # Epsilon-greedy action selection: with probability eps, pick a random valid move,
    # otherwise pick the move with the highest Q-value for the current state.
    def choose_move(self, game, greedy=True):
        valid = game.get_valid_moves()
        if not valid: return None
        if not greedy and random.random() < self.eps:
            return random.choice(valid)
        state  = game.get_state()
        best_q = max(self.q[state][c] for c in valid)
        bests  = [c for c in valid if self.q[state][c] == best_q]
        return random.choice(bests)

    # Q-learning update: after taking action col in state, 
    # receiving reward and transitioning to new state,
    # update Q[state][col] towards the target value.
    def _update(self, state, col, reward, game, terminal):
        old = self.q[state][col]
        if terminal:
            target = reward
        else:
            ns    = game.get_state()
            valid = game.get_valid_moves()
            maxnq = max(self.q[ns][c] for c in valid) if valid else 0.0
            target = reward + self.gamma * maxnq
        self.q[state][col] = old + self.alpha * (target - old)

    # Train for a number of episodes against the given opponent.
    def train(self, num_episodes, opponent):
        
        print(f"Q-Learning: Training for {num_episodes} episodes")
        
        # track Q-table size
        self.coverage_history = [] 

        for ep in range(num_episodes):
            game = Connect4Game(); result = 0
            last_state = last_col = None

            while not game.is_terminal():
                if game.current_player == self.piece:
                    state = game.get_state()
                    col   = self.choose_move(game, greedy=False)
                    game.make_move(col)
                    if game.winner == self.piece:
                        self._update(state, col, 1.0, game, True); result = 1; break
                    elif game.game_over:
                        self._update(state, col, 0.2, game, True); result = 0; break
                    else:
                        last_state = state; last_col = col
                else:
                    col = opponent.choose_move(game)
                    game.make_move(col)
                    if game.winner == self.opp_piece:
                        if last_state: self._update(last_state, last_col, -1.0, game, True)
                        result = -1; break
                    elif game.game_over:
                        if last_state: self._update(last_state, last_col, 0.2, game, True)
                        result = 0; break
                    else:
                        if last_state: self._update(last_state, last_col, 0.0, game, False)

            self.win_history.append(result)
            
            if ep % 500 == 0:
                self.coverage_history.append((ep, len(self.q)))

            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            if (ep+1) % 5000 == 0:
                wins = self.win_history[-1000:].count(1)
                print(f"  ep {ep+1}/{num_episodes} | eps={self.eps:.4f} | win%={wins/10:.1f}%")
        print("Q-Learning Done.")

    def save(self, path="results/qlearn_c4.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "q"        : dict(self.q),
                "history"  : self.win_history,
                "eps"  : self.eps,
                "coverage" : getattr(self, "coverage_history", []),
            }, f)

        print(f"Q-Learning saved to {path}")

    def load(self, path="results/qlearn_c4.pkl"):
        if not os.path.exists(path): print(f"No file at {path}"); return
        with open(path, "rb") as f: d = pickle.load(f)
        self.q = defaultdict(lambda: defaultdict(float),
                             {k: defaultdict(float, v) for k, v in d["q"].items()})
        self.win_history = d.get("history", [])
        self.eps         = d.get("eps", self.eps_min)
        print(f"Q-Learning loaded {len(self.q)} states from {path}")


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    
    # Simple feedforward network with 2 hidden layers.
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(42, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    BUFFER_SIZE   = 100_000
    BATCH_SIZE    = 256
    TARGET_UPDATE = 200

    # piece = agent's piece (1 or -1), opp_piece = opponent's piece
    def __init__(self, piece, opp_piece,
                 gamma=0.99, lr=1e-4,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.999975):

        self.piece     = piece
        self.opp_piece = opp_piece
        self.gamma     = gamma
        self.eps       = eps_start
        self.eps_min   = eps_end
        self.eps_decay = eps_decay
        self.steps     = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Device: {self.device}")

        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt    = optim.Adam(self.online.parameters(), lr=lr)
        self.replay = deque(maxlen=self.BUFFER_SIZE)
        self.win_history = []

    # empty=0, agent's pieces=1, opponent's pieces= -1.
    def _encode(self, game):
        flat = game.board.flatten()
        norm = np.where(flat == self.piece, 1.0,
               np.where(flat == self.opp_piece, -1.0, 0.0)).astype(np.float32)
        return torch.tensor(norm, device=self.device)

    # Epsilon-greedy action selection: with probability eps, pick a random valid move,
    def choose_move(self, game, greedy=True):
        valid = game.get_valid_moves()
        if not valid: return None
        if not greedy and random.random() < self.eps:
            return random.choice(valid)
        self.online.eval()
        with torch.no_grad():
            q = self.online(self._encode(game).unsqueeze(0)).squeeze()
        mask = torch.full((7,), float('-inf'), device=self.device)
        for c in valid: mask[c] = q[c]
        return int(mask.argmax())

    # Learn from a batch of replay experiences: (state, action, reward, next_state, done).
    def _learn(self):
        if len(self.replay) < self.BATCH_SIZE: return
        self.online.train()
        batch = random.sample(self.replay, self.BATCH_SIZE)
        s, a, r, n, d = zip(*batch)
        s = torch.stack(s)
        a = torch.tensor(a, dtype=torch.long,    device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        n = torch.stack(n)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        q_t = self.online(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad(): qn = self.target(n).max(1)[0]
        tgt  = r + self.gamma * qn * (1 - d)
        loss = F.smooth_l1_loss(q_t, tgt)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

    # Train 
    def train(self, num_episodes, opponent):
        print(f"DQN: Training for {num_episodes} episodes")
        for ep in range(num_episodes):
            game = Connect4Game(); result = 0; memory = []

            while not game.is_terminal():
                if game.current_player == self.piece:
                    s   = self._encode(game)
                    col = self.choose_move(game, greedy=False)
                    game.make_move(col)
                    ns  = self._encode(game)
                    if game.winner == self.piece:
                        self.replay.append((s, col, 1.0, ns, True))
                        self._learn(); result = 1; break
                    elif game.game_over:
                        self.replay.append((s, col, 0.5, ns, True))
                        self._learn(); break
                    else:
                        memory.append((s, col, ns)); 
                else:
                    col = opponent.choose_move(game)
                    game.make_move(col)
                    if game.winner == self.opp_piece:
                        result = -1
                        for i, (s, c, ns) in enumerate(memory):
                            r = -1.0 if i == len(memory)-1 else -0.1
                            self.replay.append((s, c, r, ns, i == len(memory)-1))
                            self._learn()
                        break
                    elif game.game_over:
                        for s, c, ns in memory:
                            self.replay.append((s, c, 0.5, ns, True))
                            self._learn()
                        break

            self.win_history.append(result)
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            if (ep+1) % 5000 == 0:
                wins = self.win_history[-1000:].count(1)
                print(f"  ep {ep+1}/{num_episodes} | eps={self.eps:.4f} | win%={wins/10:.1f}%")
        print("[DQN C4] Done.")

    def save(self, path="results/dqn_c4.pt"):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "history": self.win_history, "eps": self.eps, "steps": self.steps}, path)
        print(f"DQN training saved to {path}")

    def load(self, path="results/dqn_c4.pt"):
        if not os.path.exists(path): print(f"No file at {path}"); return
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])
        self.win_history = ck.get("history", [])
        self.eps   = ck.get("eps", self.eps_min)
        self.steps = ck.get("steps", 0)
        print(f"DQN training loaded from: {path}")


# ---------------------------------------------------------------------------
# Random Agent (baseline for evaluation and training, for tabular and dqn)
# ---------------------------------------------------------------------------
class RandomAgent:
    def choose_move(self, game):
        valid = game.get_valid_moves()
        return random.choice(valid) if valid else None


# ----------------------------------------------------------------------
# Connect 4 UI using Pygame
# ----------------------------------------------------------------------

class Connect4UI:

    def __init__(self, agent, agent_name, agent2=None, agent2_name=None, ai_delay=600):
        pygame.init()
        self.screen     = pygame.display.set_mode((width, height))
        if agent2 is not None:
            pygame.display.set_caption(f"Connect 4  {agent_name} (Red) vs {agent2_name or 'Agent2'} (Yellow)")
        else:
            pygame.display.set_caption(f"Connect 4  Human (Red) vs {agent_name} (Yellow)")
        self.clock      = pygame.time.Clock()
        self.font       = pygame.font.SysFont("monospace", 20)

        self.agent_name  = agent_name
        self.agent2_name = agent2_name or "Agent 2"
        self.ai_delay    = ai_delay
        self.watch_mode  = agent2 is not None

        # watch mode: both players are agents
        # play mode: human is PLAYER1_PIECE, agent is PLAYER2_PIECE
        if self.watch_mode:
            self.agents = {PLAYER1_PIECE: agent2, PLAYER2_PIECE: agent}
            # agent2 is Red (goes first), agent is Yellow (goes second)
            pygame.display.set_caption(
                f"Connect 4  {agent2_name or 'Agent2'} (Red) vs {agent_name} (Yellow)")
        else:
            self.agents = {PLAYER1_PIECE: None, PLAYER2_PIECE: agent}

        self.game    = Connect4Game()
        self.scores  = {PLAYER1_PIECE: 0, PLAYER2_PIECE: 0, 0: 0}
        self.hover   = None
        self.status  = ""
        self._last_ai = 0

        bw, bh = 100, 32
        self.btn = pygame.Rect(width - bw - 16,
                               PANEL_Y + PANEL_H//2 - bh//2, bw, bh)
        
    # Draw the blue board and all pieces/holes.
    # Uses the same coordinate system as the game logic,
    # where row 0 is the bottom and row 5 is the top, so pieces fall visually in the correct direction.
    def draw_board(self):
        pygame.draw.rect(self.screen, BLUE,
                         (0, CELL_SIZE, width, ROWS * CELL_SIZE))

        # Draw every cell, holes and pieces using the SAME cy formula
        for c in range(COLS):
            for r in range(ROWS):
                cx = int(c * CELL_SIZE + CELL_SIZE / 2)
                # row 0 => largest cy (bottom of screen), row 5 => smallest cy (top)
                cy = height - PANEL_H - int(r * CELL_SIZE + CELL_SIZE / 2)

                v = self.game.board[r][c]
                if v == PLAYER1_PIECE:
                    pygame.draw.circle(self.screen, RED, (cx, cy), RADIUS)
                elif v == PLAYER2_PIECE:
                    pygame.draw.circle(self.screen, YELLOW, (cx, cy), RADIUS)
                else:
                    # Empty cell, black hole punched through the blue board
                    pygame.draw.circle(self.screen, BLACK, (cx, cy), RADIUS)

        # Green ring on winning cells
        for (r, c) in self.game.winning_cells:
            cx = int(c * CELL_SIZE + CELL_SIZE / 2)
            cy = height - PANEL_H - int(r * CELL_SIZE + CELL_SIZE / 2)
            pygame.draw.circle(self.screen, GREEN, (cx, cy), RADIUS, 5)

    # Draw a preview piece at the top, following the mouse hover, if it's a valid move for the current player.
    def _draw_preview(self):
        p = self.game.current_player
        if self.agents[p] is not None or self.hover is None:
            return
        if self.hover in self.game.get_valid_moves():
            color = RED if p == PLAYER1_PIECE else YELLOW
            pygame.draw.circle(self.screen, color,
                               (self.hover * CELL_SIZE + CELL_SIZE//2, CELL_SIZE//2),
                               RADIUS)

    def _draw_panel(self):
        pygame.draw.rect(self.screen, GREY, (0, PANEL_Y, width, PANEL_H))
        cy = PANEL_Y + PANEL_H // 2

        sc = self.font.render(
            f"P1:{self.scores[PLAYER1_PIECE]}  "
            f"P2:{self.scores[PLAYER2_PIECE]}  "
            f"Draw:{self.scores[0]}", True, WHITE)
        self.screen.blit(sc, (16, cy - sc.get_height()//2))

        msg = self.font.render(self.status, True, WHITE)
        self.screen.blit(msg, (width//2 - msg.get_width()//2,
                                cy - msg.get_height()//2))

        col = GREY3 if self.btn.collidepoint(pygame.mouse.get_pos()) else GREY2
        pygame.draw.rect(self.screen, col, self.btn, border_radius=8)
        lbl = self.font.render("Restart", True, WHITE)
        self.screen.blit(lbl, (self.btn.centerx - lbl.get_width()//2,
                                self.btn.centery - lbl.get_height()//2))

    def _render(self):
        self.screen.fill(BLACK)
        self._draw_preview()
        self.draw_board()
        self._draw_panel()
        pygame.display.update()

    def _restart(self):
        key = self.game.winner if self.game.winner else 0
        self.scores[key] += 1
        self.game.reset()
        self._update_status()
        self._last_ai = pygame.time.get_ticks()

    def _update_status(self):
        if self.game.game_over:
            if self.game.winner == PLAYER1_PIECE:
                name = self.agent2_name if self.watch_mode else "You"
                self.status = f"{name} wins!"
            elif self.game.winner == PLAYER2_PIECE:
                self.status = f"{self.agent_name} wins!"
            else:
                self.status = "Draw!"
        else:
            if self.game.current_player == PLAYER1_PIECE:
                name = self.agent2_name if self.watch_mode else "Your turn (Red)"
                self.status = f"{name} turn" if self.watch_mode else "Your turn (Red)"
            else:
                self.status = f"{self.agent_name} turn"

    def run(self):
        self._update_status()
        self._last_ai = pygame.time.get_ticks()

        while True:
            self.clock.tick(30)
            self._render()

            current_agent = self.agents[self.game.current_player]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEMOTION:
                    self.hover = int(math.floor(event.pos[0] / CELL_SIZE))
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if self.btn.collidepoint(mx, my):
                        self._restart(); continue
                    # only allow human click in play mode
                    if (not self.watch_mode
                            and not self.game.game_over
                            and current_agent is None
                            and my < PANEL_Y):
                        col = int(math.floor(mx / CELL_SIZE))
                        if self.game.make_move(col):
                            self._update_status()
                            self._last_ai = pygame.time.get_ticks()

            # AI move, fires for both agents in watch mode
            if (not self.game.game_over
                    and current_agent is not None
                    and pygame.time.get_ticks() - self._last_ai >= self.ai_delay):
                col = current_agent.choose_move(self.game)
                if col is not None:
                    self.game.make_move(col)
                self._update_status()
                self._last_ai = pygame.time.get_ticks()


# ---------------------------------------------------------------------------
# Main: parse command-line arguments, set up agents, run UI or evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Connect 4")

    parser.add_argument("--agent",  default="minimax_ab",
                        choices=["minimax","minimax_ab","qlearn","dqn","default","random"])
    parser.add_argument("--depth",  type=int, default=5)
    parser.add_argument("--train",  default=None, choices=["qlearn","dqn"])
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--games",  type=int, default=100)
    parser.add_argument("--load",   type=str, default=None)
    parser.add_argument("--watch", default=None,
                    choices=["minimax", "minimax_ab", "qlearn", "dqn", "default", "random"],
                    help="Second agent to watch play against --agent")
    
    args = parser.parse_args()

    random_agent  = RandomAgent()
    default_agent = DefaultOpponent(PLAYER2_PIECE)

    # Set up the main agent based on command-line arguments. 
    if args.agent in ("minimax", "minimax_ab"):
        agent = MinimaxAgent(PLAYER2_PIECE, depth=args.depth,
                             use_alpha_beta=(args.agent == "minimax_ab"))
    elif args.agent == "qlearn":
        agent = QLearningAgent(PLAYER2_PIECE, PLAYER1_PIECE)
        agent.load(args.load or "results/qlearn_c4.pkl")
    elif args.agent == "dqn":
        agent = DQNAgent(PLAYER2_PIECE, PLAYER1_PIECE)
        agent.load(args.load or "results/dqn_c4.pt")
    elif args.agent == "random":
        agent = random_agent
    else:
        agent = default_agent

    # Training mode: 
    # train the specified agent against a random opponent.
    if args.train == "qlearn":
        agent = QLearningAgent(PLAYER2_PIECE, PLAYER1_PIECE)
        agent.train(args.episodes, random_agent)
        agent.save()
        sys.exit()   
    elif args.train == "dqn":
        agent = DQNAgent(PLAYER2_PIECE, PLAYER1_PIECE)
        agent.train(args.episodes, random_agent)
        agent.save()
        sys.exit()   

    # Watch mode: 
    # set up a second agent to play against the main agent,
    # runs the UI to watch play.
    if args.watch:
        if args.watch in ("minimax", "minimax_ab"):
            agent2 = MinimaxAgent(PLAYER1_PIECE, depth=args.depth,
                                use_alpha_beta=(args.watch == "minimax_ab"))
        elif args.watch == "qlearn":
            agent2 = QLearningAgent(PLAYER1_PIECE, PLAYER2_PIECE)
            agent2.load(args.load or "results/qlearn_c4.pkl")
        elif args.watch == "dqn":
            agent2 = DQNAgent(PLAYER1_PIECE, PLAYER2_PIECE)
            agent2.load(args.load or "results/dqn_c4.pt")
        elif args.watch == "random":
            agent2 = RandomAgent()
        else:
            agent2 = DefaultOpponent(PLAYER1_PIECE)
        ui = Connect4UI(agent=agent, agent_name=args.agent, agent2=agent2, agent2_name=args.watch)
    else:
        ui = Connect4UI(agent=agent, agent_name=args.agent)
    ui.run()