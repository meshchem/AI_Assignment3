import pygame
import sys
import numpy as np

# colours
BG         = pygame.Color("floralwhite")
LINE       = pygame.Color("plum")
X_COL      = pygame.Color("lightcoral")
O_COL      = pygame.Color("cornflowerblue")
WIN_COL    = pygame.Color("palegreen")
PANEL_BG   = pygame.Color("lavender")
BUTTON     = pygame.Color("thistle")
BUTTON_HOV = pygame.Color("plum")
TEXT       = pygame.Color("purple")

CELL     = 160
BOARD_PX = CELL * 3
PANEL_H  = 80
WIDTH    = BOARD_PX
HEIGHT   = BOARD_PX + PANEL_H
RADIUS   = CELL // 2 - 20

pygame.font.init()
F_SM = pygame.font.SysFont("Georgia", 20)


# game logic

class TicTacToeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.winning_cells = []

    def get_valid_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def make_move(self, row, col):
        if self.board[row, col] != 0 or self.game_over:
            return False
        self.board[row, col] = self.current_player
        self._check_over()
        if not self.game_over:
            self.current_player *= -1
        return True

    def _check_over(self):
        b = self.board
        rows  = [[(r, c) for c in range(3)] for r in range(3)]
        cols  = [[(r, c) for r in range(3)] for c in range(3)]
        diags = [[(i, i) for i in range(3)], [(i, 2-i) for i in range(3)]]
        for cells in rows + cols + diags:
            if abs(sum(b[r, c] for r, c in cells)) == 3:
                self.winner = self.current_player
                self.winning_cells = cells
                self.game_over = True
                return
        if not self.get_valid_moves():
            self.game_over = True

    def clone(self):
        g = TicTacToeGame()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.winner = self.winner
        g.game_over = self.game_over
        g.winning_cells = list(self.winning_cells)
        return g

    def get_state(self):
        return tuple(self.board.flatten())


# default opponent

class DefaultOpponent:
    def __init__(self, player):
        self.player = player

    def choose_move(self, game):
        moves = game.get_valid_moves()
        for move in moves:
            g = game.clone(); g.make_move(*move)
            if g.winner == self.player:
                return move
        opp = -self.player
        for move in moves:
            g = game.clone(); g.current_player = opp; g.make_move(*move)
            if g.winner == opp:
                return move
        for m in [(1,1),(0,0),(0,2),(2,0),(2,2),(0,1),(1,0),(1,2),(2,1)]:
            if m in moves:
                return m
        return moves[0]


# UI

class TicTacToeUI:
    def __init__(self, player1_agent=None, player2_agent=None, ai_delay=600):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tic Tac Toe")
        self.clock  = pygame.time.Clock()
        self.agents = {1: player1_agent, -1: player2_agent}
        self.ai_delay = ai_delay
        self.game   = TicTacToeGame()
        self.scores = {1: 0, -1: 0, 0: 0}
        self.status = ""
        self._last_ai = 0
        bw, bh = 100, 32
        self.btn = pygame.Rect(WIDTH - bw - 16, BOARD_PX + PANEL_H//2 - bh//2, bw, bh)

    def _render(self):
        self.screen.fill(BG)
        for i in range(1, 3):
            pygame.draw.line(self.screen, LINE, (i*CELL, 0), (i*CELL, BOARD_PX), 5)
            pygame.draw.line(self.screen, LINE, (0, i*CELL), (BOARD_PX, i*CELL), 5)
        for r in range(3):
            for c in range(3):
                cx, cy = c*CELL + CELL//2, r*CELL + CELL//2
                v = self.game.board[r, c]
                if v == 1:
                    off = RADIUS - 10
                    pygame.draw.line(self.screen, X_COL, (cx-off, cy-off), (cx+off, cy+off), 10)
                    pygame.draw.line(self.screen, X_COL, (cx+off, cy-off), (cx-off, cy+off), 10)
                elif v == -1:
                    pygame.draw.circle(self.screen, O_COL, (cx, cy), RADIUS, 8)
        if self.game.winning_cells:
            c0, c1 = self.game.winning_cells[0], self.game.winning_cells[-1]
            pygame.draw.line(self.screen, WIN_COL,
                             (c0[1]*CELL+CELL//2, c0[0]*CELL+CELL//2),
                             (c1[1]*CELL+CELL//2, c1[0]*CELL+CELL//2), 8)

        # panel
        pygame.draw.rect(self.screen, PANEL_BG, (0, BOARD_PX, WIDTH, PANEL_H))
        cy = BOARD_PX + PANEL_H//2

        # scores on the left
        scores_surf = F_SM.render(f"X:{self.scores[1]}  O:{self.scores[-1]}  Draw:{self.scores[0]}", True, TEXT)
        self.screen.blit(scores_surf, (16, cy - scores_surf.get_height()//2))

        # status in the centre
        msg = F_SM.render(self.status, True, TEXT)
        self.screen.blit(msg, (WIDTH//2 - msg.get_width()//2, cy - msg.get_height()//2))

        # restart button on the right
        col = BUTTON_HOV if self.btn.collidepoint(pygame.mouse.get_pos()) else BUTTON
        pygame.draw.rect(self.screen, col, self.btn, border_radius=8)
        lbl = F_SM.render("Restart", True, TEXT)
        self.screen.blit(lbl, (self.btn.centerx - lbl.get_width()//2, self.btn.centery - lbl.get_height()//2))

        pygame.display.flip()

    def _update_status(self):
        if self.game.game_over:
            self.status = "X wins!" if self.game.winner == 1 else "O wins!" if self.game.winner == -1 else "Draw!"
        else:
            p = "X" if self.game.current_player == 1 else "O"
            who = type(self.agents[self.game.current_player]).__name__ if self.agents[self.game.current_player] else "Human"
            self.status = f"{p}'s turn"

    def _restart(self):
        self.scores[self.game.winner if self.game.winner else 0] += 1
        self.game.reset()
        self._update_status()
        self._last_ai = pygame.time.get_ticks()

    def run(self):
        self._update_status()
        self._last_ai = pygame.time.get_ticks()
        while True:
            self.clock.tick(30)
            self._render()
            agent = self.agents[self.game.current_player]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if self.btn.collidepoint(mx, my):
                        self._restart(); continue
                    if not self.game.game_over and agent is None and my < BOARD_PX:
                        if self.game.make_move(my // CELL, mx // CELL):
                            self._update_status()
            if not self.game.game_over and agent is not None:
                if pygame.time.get_ticks() - self._last_ai >= self.ai_delay:
                    self.game.make_move(*agent.choose_move(self.game))
                    self._update_status()
                    self._last_ai = pygame.time.get_ticks()


if __name__ == "__main__":
    ui = TicTacToeUI(player1_agent=None, player2_agent=DefaultOpponent(-1))
    ui.run()