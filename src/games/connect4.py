# adapted from https://github.com/KeithGalli/Connect4-Python/tree/master

import numpy as np
import pygame
import sys
import math

# colours
BLUE   = pygame.Color("steelblue")
BLACK  = pygame.Color("black")
RED    = pygame.Color("red")
YELLOW = pygame.Color("yellow")
GREEN  = pygame.Color("lawngreen")
GREY   = pygame.Color("gray20")
GREY2  = pygame.Color("gray35")
GREY3  = pygame.Color("gray50")
WHITE  = pygame.Color("white")

ROWS    = 6
COLS = 7
CELL_SIZE   = 100
RADIUS       = int(CELL_SIZE / 2 - 5)

width   = COLS * CELL_SIZE
height  = (ROWS + 1) * CELL_SIZE + 80   # +80 for panel
PANEL_H = 80
PANEL_Y = height - PANEL_H

pygame.font.init()
myfont = pygame.font.SysFont("monospace", 20)


# game logic

class Connect4Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((ROWS, COLS))
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.winning_cells = []

    def is_valid_location(self, col):
        return self.board[ROWS - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(ROWS):
            if self.board[r][col] == 0:
                return r

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def get_valid_moves(self):
        return [c for c in range(COLS) if self.is_valid_location(c)]

    def make_move(self, col):
        if not self.is_valid_location(col) or self.game_over:
            return False
        row = self.get_next_open_row(col)
        self.drop_piece(row, col, self.current_player)
        if self.winning_move(self.current_player):
            self.winner = self.current_player
            self.game_over = True
        elif not self.get_valid_moves():
            self.game_over = True
        else:
            self.current_player *= -1
        return True

    def winning_move(self, piece):
        # horizontal
        for c in range(COLS - 3):
            for r in range(ROWS):
                if all(self.board[r][c+i] == piece for i in range(4)):
                    self.winning_cells = [(r, c+i) for i in range(4)]
                    return True
        # vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                if all(self.board[r+i][c] == piece for i in range(4)):
                    self.winning_cells = [(r+i, c) for i in range(4)]
                    return True
        # positive diagonal
        for c in range(COLS - 3):
            for r in range(ROWS - 3):
                if all(self.board[r+i][c+i] == piece for i in range(4)):
                    self.winning_cells = [(r+i, c+i) for i in range(4)]
                    return True
        # negative diagonal
        for c in range(COLS - 3):
            for r in range(3, ROWS):
                if all(self.board[r-i][c+i] == piece for i in range(4)):
                    self.winning_cells = [(r-i, c+i) for i in range(4)]
                    return True
        return False

    def print_board(self):
        print(np.flip(self.board, 0))

    def clone(self):
        g = Connect4Game()
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
        for col in moves:
            g = game.clone(); g.make_move(col)
            if g.winner == self.player:
                return col
        opp = -self.player
        for col in moves:
            g = game.clone(); g.current_player = opp; g.make_move(col)
            if g.winner == opp:
                return col
        for col in [3, 2, 4, 1, 5, 0, 6]:
            if col in moves:
                return col
        return moves[0]


# pygame UI

class Connect4UI:

    def __init__(self, player1=None, player2=None, ai_delay=600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Connect 4")
        self.clock  = pygame.time.Clock()
        self.agents = {1: player1, -1: player2}
        self.ai_delay = ai_delay
        self.game   = Connect4Game()
        self.scores = {1: 0, -1: 0, 0: 0}
        self.status = ""
        self.hover  = None
        self._last_ai = 0
        bw, bh = 100, 32
        self.btn = pygame.Rect(width - bw - 16, PANEL_Y + PANEL_H//2 - bh//2, bw, bh)

    def draw_board(self):
        for c in range(COLS):
            for r in range(ROWS):
                pygame.draw.rect(self.screen, BLUE,
                    (c*CELL_SIZE, r*CELL_SIZE + CELL_SIZE, CELL_SIZE, CELL_SIZE))
                pygame.draw.circle(self.screen, BLACK,
                    (int(c*CELL_SIZE + CELL_SIZE/2), int(r*CELL_SIZE + CELL_SIZE + CELL_SIZE/2)), RADIUS)
        for c in range(COLS):
            for r in range(ROWS):
                v = self.game.board[r][c]
                if v == 1:
                    pygame.draw.circle(self.screen, RED,
                        (int(c*CELL_SIZE + CELL_SIZE/2), height - PANEL_H - int(r*CELL_SIZE + CELL_SIZE/2)), RADIUS)
                elif v == -1:
                    pygame.draw.circle(self.screen, YELLOW,
                        (int(c*CELL_SIZE + CELL_SIZE/2), height - PANEL_H - int(r*CELL_SIZE + CELL_SIZE/2)), RADIUS)
        for (r, c) in self.game.winning_cells:
            pygame.draw.circle(self.screen, GREEN,
                (int(c*CELL_SIZE + CELL_SIZE/2), height - PANEL_H - int(r*CELL_SIZE + CELL_SIZE/2)), RADIUS, 5)

    def _draw_preview(self):
        p = self.game.current_player
        if self.agents[p] is not None or self.hover is None:
            return
        if self.hover in self.game.get_valid_moves():
            color = RED if p == 1 else YELLOW
            pygame.draw.circle(self.screen, color, (self.hover * CELL_SIZE + CELL_SIZE//2, CELL_SIZE//2), RADIUS)

    def _draw_panel(self):
        pygame.draw.rect(self.screen, GREY, (0, PANEL_Y, width, PANEL_H))
        cy = PANEL_Y + PANEL_H // 2

        scores_surf = myfont.render(f"P1:{self.scores[1]}  P2:{self.scores[-1]}  Draw:{self.scores[0]}", True, WHITE)
        self.screen.blit(scores_surf, (16, cy - scores_surf.get_height()//2))

        msg = myfont.render(self.status, True, WHITE)
        self.screen.blit(msg, (width//2 - msg.get_width()//2, cy - msg.get_height()//2))

        col = GREY3 if self.btn.collidepoint(pygame.mouse.get_pos()) else GREY2
        pygame.draw.rect(self.screen, col, self.btn, border_radius=8)
        lbl = myfont.render("Restart", True, WHITE)
        self.screen.blit(lbl, (self.btn.centerx - lbl.get_width()//2, self.btn.centery - lbl.get_height()//2))

    def _render(self):
        self.screen.fill(BLACK)
        self._draw_preview()
        self.draw_board()
        self._draw_panel()
        pygame.display.update()

    def _update_status(self):
        if self.game.game_over:
            self.status = "P1 wins!" if self.game.winner == 1 else "P2 wins!" if self.game.winner == -1 else "Draw!"
        else:
            p = "P1" if self.game.current_player == 1 else "P2"
            who = type(self.agents[self.game.current_player]).__name__ if self.agents[self.game.current_player] else "Human"
            self.status = f"{p}'s turn [{who}]"

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
                if event.type == pygame.MOUSEMOTION:
                    self.hover = int(math.floor(event.pos[0] / CELL_SIZE))
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if self.btn.collidepoint(mx, my):
                        self._restart(); continue
                    if not self.game.game_over and agent is None and my < PANEL_Y:
                        col = int(math.floor(mx / CELL_SIZE))
                        if self.game.make_move(col):
                            self._update_status()
            if not self.game.game_over and agent is not None:
                if pygame.time.get_ticks() - self._last_ai >= self.ai_delay:
                    self.game.make_move(agent.choose_move(self.game))
                    self._update_status()
                    self._last_ai = pygame.time.get_ticks()


if __name__ == "__main__":
    ui = Connect4UI(player1=None, player2=DefaultOpponent(-1))
    ui.run()