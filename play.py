import pygame
import game

import torch
import numpy as np

from MCTS import MCTS


ROWS = 9
SIDE = 30

SCREEN_WIDTH = ROWS * SIDE
SCREEN_HEIGHT = ROWS * SIDE

EMPTY = -1  # 0
BLACK = (0, 0, 0)  # 1
WHITE = (255, 255, 255)  # -1
DIRE = [(1, 0), (0, 1), (1, 1), (1, -1)]



class Gobang(game.Game):
    def __init__(self, title, size, fps=15):
        super(Gobang, self).__init__(title, size, fps)
        self.board = [[EMPTY for i in range(ROWS)] for j in range(ROWS)]
        self.select = (-1, -1)
        self.black = True
        self.draw_board()
        self.bind_click(1, self.click)

        # AI
        Net = torch.load("./model_test1/model_2.pkl")
        self.tree = MCTS(board_size=ROWS, net=Net)
        self.record = np.zeros([ROWS, ROWS])  # 空棋盘
        # self.record, self.game_continue = self.tree.interact_game_init()
        self.tree.renew()
        _, _ = self.tree.simulation()
        self.game_continue = True
        print(self.record)
        print("========")
        self.AI_i = -1
        self.AI_j = -1


    def click(self, x, y):
        if self.end:
            return
        i, j = y // SIDE, x // SIDE  # 计算出坐标
        # print(self.board)
        if self.board[i][j] != EMPTY: # 判断该点是否为空
            return

        if self.black:
            self.board[i][j] = BLACK  # 人执黑
            self.draw_chess(self.board[i][j], i, j)
            position = (int(i), int(j))
            tmp_record, self.game_continue = self.tree.human(position)
            self.record, self.game_continue, action = self.tree.machine(position, self.game_continue, tmp_record)
            self.AI_i = action[0]
            self.AI_j = action[1]
            # print(self.record)
            # print(self.AI_i)
            # print(self.AI_j)
            # print("-------")

        # else:
        #     self.board[self.AI_i][self.AI_j] = WHITE
            # i = self.AI_i
            # j = self.AI_j

        self.board[self.AI_i][self.AI_j] = WHITE
        self.draw_chess(self.board[self.AI_i][self.AI_j], self.AI_i, self.AI_j)
        # self.board[i][j] = BLACK if self.black else WHITE

        # self.black = not self.black

        # =================
        chess = self.check_win()
        if chess:
            self.end = True
            i, j = chess[0]
            winer = "Black"
            if self.board[i][j] == WHITE:
                winer = "White"
            pygame.display.set_caption("五子棋 ---- %s win!" % (winer))
            for c in chess:
                i, j = c
                self.draw_chess((100, 255, 255), i, j)
                self.timer.tick(5)

    def check_win(self):
        for i in range(ROWS):
            for j in range(ROWS):
                win = self.check_chess(i, j)
                if win:
                    return win
        return None

    def check_chess(self, i, j):
        if self.board[i][j] == EMPTY:
            return None
        color = self.board[i][j]
        for dire in DIRE:
            x, y = i, j
            chess = []
            while self.board[x][y] == color:
                chess.append((x, y))
                x, y = x + dire[0], y + dire[1]
                if x < 0 or y < 0 or x >= ROWS or y >= ROWS:
                    break
            if len(chess) >= 5:
                return chess
        return None

    def draw_chess(self, color, i, j):
        center = (j * SIDE + SIDE // 2, i * SIDE + SIDE // 2)
        pygame.draw.circle(self.screen, color, center, SIDE // 2 - 2)
        pygame.display.update(pygame.Rect(j * SIDE, i * SIDE, SIDE, SIDE))

    def draw_board(self):
        self.screen.fill((139, 87, 66))
        for i in range(ROWS):
            start = (i * SIDE + SIDE // 2, SIDE // 2)
            end = (i * SIDE + SIDE // 2, ROWS * SIDE - SIDE // 2)
            pygame.draw.line(self.screen, 0x000000, start, end)
            start = (SIDE // 2, i * SIDE + SIDE // 2)
            end = (ROWS * SIDE - SIDE // 2, i * SIDE + SIDE // 2)
            pygame.draw.line(self.screen, 0x000000, start, end)
        center = ((ROWS // 2) * SIDE + SIDE // 2, (ROWS // 2) * SIDE + SIDE // 2)
        pygame.draw.circle(self.screen, (0, 0, 0), center, 4)
        pygame.display.update()


if __name__ == '__main__':
    print('''
    Welcome to 五子棋!
    click LEFT MOUSE BUTTON to play game.
    ''')
    gobang = Gobang("五子棋", (SCREEN_WIDTH, SCREEN_HEIGHT))
    gobang.run()
