
import torch

from model_small import NeuralNetwork as NN

from MCTS import MCTS

import utils



def train():
    nn = NN(input_layers=3, board_size=utils.board_size, learning_rate=0.1)

    tree = MCTS(board_size=8, net=nn)
    nn.adjust_lr(1e-3)
    record = []
    game_time = 3600

    while True:
        game_record, eval, steps = tree.game()

        print(game_record)
        print("+++++")

        if len(game_record) % 2 == 1:
            print("game {} completed, black win, "
                  "this game length is {}".format(game_time, len(game_record)))
        else:
            print("game {} completed, white win, "
                  "this game length is {}".format(game_time, len(game_record)))
        #
        # 生成训练数据
        train_data = utils.generate_training_data(game_record=game_record, board_size=utils.board_size)




if "__main__" == __name__:
    train()















