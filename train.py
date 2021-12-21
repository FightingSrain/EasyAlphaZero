import torch

from model_small import NeuralNetwork as NN

from MCTS import MCTS

import utils
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train():
    nn = NN(input_layers=3, board_size=utils.board_size, learning_rate=0.1)

    tree = MCTS(board_size=utils.board_size, net=nn)  # 初始化MCTS
    nn.adjust_lr(1e-3)  # 调整学习率
    stack = utils.random_stack()

    record = []
    game_time = 1  # 3600

    while game_time < 3600:
        game_record, eval, steps = tree.game()

        print(game_time)
        print("========")

        if len(game_record) % 2 == 1:
            print("game {} completed, black win, "
                  "this game length is {}".format(game_time, len(game_record)))
        else:
            print("game {} completed, white win, "
                  "this game length is {}".format(game_time, len(game_record)))
        print("The average eval:{}, the average steps:{}".format(eval, steps))
        #
        # 生成训练数据
        train_data = utils.generate_training_data(game_record=game_record, board_size=utils.board_size)

        for i in range(len(train_data)):
            stack.push(train_data[i])
        my_loader = utils.generate_data_loader(stack)

        #
        if game_time % 100 == 0:
            for _ in range(5):
                record.extend(nn.train(my_loader, game_time))

        if game_time % 2 == 0:
            torch.save(nn, "./model_test1/model_{}.pkl".format(game_time))
            # torch.save(nn.model.state_dict(), "./model_test1/model{}_.pth".format(game_time))
        if game_time % 200 == 0:
            test_game_record, _, _ = tree.game(train=False)
            print("We finished a test game at {} game time".format(game_time))

        if game_time % 200 == 0:
            plt.figure()
            plt.plot(record)
            plt.title("cross entropy loss")
            plt.xlabel("step passed")
            plt.ylabel("Loss")
            # plt.savefig("loss record_{}.svg".format(game_time))
            plt.close()

        game_time += 1


if "__main__" == __name__:
    train()
