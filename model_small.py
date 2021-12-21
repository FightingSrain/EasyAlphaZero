import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class share_model(nn.Module):
    def __init__(self, input_layer):
        super(share_model, self).__init__()
        # self.conv1 = nn.Conv2d(input_layer, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(input_layer, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        res = self.relu(self.bn3(self.conv3(x)))
        return res


# test
# mod = share_model(3)
# ins = torch.ones((16, 3, 8, 8))
# res = mod(ins)
# print(res.size())

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.share_model = share_model(input_size)

        self.value_conv1 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=16)
        self.value_fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=256)
        self.value_fc2 = nn.Linear(in_features=256, out_features=1)

        self.policy_conv1 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=16)
        self.policy_fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=output_size * output_size)

        self.bn1_v = nn.BatchNorm2d(16)
        self.bn1_p = nn.BatchNorm2d(16)

    def forward(self, state):
        share_feature = self.share_model(state)

        v = self.value_conv1(share_feature)
        v = F.relu(self.bn1_v(v)).view(-1, 16 * 5 * 5)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        p = self.policy_conv1(share_feature)
        p = F.relu(self.bn1_p(p)).view(-1, 16 * 5 * 5)
        prob = self.policy_fc1(p)
        # prob [b, 8*8]
        # value [b, 1]
        return prob, value


# test
# mod = model(3, 8)
# ins = torch.ones((16, 3, 8, 8))
# p, v = mod(ins)
# print(p.size())
# print(v.size())
class NeuralNetwork:
    def __init__(self, input_layers, board_size, use_cuda=True, learning_rate=0.1):
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = Model(input_size=input_layers, output_size=board_size).cuda().double()
        else:
            self.model = Model(input_size=input_layers, output_size=board_size)

        self.optimier = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.mse = nn.MSELoss()
        self.crossloss = nn.CrossEntropyLoss()

    def train(self, data_loader, game_time):
        self.model.train()
        loss_rocord = []
        for batch_idx, (state, distribution, winner) in enumerate(data_loader):
            tmp = []
            state = Variable(state).double()
            distribution = Variable(distribution).double()
            winner = Variable(winner).double()
            # print(state.size())
            # print(distribution.size())
            # print(winner.size())
            # print("+++++++++")
            if self.use_cuda:
                state, distribution, winner = state.cuda(), distribution.cuda(), winner.cuda()

            prob, value = self.model(state)
            output = F.log_softmax(prob, 1)
            cross_entropy = - torch.mean(torch.sum(distribution * output, 1))  # 交叉熵损失
            mse = F.mse_loss(value, winner)  # 价值损失
            loss = cross_entropy + mse

            self.optimier.zero_grad()
            loss.backward()
            self.optimier.step()

            tmp.append(cross_entropy.data)
            if batch_idx % 10 == 0:
                print("We have played {} games, and batch {}, "
                      "the cross entropy loss is {}, "
                      "the mse loss is {}".format(
                    game_time, batch_idx, cross_entropy.data, mse.data))
                loss_rocord.append(sum(tmp) / len(tmp))  # 平均交叉熵
        return loss_rocord

    def eval(self, state):
        self.model.eval()
        if self.use_cuda:
            state = torch.from_numpy(state).unsqueeze(0).cuda()
        else:
            state = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            prob, value = self.model(state)
        return F.softmax(prob, 1), value

    def adjust_lr(self, lr):
        for group in self.optimier.param_groups:
            group['lr'] = lr
