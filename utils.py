import torch
import torch.utils.data as torch_data

import numpy as np

num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

char2num = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
            "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20}

temperature = 1  # 温度系数
Cpuct = 0.1
board_size = 9

batch = 20
learning_rate = 0.1

class distribution_calculater:
    def __init__(self, size):
        # self.map 存储棋盘每个位置选择的次数
        # self.order 存储棋盘坐标
        self.map = {}  # {'aa': 0, 'ab': 0, 'ac': 0, 'ad': 0, 'ae': 0,..., 'hh': 0}
        self.order = []  # ['aa', 'ab', 'ac', 'ad', 'ae',..., 'hh']

        for i in range(size):
            for j in range(size):
                name = num2char[i] + num2char[j]
                self.order.append(name)
                self.map[name] = 0

    def push(self, key, value):
        self.map[key] = value

    def get(self, train=True):
        result = []  # 存储所有选择好的位置的次数
        choice_pool = []  # 存储已经选择过的位置
        choice_prob = []  # 存储该位置的选择次数
        for key in self.order:  # 遍历棋盘下标
            if self.map[key] != 0:  # 如果选择次数，不为0
                choice_pool.append(key)
                tmp = np.float_power(self.map[key], 1 / temperature)  # 选择概率 pi^a = N(s, a)^(1/temperature)
                choice_prob.append(tmp)
                result.append(tmp)
                self.map[key] = 0  # 存储后重置
            else:
                result.append(0)  # 没选择就是0

        sum_n = sum(result)
        for i in range(len(result)):
            if result[i]:
                result[i] /= sum_n
        choice_prob = [choice / sum_n for choice in choice_prob]

        if train:
            # 0.8的概率选择当前策略网络给出的结果，0.2的概率选择采用迪利克雷分布随机搜索的结果
            # 迪利克雷分布保证所有状态都遍历过
            move = np.random.choice(choice_pool,
                                    p=0.8 * np.array(choice_prob) +
                                      0.2 * np.random.dirichlet(0.3 * np.ones(len(choice_prob))))
        else:
            # 测试时直接选择最有可能的位置
            move = choice_pool[np.argmax(choice_prob)]

        return move, result


# test
# dc = distribution_calculater(8)

# 将位置坐标变为字符标志 （0，0）--> 'aa'
def move_to_str(action):
    return num2char[action[0]] + num2char[action[1]]


def str_to_move(str):
    return np.array([char2num[str[0]], char2num[str[1]]])


# state cur_player board_size 状态合并
def transfer_to_input(state, current_player, board_size):
    if current_player == 1:
        plant3 = np.ones([board_size, board_size]).astype(float)
        plant2 = np.array(state > 0).astype(float)
        plant1 = np.array(state < 0).astype(float)
    else:
        plant3 = np.zeros([board_size, board_size])
        plant2 = np.array(state < 0).astype(float)
        plant1 = np.array(state > 0).astype(float)
    return np.stack([plant1, plant2, plant3])


# 验证算法是否可行
def valid_move(state):
    # np.argwhere(state == 0)
    # 返回state中符合条件的值，即state向量中符合state==0的值
    return list(np.argwhere(state == 0))


def generate_training_data(game_record, board_size):
    board = np.zeros([board_size, board_size])
    data = []
    player = 1
    winner = -1 if len(game_record) % 2 == 0 else 1
    for i in range(len(game_record)):
        step = str_to_move(game_record[i]['action'])
        state = transfer_to_input(board, player, board_size)
        data.append({"state": state, "distribution": game_record[i]['distirbution'], "value": winner})
        board[step[0], step[1]] = player
        player, winner = -player, -winner
    return data


class random_stack:
    def __init__(self, length=1000):
        self.state = []
        self.distribution = []
        self.winner = []
        self.length = length

    def isEmpty(self):
        return len(self.state) == 0

    def push(self, item):
        self.state.append(item["state"])
        self.distribution.append(item["distribution"])
        self.winner.append(item["value"])
        # 超过self.length大小后，新增加的覆盖最开始的
        if len(self.state) >= self.length:
            self.state = self.state[1:]
            self.distribution = self.distribution[1:]
            self.winner = self.winner[1:]

    def seq(self):
        return self.state, self.distribution, self.winner

# 将生成的数据打包，并打乱
def generate_data_loader(stack):
    state, distribution, winner = stack.seq()
    tensor_x = torch.stack(tuple([torch.from_numpy(s) for s in state]))
    tensor_y1 = torch.stack(tuple([torch.Tensor(y1) for y1 in distribution]))
    tensor_y2 = torch.stack(tuple([torch.Tensor([float(y2)]) for y2 in winner]))

    dataset = torch_data.TensorDataset(tensor_x, tensor_y1, tensor_y2)
    my_loader = torch_data.DataLoader(dataset, batch_size=batch, shuffle=True)
    return my_loader






