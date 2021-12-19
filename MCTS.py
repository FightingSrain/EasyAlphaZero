import numpy as np
import utils
import sys
import time

from five_stone_game import main_process as five_stone_game

distrib_calculater = utils.distribution_calculater(size=8)


class node:
    def __init__(self, parent, player):
        self.parent = parent
        self.counter = 0.0
        self.child = {}
        self.node_player = player

    def add_child(self, action, priorProb):
        action_name = utils.move_to_str(action)
        self.child[action_name] = edge(action=action, parent_node=self, priorProb=priorProb)

    def get_child(self, action):
        child_node, _ = self.child[action].get_child()
        return child_node

    def eval_or_not(self):
        return len(self.child) == 0

    def backup(self, v):
        self.counter += 1
        # 递归回溯
        if self.parent:
            self.parent.backup(v)

    def get_dsitribution(self, train=True):
        # 遍历孩子节点，将孩子节点次数
        for key in self.child.keys():
            distrib_calculater.push(key, self.child[key].counter)
        return distrib_calculater.get(train=train)

    # 用于根据UCB公式选取node
    def UCB_sim(self):
        UCB_max = -sys.maxsize
        UCB_max_key = None
        for key in self.child.keys():
            if self.child[key].UCB_value() > UCB_max:
                UCB_max_key = key
                UCB_max = self.child[key].UCB_value()
        cur_node, expend = self.child[UCB_max_key].get_child()
        return cur_node, expend, self.child[UCB_max_key].action


class edge:
    def __init__(self, action, parent_node, priorProb):
        self.action = action
        self.counter = 1.0  # 走的步数
        self.parent_node = parent_node
        self.priorProb = priorProb
        self.child_node = None

        self.action_value = 0.0

    def backup(self, v):
        self.action_value += v
        self.counter += 1
        self.parent_node.backup(-v)

    def get_child(self):
        if self.child_node is None:
            self.counter += 1
            self.child_node = node(self, -self.parent_node.node_player)
            return self.child_node, True
        else:
            self.counter += 1
            return self.child_node, False

    def UCB_value(self):  # 计算当前的UCB value
        # Q 是节点价值
        if self.action_value:
            Q = self.action_value / self.counter
        else:
            Q = 0
        # utils.Cpuct * self.priorProb * np.sqrt(self.parent_node.counter) / (1 + self.counter)
        # utils.Cpuct 为参数调节探索力度
        # self.parent_node.counter 为父节点访问次数
        # self.counter 当前节点访问次数
        return Q + utils.Cpuct * self.priorProb * np.sqrt(self.parent_node.counter) / (1 + self.counter)


class MCTS:
    def __init__(self, board_size=8, simulation_per_step=400, net=None):
        self.board_size = board_size
        self.s_per_step = simulation_per_step  # 模拟400步

        self.model = net

        self.cur_node = node(None, 1)

        # 落子执行进程， 模拟进程
        self.game_process = five_stone_game(board_size=board_size)
        self.simulate_prcess = five_stone_game(board_size=board_size)

    def renew(self):
        self.cur_node = node(None, 1)  # 重新初始化节点
        self.game_process.renew()  # 重新初始化进程

    def MCTS_step(self, action):
        next_node = self.cur_node.get_child(action)  # 根据动作扩展子节点
        next_node.parent = None
        return next_node

    # 模拟 模拟直到一局游戏结束
    def simulation(self):
        eval_counter = 0
        step_per_simulate = 0

        for _ in range(self.s_per_step):
            expand = False
            game_continue = True

            cur_node = self.cur_node

            self.simulate_prcess.simulate_reset(self.game_process.current_board_state(True))
            state = self.simulate_prcess.current_board_state()  # [8, 8]
            # print(state)

            while game_continue and not expand:
                if cur_node.eval_or_not():
                    # self.simulate_prcess.which_player() 当前玩家编号
                    state_prob, _ = self.model.eval(
                        utils.transfer_to_input(state, self.simulate_prcess.which_player(), self.board_size))
                    valid_move = utils.valid_move(state)  # 判断当前状态中可以走的点（坐标）
                    eval_counter += 1
                    # 遍历可行位置点，为当前cur_node添加孩子节点
                    for move in valid_move:
                        cur_node.add_child(action=move, priorProb=state_prob[0, move[0] * self.board_size + move[1]])
                cur_node, expand, action = cur_node.UCB_sim()
                # print(action)
                # print("=====")
                game_continue, state = self.simulate_prcess.step(action)
                print(state)
                print("=====")
                step_per_simulate += 1

            if not game_continue:
                cur_node.backup(1)
            elif expand:
                _, state_v = self.model.eval(
                    utils.transfer_to_input(state, self.simulate_prcess.which_player(), self.board_size))
                cur_node.backup(state_v)
        return eval_counter / self.s_per_step, step_per_simulate / self.s_per_step

    def game(self, train=True):  # 主程序
        game_coutinue = True
        game_record = []

        begin_time = int(time.time())  # 记录开始时间
        step = 1
        total_eval = 0
        total_step = 0

        while game_coutinue:
            begin_time1 = int(time.time())

            avg_eval, avg_s_per_step = self.simulation()  # 模拟
            action, distribution = self.cur_node.get_dsitribution(train=train)
            game_coutinue, state = self.game_process.step(utils.str_to_move(action))

            self.cur_node = self.MCTS_step(action=action)

            # 记录每一步的选择走的位置的 概率分布 和 走的动作
            game_record.append({"distirbution": distribution, "action": action})

            end_time1 = int(time.time())
            print("step:{},"
                  "cost:{}s, "
                  "total time:{}:{} "
                  "Avg eval:{}, "
                  "Aver step:{}".format(step,
                                        end_time1 - begin_time1,
                                        int((end_time1 - begin_time) / 60),
                                        (end_time1 - begin_time) % 60,
                                        avg_eval,
                                        avg_s_per_step),
                  end="\r")
            step += 1

        self.renew()  # 重新构造MCT

        end_time = int(time.time())
        minute = int((end_time - begin_time) / 60)
        second = (end_time - begin_time) % 60
        print("In last game, we cost {}:{}".format(minute, second), end="\n")

        return game_record, total_eval/step, total_step/step


# test
# mcts = MCTS()
# mcts.simulation()
