# 这段代码实现了一个联邦平均（FedAvg）服务器类，用于在联邦学习环境中协调多个用户的训练过程。以下是逐行详细解释：
import torch
import os

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
# torch 是 PyTorch 的核心库。
# os 提供了操作系统相关的功能。
# 从自定义模块 FLAlgorithms 中导入 UserAVG 和 Server 类，
# 以及从 utils.model_utils 中导入 read_data 和 read_user_data 函数。
# numpy 是一个用于科学计算的库。


# Implementation for FedAvg Server
class FedAvg(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)
        # __init__ 方法初始化 FedAvg 服务器的各项参数，并继承了 Server 类。
        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        for i in range(total_users):
            id, train, test = read_user_data(i, data, dataset)
            user = UserAVG(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        # 使用 read_data 函数读取数据集，并为每个用户创建一个 UserAVG 实例，初始化所有用户的数据。
        # self.users 存储了所有用户对象，self.total_train_samples 记录了所有用户的训练样本总数。
        print("Number of users / total users:", num_users, " / " , total_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        # send_grads 方法将当前模型的梯度发送给所有用户。
        assert (self.users is not None and len(self.users) > 0)
        # 对于每个模型参数，如果没有梯度，则用零值替代。
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)
            # 然后将这些梯度分发给每个用户。

    def train(self):
        # train 方法定义了联邦平均算法的训练过程。
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()
            # 在每一轮全局迭代中，首先将全局模型的参数发送给所有用户。
            # Evaluate model each interation
            self.evaluate()
            # 然后评估当前模型。
            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
                # 选择参与本轮训练的用户，并让每个用户进行本地训练。
            self.aggregate_parameters()
            # 聚合用户上传的参数，更新全局模型。
            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
        #print(loss)
        self.save_results()
        self.save_model()
        # 最后保存训练结果和模型。