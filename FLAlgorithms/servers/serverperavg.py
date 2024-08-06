# 在这段代码中，我们定义了一个名为 PerAvg 的类，它继承自 Server 类，
# 用于实现个性化联邦平均（Per-FedAvg）算法的服务器端功能。下面是对该类及其方法的详细说明：

import torch
import os

from FLAlgorithms.users.userperavg import UserPerAvg
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
# 这些模块和类包括用于读取数据、定义用户和服务器类的基础功能等。
# Implementation for per-FedAvg Server


class PerAvg(Server):
    def __init__(self,device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users,times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)
        # 初始化父类属性: 通过调用 super().__init__ 初始化父类 Server 的属性。
        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        # 读取数据: 使用 read_data 函数读取数据集，并初始化每个用户的数据。
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserPerAvg(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer ,total_users , num_users)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            # 创建用户实例: 为每个用户创建一个 UserPerAvg 实例，并添加到 self.users 列表中，
            # 同时更新 self.total_train_samples。
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating Local Per-Avg.")

    def send_grads(self):
        # 该方法将服务器模型的梯度发送给所有用户。
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        # 该方法用于在多个全局迭代轮次中训练模型：
        # 发送参数: 在每一轮开始时，将服务器的最新参数发送给所有用户。
        # 评估全局模型: 使用 evaluate_one_step 方法评估全局模型。
        # 选择用户并进行训练: 随机选择一些用户并让他们进行本地训练。
        # 聚合参数: 将用户更新后的参数聚合到服务器模型中。
        # 保存结果和模型: 在所有轮次结束后，保存训练结果和最终模型。
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users  发送所有参数给用户
            self.send_parameters()

            # Evaluate gloal model on user for each interation 在每轮中评估全局模型
            print("Evaluate global model with one step update")
            print("")
            self.evaluate_one_step()

            # choose several users to send back upated model to server 选择用户并进行训练
            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
            # 聚合用户更新后的参数
            self.aggregate_parameters()
        # 保存训练结果和模型
        self.save_results()
        self.save_model()
