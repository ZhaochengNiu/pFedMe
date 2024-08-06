# pFedMe 类通过继承 Server 类并添加个性化更新和聚合方法，实现了个性化联邦学习算法 pFedMe 的服务器端功能。
# 这些方法包括初始化用户、发送梯度、训练模型、选择用户、评估模型以及保存结果和模型等。
# 该实现确保了每个用户在本地训练过程中能根据自身数据个性化地更新模型，并在服务器端进行有效的参数聚合以提升整体模型性能。

import torch
import os

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
# 这些模块和类包括用于读取数据、定义用户和服务器类的基础功能等。
 
# Implementation for pFedMe Server
# 在这段代码中，我们实现了一个名为 pFedMe 的类，它继承自 Server 类，
# 用于实现个性化联邦学习算法 pFedMe 的服务器端功能。下面是对该类及其方法的详细说明：

class pFedMe(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)
        # 初始化父类属性: 通过调用 super().__init__ 初始化父类 Server 的属性。
        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        # 读取数据: 使用 read_data 函数读取数据集，并初始化每个用户的数据。
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserpFedMe(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            # 创建用户实例: 为每个用户创建一个 UserpFedMe 实例，并添加到 self.users 列表中，同时更新 self.total_train_samples。
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedMe server.")

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
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()
            # 发送参数: 在每一轮开始时，将服务器的最新参数发送给所有用户。
            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()
            # 评估全局模型: 使用 evaluate 方法评估全局模型。
            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            # 训练用户: 对所有用户进行本地训练。
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)
            # 选择用户并进行个性化评估和聚合: 随机选择一些用户，评估个性化模型，并聚合个性化参数。
            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters()
            # 保存结果和模型: 在所有轮次结束后，保存训练结果和最终模型。

        #print(loss)
        self.save_results()
        self.save_model()
    
  
