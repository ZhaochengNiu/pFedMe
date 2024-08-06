# 以下是 UserPerAvg 类的实现，用于 Per-FedAvg 联邦学习算法。此实现包括初始化类、设置梯度、训练模型等方法。

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import MySGD, FEDLOptimizer
from FLAlgorithms.users.userbase import User

# Implementation for Per-FedAvg clients


class UserPerAvg(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, total_users , num_users):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs)
        self.total_users = total_users
        self.num_users = num_users
        
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
        # self.loss 根据模型类型选择不同的损失函数。
        self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer 使用 MySGD 优化器进行模型参数更新。

    def set_grads(self, new_grads):
        # 该方法用于设置模型的梯度，可以接受 nn.Parameter 或列表形式的梯度。
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()

    def train(self, epochs):
        # 在每个本地训练周期中，执行两步更新，并在第一步后恢复模型参数。
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):  # local update 
            self.model.train()

            temp_model = copy.deepcopy(list(self.model.parameters()))

            #step 1
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

            #step 2
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()

            # restore the model parameters to the one before first update
            for old_p, new_p in zip(self.model.parameters(), temp_model):
                old_p.data = new_p.data.clone()
                
            self.optimizer.step(beta = self.beta)

            # clone model to user model 
            self.clone_model_paramenter(self.model.parameters(), self.local_model)

        return LOSS    

    def train_one_step(self):
        # 执行一步更新，用于在一个测试批次上进行参数更新。
        self.model.train()
        # step 1
        X, y = self.get_next_test_batch()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
        # step 2
        X, y = self.get_next_test_batch()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step(beta=self.beta)