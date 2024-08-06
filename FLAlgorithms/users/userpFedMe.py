# 在 UserpFedMe 类的实现中，我们主要是将个性化联邦学习算法 pFedMe 的训练逻辑实现。
# 这些步骤包括初始化类、设置梯度、训练模型等。

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.users.userbase import User
import copy

# Implementation for pFeMe clients


class UserpFedMe(User):
    # 这样实现的 UserpFedMe 类能够在个性化联邦学习框架下进行有效的模型训练和更新。
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, K, personal_learning_rate):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
        # self.loss 根据模型类型选择不同的损失函数。
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        # self.K 和 self.personal_learning_rate 是 pFedMe 特有的参数，用于个性化训练步骤和学习率。
        self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)
        # self.optimizer 使用 pFedMeOptimizer 进行个性化优化。

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
        # self.model.train() 设置模型为训练模式。
        for epoch in range(1, self.local_epochs + 1):  # local update
            self.model.train()
            X, y = self.get_next_train_batch()

            # K = 30 # K is number of personalized steps
            for i in range(self.K):
                # for i in range(self.K): 循环进行个性化训练步骤。
                self.optimizer.zero_grad()
                # self.optimizer.zero_grad() 清除梯度。
                output = self.model(X)
                # output = self.model(X) 前向传播。
                loss = self.loss(output, y)
                # loss = self.loss(output, y) 计算损失。
                loss.backward()
                # loss.backward() 反向传播。
                self.persionalized_model_bar, _ = self.optimizer.step(self.local_model)
                # self.persionalized_model_bar, _ = self.optimizer.step(self.local_model) 更新个性化模型参数。
            # update local weight after finding aproximate theta
            for new_param, localweight in zip(self.persionalized_model_bar, self.local_model):
                localweight.data = localweight.data - self.lamda * self.learning_rate * (localweight.data - new_param.data)
                # 更新本地权重。
        #update local model as local_weight_upated
        #self.clone_model_paramenter(self.local_weight_updated, self.local_model)
        self.update_parameters(self.local_model)
        # self.update_parameters(self.local_model) 更新本地模型参数。

        return LOSS