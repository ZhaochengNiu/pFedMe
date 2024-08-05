# 这段代码实现了 FedAvg 客户端类 UserAVG，用于在联邦学习环境中处理每个用户的本地训练过程。以下是逐行详细解释：
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
# torch 是 PyTorch 的核心库。
# torch.nn 包含了神经网络模块。
# torch.nn.functional 提供了常用的神经网络函数。
# os 和 json 提供了操作系统相关功能和 JSON 文件处理。
# torch.utils.data.DataLoader 是 PyTorch 数据加载器模块。
# 从自定义模块 FLAlgorithms.users.userbase 导入 User 基类。


# Implementation for FedAvg clients
class UserAVG(User):
    # UserAVG 类继承自 User 基类。
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs)
        # __init__ 方法初始化 UserAVG 的各项参数，并调用 User 基类的初始化方法。
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # 根据模型类型选择损失函数，如果模型是 Mclr_CrossEntropy，使用交叉熵损失，否则使用负对数似然损失。
        # 使用随机梯度下降（SGD）优化器。

    def set_grads(self, new_grads):
        # set_grads 方法设置模型的梯度，可以接受单个参数或参数列表。
        if isinstance(new_grads, nn.Parameter):
            # 如果 new_grads 是 nn.Parameter 实例，将其数据复制到模型的参数中。
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            # 如果 new_grads 是列表，逐个复制每个参数的数据。
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        # train 方法执行本地训练过程。
        LOSS = 0
        self.model.train()
        # 初始化 LOSS 变量并将模型设为训练模式。
        for epoch in range(1, self.local_epochs + 1):
            # 循环进行本地训练的每个 epoch：
            self.model.train()
            X, y = self.get_next_train_batch()
            # 获取下一批训练数据 X 和标签 y。
            self.optimizer.zero_grad()
            # 清零梯度。
            output = self.model(X)
            # 将数据传入模型，计算输出。
            loss = self.loss(output, y)
            loss.backward()
            # 计算损失，并进行反向传播。
            self.optimizer.step()
            # 更新模型参数。
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            # 将模型参数克隆到本地模型参数中。
            # 返回 LOSS。
        return LOSS

