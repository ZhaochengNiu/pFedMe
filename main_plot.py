#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from utils.plot_utils import *
import torch
torch.manual_seed(0)
# 导入必要的库和模块，包括用于处理HDF5文件的h5py，
# 绘图的matplotlib.pyplot，
# 数值计算的numpy，
# 命令行参数解析的argparse，
# 动态导入模块的importlib，
# 生成随机数的random，
# 操作系统接口os，
# 以及绘图工具函数plot_utils中的所有函数。
# 导入torch并设置随机种子，以确保结果可复现。


# 这段代码的功能是通过两个条件语句选择不同的数据集（MNIST和Synthetic），
# 并分别使用不同的参数配置，调用绘图函数生成联邦学习算法比较的图表。以下是逐行解释：

# if(0)语句中的代码块不会执行，因为条件为0。
# 代码块包含了针对MNIST数据集的参数配置和调用plot_summary_one_figure_mnist_Compare函数生成图表的逻辑。
if(0): # plot for MNIST convex 
    numusers = 5
    num_glob_iters = 800
    dataset = "Mnist"
    local_ep = [20,20,20,20]
    lamda = [15,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005]
    beta =  [1.0, 1.0, 0.001, 1.0]
    batch_size = [20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.1,0.1,0.1,0.1]
    algorithms = [ "pFedMe_p","pFedMe","PerAvg_p","FedAvg"]
    plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)


# if(1)语句中的代码块会执行，因为条件为1。
# 这个代码块包含了针对Synthetic数据集的参数配置，并调用plot_summary_one_figure_synthetic_Compare函数生成图表的逻辑。
if(1): # plot for Synthetic covex
    numusers = 10
    num_glob_iters = 600
    dataset = "Synthetic"
    local_ep = [20,20,20,20]
    lamda = [20,20,20,20]
    learning_rate = [0.005, 0.005, 0.005, 0.005]
    beta =  [1.0, 1.0, 0.001, 1.0]
    batch_size = [20,20,20,20]
    K = [5,5,5,5]
    personal_learning_rate = [0.01,0.01,0.01,0.01] 
    algorithms = [ "pFedMe_p","pFedMe","PerAvg_p","FedAvg"]
    # numusers：用户数量。
    # num_glob_iters：全局迭代次数。
    # dataset：数据集名称。
    # local_ep：本地训练轮数。
    # lamda：正则化参数。
    # learning_rate：学习率。
    # beta：算法参数，pFedMe的平均移动参数或PerAvg的第二学习率。
    # batch_size：批量大小。
    # K：计算步骤。
    # personal_learning_rate：个性化学习率，用于pFedMe计算近似θ。
    # algorithms：所使用的联邦学习算法列表。
    plot_summary_one_figure_synthetic_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

