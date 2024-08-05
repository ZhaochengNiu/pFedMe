#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)
# 这段代码首先导入了所需的库和模块：
# h5py：用于处理HDF5文件格式的数据。
# matplotlib.pyplot：用于绘制图表。
# numpy：用于数值计算。
# argparse：用于解析命令行参数。
# importlib：用于动态导入模块。
# random：用于生成随机数。
# os：用于与操作系统交互。
# FedAvg、pFedMe、PerAvg：分别从FLAlgorithms.servers模块中导入的三种联邦学习算法。
# 从FLAlgorithms.trainmodel.models模块导入的各种模型类。
# 从utils.plot_utils模块导入的绘图工具函数。
# torch：用于深度学习（PyTorch）。
# 设置了随机种子，确保结果可复现。


def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    # 设备选择：根据是否有GPU以及gpu参数选择使用GPU或CPU。
    for i in range(times):
        # 循环执行多次训练和测试：根据times参数确定训练和测试的次数。
        print("---------------Running time:------------",i)
        # 模型生成 Generate model
        # 根据输入的model和dataset参数生成不同的模型实例，并移动到指定设备（CPU或GPU）。
        # mclr（多类逻辑回归），cnn（卷积神经网络），dnn（深度神经网络）。
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model
            else:
                model = Mclr_Logistic(60,10).to(device), model
                
        if(model == "cnn"):
            if(dataset == "Mnist"):
                model = Net().to(device), model
            elif(dataset == "Cifar10"):
                model = CNNCifar(10).to(device), model
            
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN().to(device), model
            else: 
                model = DNN(60,20,10).to(device), model
        # select algorithm
        # 选择算法：根据输入的algorithm参数选择不同的联邦学习算法，并实例化相应的服务器对象（FedAvg、pFedMe、PerAvg）。
        if(algorithm == "FedAvg"):
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i)
        
        if(algorithm == "pFedMe"):
            server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i)

        if(algorithm == "PerAvg"):
            server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i)
        server.train()
        server.test()
        # 训练和测试：调用服务器对象的train和test方法进行训练和测试。
    # Average data 
    if(algorithm == "PerAvg"):
        algorithm == "PerAvg_p"
    if(algorithm == "pFedMe"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pFedMe_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)
        # 数据平均：在算法为PerAvg或pFedMe时，调用average_data函数进行数据平均处理。
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)


if __name__ == "__main__":
    # 命令行参数解析：使用argparse库解析命令行参数，并将这些参数传递给main函数。
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)
    # 打印训练过程总结：输出一些关于训练过程的关键信息，如算法、批大小、学习率等。

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta = args.beta, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        gpu=args.gpu
        )
    # 调用主函数：根据解析的参数调用main函数，执行联邦学习实验。
