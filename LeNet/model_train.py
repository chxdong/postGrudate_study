import copy
import time

import pandas as pd
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn

def  train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True
                              )
    train_data, val_data = Data.random_split(train_data,[round(0.8*len(train_data)), round(0.2*len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)

    return train_dataloader, val_dataloader



def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所用得设备
    device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()

    # 将模型放到训练设备当中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []

    # 训练时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)


        # 初始化参数
        # 训练集loss值和准确度
        train_loss = 0.0
        train_corrects = 0

        # 验证集loss值和准确度
        val_loss = 0.0
        val_corrects = 0

        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征
            b_x = b_x.to(device)

            b_y = b_y.to(device)

            model.train()

            output = model(b_x)

            pre_lab = torch.argmax(output, dim =1)

            loss = criterion(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # 验证
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 开启评估模式
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim =1)
            loss = criterion(output, b_y)
            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print("{} 训练集loss: {:.4f}, 训练集准确度: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} 验证集loss: {:.4f}, 验证集准确度: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print("训练耗时: {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    model.load_state_dict(best_model_wts)
    torch.save(model.load_state_dict(best_model_wts), "../LeNet/best_model.pth")

    train_process = pd.DataFrame({
        "epoch": range(num_epochs),
        "train_loss": train_loss_all,
        "train_acc": train_acc_all,
        "val_loss": val_loss_all,
        "val_acc": val_acc_all
    })
    return train_process

def matplotlib_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all,'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all,'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")


    plt.plot(train_process["epoch"], train_process.train_acc_all,'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process.val_acc_all,'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    LeNet = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, num_epochs=20)
    matplotlib_acc_loss(train_process)

