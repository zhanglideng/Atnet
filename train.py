# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.loss import *
from utils.print_time import *
from utils.save_log_to_excel import *
from dataloader import AtDataSet
from At_model import *
import time
import xlwt
from utils.ms_ssim import *
import os

LR = 0.004  # 学习率
EPOCH = 100  # 轮次
BATCH_SIZE = 4  # 批大小
excel_train_line = 1  # train_excel写入的行的下标
excel_val_line = 1  # val_excel写入的行的下标
alpha = 1  # 损失函数的权重
accumulation_steps = 1  # 梯度积累的次数，类似于batch-size=64
itr_to_lr = 10000 // BATCH_SIZE  # 训练10000次后损失下降50%
itr_to_excel = 64 // BATCH_SIZE  # 训练64次后保存相关数据到excel
loss_num = 5  # 包括参加训练和不参加训练的loss
weight = [1, 1, 1, 1, 1]
train_haze_path = '/home/aistudio/data/data20070/nyu/train/'  # 去雾训练集的路径
val_haze_path = '/home/aistudio/data/data20070/nyu/val/'  # 去雾验证集的路径
gt_path = '/home/aistudio/data/data20070/nyu/gth/'
d_path = '/home/aistudio/data/data20070/nyu/depth/'
save_path = './checkpoints/best_cnn_model.pt'  # 保存模型的路径
excel_save = './result.xls'  # 保存excel的路径

'''
def adjust_learning_rate(op, i):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.90 ** (i // itr_to_lr))
    for param_group in op.param_groups:
        param_group['lr'] = lr
'''

# 初始化excel
f, sheet_train, sheet_val = init_excel()
# 加载模型
net = At()
net = net.cuda()
print(net)

# 数据转换模式
transform = transforms.Compose([transforms.ToTensor()])
# 读取训练集数据
train_path_list = [train_haze_path, gt_path, d_path]
train_data = EdDataSet(transform, train_path_list)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 读取验证集数据
val_path_list = [val_haze_path, gt_path, d_path]
val_data = EdDataSet(transform, val_path_list)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)

min_loss = 999999999
min_epoch = 0
itr = 0
start_time = time.time()

# 开始训练
print("\nstart to train!")
for epoch in range(EPOCH):
    index = 0
    train_epo_loss = 0
    loss = 0
    loss_excel = [0] * loss_num
    for haze_image, gt_image, A_image, t_image in train_data_loader:
        index += 1
        itr += 1
        J, A, t = net(haze_image)
        loss_image = [gt_image, A_image, t_image, J, A, t]
        loss, temp_loss = loss_function(loss_image, weight)
        loss_excel = [loss_excel[i] + temp_loss[i].item() for i in range(len(loss_excel))]
        loss = loss / accumulation_steps
        loss.backward()
        iter_loss = loss.item()
        # 3. update parameters of net
        if ((index + 1) % accumulation_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient
        if np.mod(index, itr_to_excel) == 0:
            loss_excel = [loss_excel[i] / itr_to_excel for i in range(len(loss_excel))]
            print('epoch %d, %03d/%d' % (epoch + 1, index, len(train_data_loader)))
            print('J_L2=%.5f\n' 'J_SSIM=%.5f\n' 'A_L2=%.5f\n' 't_L2=%.5f\n' 't_SSIM=%.5f' %
                  (loss_excel[0], loss_excel[1], loss_excel[2], loss_excel[3], loss_excel[4]))
            print_time(start_time, index, EPOCH, len(train_data_loader), epoch)
            excel_train_line = write_excel(sheet=sheet_train,
                                           data_type='train',
                                           line=excel_train_line,
                                           epoch=epoch,
                                           itr=itr,
                                           loss=loss_excel)
            f.save(excel_save)
            loss_excel = [0] * loss_num
    optimizer.step()
    optimizer.zero_grad()
    loss_excel = [0] * loss_num
    with torch.no_grad():
        for haze_image, gt_image, A_image, t_image in val_data_loader:
            J, A, t = net(haze_image)
            loss, temp_loss = loss_function(
                [gt_image, output_image, gt_scene_feature, dehaze_image, hazy_scene_feature], weight)
            loss_excel = [loss_excel[i] + temp_loss[i].item() for i in range(len(loss_excel))]
    train_epo_loss = train_epo_loss / len(train_data_loader)
    val_epoch_loss = sum(loss_excel)
    loss_excel = [loss_excel[i] / len(val_data_loader) for i in range(len(loss_excel))]
    print('\nepoch %d train loss = %.5f' % (epoch + 1, train_epo_loss))
    print('J_L2=%.5f\n' 'J_SSIM=%.5f\n' 'A_L2=%.5f\n' 't_L2=%.5f\n' 't_SSIM=%.5f'
          % (loss_excel[0], loss_excel[1], loss_excel[2], loss_excel[3], loss_excel[4]))
    excel_val_line = write_excel(sheet=sheet_val,
                                 data_type='val',
                                 line=excel_val_line,
                                 epoch=epoch,
                                 itr=False,
                                 loss=loss_excel)
    f.save(excel_save)
    if val_epoch_loss < min_loss:
        min_loss = val_epoch_loss
        min_epoch = epoch
        torch.save(net, save_path)
        print('saving the epoch %d model with %.5f' % (epoch + 1, min_loss))
print('Train is Done!')
