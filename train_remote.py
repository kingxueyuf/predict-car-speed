import torch.nn as nn
from torch.autograd import Variable as V
import torch as th
from torchvision import models
import os
import torch.optim as optim
import random
import numpy as np
import cv2 as cv2
from alexlstm import AlexLSTM
from datasetutil import DatasetUtil
from importlib import reload

batch_size = 5
time_stamp = 20
frame_offset_per_time_stamp = 10
train_dataset = os.listdir("img/")
total_img_num = len(train_dataset)
iteration_per_epoch = int(total_img_num / (batch_size * time_stamp * frame_offset_per_time_stamp))

def train():
    net = AlexLSTM().cuda()
    util = DatasetUtil()
    criterion = nn.MSELoss(False)
    lr = 0.0001
    min_loss = 100
    for epoch in range(100):
        for offset in range(frame_offset_per_time_stamp):  # offset should smaller than frame_offset_per_time_stamp
            running_loss = 0.0
            for i in range(iteration_per_epoch):
                x,y = util.fetch_image_and_label(batch_size, time_stamp, frame_offset_per_time_stamp, total_img_num, offset +i*batch_size*time_stamp * frame_offset_per_time_stamp)

                # wrap them in Variable
                x = V(th.from_numpy(x).float()).cuda()
                y = V(th.from_numpy(y).float()).cuda()

                optimizer = optim.Adam(net.parameters(), lr=lr)
                optimizer.zero_grad()# zero the parameter gradients
                # forward + backward + optimize
                predict = net(x)

                print("------------ PREDICT ------------")
                print(predict)
                print("------------ LABEL --------------")
                print(y)
                loss = criterion(predict, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                if running_loss <= min_loss :
                    min_loss = running_loss
                    print("--- Found smaller loss ---")
                    th.save(net.state_dict(), 'weight/epoch%d_offset%d_iter%d.p' % (epoch,offset,i))
                print('[epoch : %d, offset : %d, i : %d] loss: %.3f' % (epoch, offset, i, running_loss))
                running_loss = 0.0
            print("Saving model per offset...")
            th.save(net.state_dict(), 'weight/epoch%d_offset%d.p' % (epoch,offset))
    print('Finished Training')

train()

