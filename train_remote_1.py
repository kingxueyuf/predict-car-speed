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

batch_size = 25
frames_per_forward = 20
frames = 17 * 60 * 20 - 1
train_dataset = os.listdir("img/")
iter_per_epoch = int(frames / (batch_size * frames_per_forward))

def train():
    net = AlexLSTM().cuda()
    util = DatasetUtil()
    criterion = nn.MSELoss()
    lr = 0.0001
    min_loss = 9
    for epoch in range(1000):
        for iteration in range(iter_per_epoch):
            x,y = util.fetch_image_and_label(batch_size, frames_per_forward, frames - frames_per_forward)
            # wrap them in Variable
            x = V(th.from_numpy(x).float()).cuda()
            y = V(th.from_numpy(y).float()).cuda()
            
            optimizer = optim.Adam(net.parameters(), lr=lr)
            optimizer.zero_grad()# zero the parameter gradients
            
            # forward + backward + optimize
            predict = net(x)
            loss = criterion(predict, y)
            loss.backward()
            optimizer.step()
            
            running_loss = 0
            running_loss += loss.data[0]
            
            print('epoch%d_iteration%d_loss%f' % (epoch,iteration,running_loss))
            if running_loss <= min_loss:
                min_loss = running_loss
                th.save(net.state_dict(), 'weight_3/epoch%d_iteration%d_loss%f.p' % (epoch,iteration,min_loss))
        if epoch % 2 == 0:
            th.save(net.state_dict(), 'weight_3/epoch%d.p' % (epoch))
    print('Finished Training')

train()

