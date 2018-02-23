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


def load_model(): 
    model_path = "weight/epoch440.p"
    m = AlexLSTM()
    m.load_state_dict(th.load(model_path))
    m = m.cuda()
    return m

batch_size = 1
time_stamp = 30
image_num_per_time_stamp = 2
video_length_in_seconds = 17 * 60
train_dataset = os.listdir("img/")
iter_per_epoch = int(video_length_in_seconds / (batch_size * time_stamp))

criterion = nn.MSELoss()
model = load_model()
util = DatasetUtil()

for i in range(1000):
    x,y = util.fetch_image_and_label(batch_size, time_stamp, image_num_per_time_stamp, video_length_in_seconds - time_stamp)
    x = V(th.from_numpy(x).float()).cuda()
    y = V(th.from_numpy(y).float()).cuda()
    predict = model(x)
    loss = criterion(predict, y)
    print("------------------------------------------")
    print("---loss---")
    print(loss)
    print("---Predict---")
    print(predict)
    print("---Label---")
    print(y)
        