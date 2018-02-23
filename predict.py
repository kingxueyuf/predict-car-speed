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
video_length_in_seconds = 9 * 60 
train_dataset = os.listdir("img/")
iter_per_epoch = int(video_length_in_seconds / (batch_size * time_stamp))

criterion = nn.MSELoss()
model = load_model()
util = DatasetUtil()

row_to_speed = {}
for i in range(iter_per_epoch * 1000):
    x,offset_index_map = util.fetch_to_predict_input(batch_size, time_stamp, image_num_per_time_stamp, video_length_in_seconds - time_stamp)
    x = V(th.from_numpy(x).float()).cuda()
    predict = model(x)
    for key, value in predict.items():
        speed = predict[key]
        if value in row_to_speed:
            row_to_speed[value] = (row_to_speed[value] + speed) / 2
        else:
            row_to_speed[value] = speed
   
    print("---Predict---")
    print(predict)
        