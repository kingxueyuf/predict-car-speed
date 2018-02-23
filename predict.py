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
import sys 

def load_model(): 
    model_path = "weight_2/epoch7_iteration24_loss  1.p"
    m = AlexLSTM()
    m.load_state_dict(th.load(model_path))
    m = m.cuda()
    return m

def write_to_file(row_to_speed):
    with open('result.txt', 'a') as result:
        for key in sorted(row_to_speed.keys()):
            arr = row_to_speed[key]
            _sum = 0
            for i in range(len(arr)):
                _sum += arr[i].data[0]
            _sum = _sum / len(arr)
            line = str(key) + " " + str(_sum)
            result.write(line+"\n")

batch_size = 1
frames_per_forward = 20
frames = 9 * 60 * 20 - 3

criterion = nn.MSELoss()
model = load_model()
util = DatasetUtil()

row_to_speed = {}

while start <= (frames - frames_per_forward):
    x = util.fetch_to_predict_input(frames_per_forward, start)
    x = V(th.from_numpy(x).float(), volatile=True).cuda()
    predict = model(x) # batch_size, (20-1=19)
    for j in range(predict.size()[1]):
        # No.j logit
        index = start + j + 1
        if index not in row_to_speed.keys():
            row_to_speed[index] = []
        row_to_speed[index].append(predict[0][j])
    start += 1
write_to_file(row_to_speed)
            