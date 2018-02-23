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

batch_size = 25
frames_per_forward = 20
frames = 9 * 60 * 20 - 3
train_dataset = os.listdir("img/")
iter_per_epoch = int(frames / (batch_size * frames_per_forward))

criterion = nn.MSELoss()
model = load_model()
util = DatasetUtil()

def write_to_file(row_to_speed):
    with open('result.txt', 'a') as result:
        for key in sorted(row_to_speed.keys()):
            arr = row_to_speed[key]
            _sum = 0
            for i in range(len(arr)):
                _sum += arr[i]
            _sum = _sum / len(arr)
            line = str(key) + " " + str(_sum)
            result.write(line+"\n")
            
row_to_speed = {}
count = 0
max_row = 10
max_repeat = 5
for i in range(iter_per_epoch * 1000):
    x,dic1 = util.fetch_to_predict_input(batch_size, frames_per_forward, frames - frames_per_forward)
    x = V(th.from_numpy(x).float(), volatile=True).cuda()
    predict = model(x) # batch_size, (20-1=19)
    
    for i in predict.size()[0]:
        # No.i batch
        for j in predict.size()[1]:
            # No.j logit
            index = dic1[i][j+1]
            if index not in row_to_speed.keys():
                row_to_speed[index] = []
            row_to_speed[index].append(predict[i][j])
    
    print('row length', len(row_to_speed))
    if len(row_to_speed) >= max_row:
        count += 1
        print("repeat")
        if count >= max_repeat:
            print("write to file")
            write_to_file(row_to_speed)
            sys.exit()
            