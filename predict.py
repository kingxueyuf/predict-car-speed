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

def write_to_file(row_to_speed):
    with open('result.txt', 'a') as result:
        for key in sorted(row_to_speed.keys()):
            line = str(key) + " " + str(row_to_speed[key].data[0])
            result.write(line+"\n")
    
    
    
row_to_speed = {}
count = 0
max_row = 10500
max_repeat = 1000
for i in range(iter_per_epoch * 1000):
    x,offset_index_map = util.fetch_to_predict_input(batch_size, time_stamp, image_num_per_time_stamp, video_length_in_seconds - time_stamp - 1)
    x = V(th.from_numpy(x).float(), volatile=True).cuda()
    predict = model(x)
#     print("---Predict---")
#     print(predict)
    predict = predict[0]
    for key, value in offset_index_map.items():
        speed = predict[key]
        if value in row_to_speed:
            row_to_speed[value] = (row_to_speed[value] + speed) / 2
        else:
            row_to_speed[value] = speed
    print('row length', len(row_to_speed))
    if len(row_to_speed) > max_row:
        count += 1
        print("repeat")
        if count > max_repeat:
            print("write to file")
            write_to_file(row_to_speed)
            sys.exit()
            
    
    