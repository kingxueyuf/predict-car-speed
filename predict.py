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

dict = {}
for i in range(iter_per_epoch):
    # Per 30s
    batch_offset = i * time_stamp * 20
    x = np.zeros((batch_size, time_stamp * image_num_per_time_stamp, 160, 420, 3)) # (160, 420) (480, 640) 
    for j in range(10):
        for k in range(time_stamp):
            for l in range(image_num_per_time_stamp):
                index = batch_offset + j*2 + l + k * 20
                bgr_img = cv2.imread("img/frame" + str(index) + ".jpg")
                
                b,g,r = cv2.split(bgr_img) 
                rgb_img = cv2.merge([r,g,b])
                x[i,index] = rgb_img[190:350, 100:520, :]
            
    x = V(th.from_numpy(x).float()).cuda()
    predict = model(x)
    # 30 * 2 * 1
    
    print("---Predict---")
    print(predict)
        