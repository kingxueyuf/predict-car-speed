import torch.nn as nn
from torch.autograd import Variable as V
import torch as th
from torchvision import models
import os
import torch.optim as optim
import random
import numpy as np
import cv2 as cv2

class AlexLSTM(nn.Module):
    def __init__(self, n_layers=2, h_size=420):
        super(AlexLSTM, self).__init__()
        self.h_size = h_size
        self.n_layers = n_layers

        alexnet = models.alexnet(pretrained=True)
        self.conv = nn.Sequential(*list(alexnet.children())[:-1])

        self.lstm = nn.LSTM(68096, h_size, dropout=0.2, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(h_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # (batch_size, 3, time_stamp, 480, 640)
        batch_size, timesteps = x.size()[0], x.size()[2]
        state = self._init_state(b_size=batch_size)

        convs = []
        for t in range(timesteps):
            conv = self.conv(x[:, :, t, :, :])
#             print("conv shape : ", conv.size())
            conv = conv.view(batch_size, -1)
#             print("conv reshape :", conv.size())
            convs.append(conv)
        convs = th.stack(convs, 0)
        print("alex output shape : ",convs.size()) # ([20, 5, 68096]) (seq_len, batch, input_size)
        print("lstm input shape : ",convs.size())
        lstm, _ = self.lstm(convs, state) # lstm input (seq_len, batch, input_size)
        print("lstm output shape : ",lstm.size()) # torch.Size([20, 5, 420]) (seq_len, batch, hidden_size * num_directions)
        print("fc input shape : ",lstm.size())
        logit = self.fc(lstm) # seq_len, batch, input_size ([20, 5, 1])
        print("fc output shape : ",logit.size())
        
        logit = logit.transpose(1,0).squeeze(2) # [batch, seq_len]
        return logit

    def _init_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            V(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01)),
            V(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01))
        )

def fetch_image_and_label(batch_size, time_stamp):
    numbers = []
    while(len(numbers) != batch_size):
        a = random.randint(0,total_img_num-time_stamp)
        if a not in numbers:
            numbers.append(a)
    label = []
    file_in = open('../data/train.txt', 'r')
    for line in file_in.readlines():
        label.append(float(line))
    
    x = np.zeros((batch_size, time_stamp, 480, 640, 3))
    y = np.zeros((batch_size, time_stamp))
    for i in range(batch_size):
        for j in range(time_stamp):
            img_name = numbers[i] + j
            image_path = '../img/frame' + str(img_name) + ".jpg"
            img = cv2.imread(image_path)
            x[i,j] = img
            y[i,j] = label[numbers[i] + j]
    
    x = x.transpose(0, 4, 1, 2, 3) # (batch_size, 3, time_stamp, 480, 640)
    return x, y

def load_model(): 
    model_path = "../weight/epoch_19.p"
    m = AlexLSTM()
    m.load_state_dict(th.load(model_path))
    m = m.cuda()
    return m

batch_size = 1 #5
time_stamp = 40  #20
criterion = nn.MSELoss()
model = load_model()
x, label = fetch_image_and_label(batch_size, time_stamp)
x = V(th.from_numpy(x).float()).cuda()
predict = model(x)
loss = criterion(predict, y)
print("loss : ", loss)
print("---Predict---")
print(predict)
print("---Label---")
print(y)