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
    def __init__(self, n_layers=3, h_size=420):
        super(AlexLSTM, self).__init__()
        self.h_size = h_size
        self.n_layers = n_layers

        alexnet = models.alexnet(pretrained=True)
        self.conv = nn.Sequential(*list(alexnet.children())[:-1])

        self.lstm = nn.LSTM(12288, h_size, dropout=0.15, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(h_size, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
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
#         print("alex output shape : ",convs.size()) # ([20, 5, 12288]) (seq_len, batch, input_size)
#         print("lstm input shape : ",convs.size())
        lstm, _ = self.lstm(convs, state) # lstm input (seq_len, batch, input_size)
#         print("lstm output shape : ",lstm.size()) # torch.Size([20, 5, 420]) (seq_len, batch, hidden_size * num_directions)
#         print("fc input shape : ",lstm.size())
        logit = self.fc(lstm) # seq_len, batch, input_size ([20, 5, 1])
#         print("fc output shape : ",logit.size())
        
        logit = logit.transpose(1,0).squeeze(2)
        return logit

    def _init_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            V(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01)),
            V(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01))
        )