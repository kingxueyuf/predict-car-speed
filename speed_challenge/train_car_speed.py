
# coding: utf-8

# In[ ]:



# coding: utf-8

# In[42]:


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
        
        logit = logit.transpose(1,0).squeeze(2)
        return logit

    def _init_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            V(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01)),
            V(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01))
        )
    
net = AlexLSTM()
# print(net)

batch_size = 5
time_stamp = 20
train_dataset = os.listdir("img/")
total_img_num = len(train_dataset)
iteration_per_epoch = int(total_img_num / (batch_size*time_stamp))
lr = 0.0001
criterion = nn.MSELoss(False)

def fetch_image_and_label(batch_size, time_stamp):
    numbers = []
    while(len(numbers) != batch_size):
        a = random.randint(0,total_img_num-time_stamp)
        if a not in numbers:
            numbers.append(a)
    label = []
    file_in = open('data/train.txt', 'r')
    for line in file_in.readlines():
        label.append(float(line))
    
    x = np.zeros((batch_size, time_stamp, 480, 640, 3))
    y = np.zeros((batch_size, time_stamp))
    for i in range(batch_size):
        for j in range(time_stamp):
            img_name = numbers[i] + j
            image_path = 'img/frame' + str(img_name) + ".jpg"
            img = cv2.imread(image_path)
            x[i,j] = img
            y[i,j] = label[numbers[i] + j]
    
    x = x.transpose(0, 4, 1, 2, 3) # (batch_size, 3, time_stamp, 480, 640)
    return x, y
    
min_loss = 100
for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i in range(iteration_per_epoch):
        x,y = fetch_image_and_label(batch_size, time_stamp)

        # wrap them in Variable
        x = V(th.from_numpy(x).float())
        y = V(th.from_numpy(y).float())

        optimizer = optim.Adam(net.parameters(), lr=lr)
        optimizer.zero_grad()# zero the parameter gradients
        # forward + backward + optimize
        predict = net(x)
        
        print("predict shape : ", predict.size())
        print("label shape : ", y.size())
        print("------ PREDICT start------")
        print(predict)
        print("------ PREDICT   end------")
        print("------ LABEL start------")
        print(y)
        print("------ LABEL   end------")
        loss = criterion(predict, y)
        loss.backward()
        optimizer.step()

        # print statistics
        print("loss shape : ",loss.data.size())
        running_loss += loss.data[0]
        if running_loss <= min_loss :
            min_loss = running_loss
            print("Saving model ...")
            th.save(net.state_dict(), 'save/%d_%s.p' % (i, epoch))
        print('[epoch : %d, iteration : %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0
    print("Saving model ...")
    th.save(net.state_dict(), 'save/epoch_%s.p' % (epoch))

print('Finished Training')



