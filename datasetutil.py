import random
import numpy as np
import cv2 as cv2
import os

class DatasetUtil():
    
    def fetch_image_and_label(self, batch_size, time_stamp, frame_offset_per_time_stamp, total_img_num):
        numbers = []
        while(len(numbers) != batch_size):
            a = random.randint(0, total_img_num-time_stamp)
            if a not in numbers:
                numbers.append(a)
        label = []
        file_in = open('data/train.txt', 'r')
        for line in file_in.readlines():
            label.append(float(line))

        x = np.zeros((batch_size, time_stamp, 160, 420, 3)) # (160, 420) (480, 640) 
        y = np.zeros((batch_size, time_stamp))
        for i in range(batch_size):
            for j in range(time_stamp):
                offset = j * frame_offset_per_time_stamp
                index = numbers[i] + offset

                bgr_img = cv2.imread("img/frame" + str(index) + ".jpg")

                b,g,r = cv2.split(bgr_img)       # get b,g,r
                rgb_img = cv2.merge([r,g,b])     # switch it to rgb

                x[i,j] = rgb_img[190:350, 100:520, :] # crop
                y[i,j] = label[index]

        x = x.transpose(0, 4, 1, 2, 3) # (batch_size, 3, time_stamp, 480, 640)
        return x, y
