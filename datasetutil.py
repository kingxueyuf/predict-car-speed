import random
import numpy as np
import cv2 as cv2
import os

class DatasetUtil():
    
    def fetch_image_and_label(self, batch_size, time_stamp, frame_range):
        label = []
        file_in = open('data/train.txt', 'r')
        for line in file_in.readlines():
            label.append(float(line))

        x = np.zeros((batch_size, time_stamp, 160, 420, 3)) # (160, 420) (480, 640) 
        y = np.zeros((batch_size, time_stamp - 1))
        for i in range(batch_size):
            # For each batch
            start_frame = random.randint(0,frame_range)
            for j in range(time_stamp):
                index = start_frame + j
                bgr_img = cv2.imread("img/frame" + str(index) + ".jpg")
                
                b,g,r = cv2.split(bgr_img)
                rgb_img = cv2.merge([r,g,b])
                
                x[i,j] = rgb_img[190:350, 100:520, :]
                if j != 0:
                    y[i,j - 1] = label[index]

        x = x.transpose(0, 4, 1, 2, 3) # (batch_size, 3, time_stamp, 480, 640)
        return x, y
    
    def fetch_to_predict_input(self, batch_size, time_stamp, frame_range):
        x = np.zeros((batch_size, time_stamp, 160, 420, 3)) # (160, 420) (480, 640) 
        dic1 = np.zeros(batch_size, time_stamp)
        for i in range(batch_size):
            # For each batch
            start_frame = random.randint(0,frame_range)
            for j in range(time_stamp):
                index = start_frame + j
                bgr_img = cv2.imread("img_test/frame" + str(index) + ".jpg")
                
                b,g,r = cv2.split(bgr_img)
                rgb_img = cv2.merge([r,g,b])
                
                x[i,j] = rgb_img[190:350, 100:520, :]
                dic1[i,j] = index

        x = x.transpose(0, 4, 1, 2, 3) # (batch_size, 3, time_stamp, 480, 640)
        return x, dic1
