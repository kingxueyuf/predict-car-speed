
# coding: utf-8

# In[2]:


import cv2
import matplotlib.pyplot as plt


print(cv2.__version__)
def write_video_to_images():
    vidcap = cv2.VideoCapture('../data/train.mp4')
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      cv2.imwrite("../img/frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1

def crop_image():
    img = cv2.imread('../img/frame0.jpg')
    plt.imshow(img)
    
crop_image()

    