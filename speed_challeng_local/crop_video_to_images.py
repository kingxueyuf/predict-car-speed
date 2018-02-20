
# coding: utf-8

# In[2]:


import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('data/train.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("img/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1


