
# coding: utf-8

# In[2]:



# coding: utf-8

# In[2]:


import cv2
import matplotlib.pyplot as plt


print(cv2.__version__)
def write_video_to_images():
    vidcap = cv2.VideoCapture('data/test.mp4')
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      cv2.imwrite("img_test/frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1

def crop_image():
    bgr_img = cv2.imread('img/frame0.jpg')
    
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb

    plt.imshow(rgb_img)
    plt.show()
    

    rgb_img = rgb_img[190:350, 100:520] # crop
    plt.imshow(rgb_img)
    plt.show()
    
    
write_video_to_images()

    

