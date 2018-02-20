
# coding: utf-8

# In[ ]:


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
    m.load_state_dict(model_path)
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

