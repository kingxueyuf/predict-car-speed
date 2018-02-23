# predict-car-speed

# Learn and Predict Vehicle Speed From A Video

![highway](images/highway.jpg)

**This is a project report for the comma.ai programming challenge.**

Basically, the goal is to predict the speed of a car from a video.

## Dataset Exploration

train.mp4 17:00 640*480
train.txt 20400 lines label!

### Train Dataset

The training dataset is about 17 minutes in total, which contains 20400 frames.
Two types of scenes seem to appear:

* **0:00 - 12:30**: highway (12 min 30 sec)
* **12:31 - 17:00**: street (4 min 30 sec)

![plot_train_speed](images/plot_train_speed.jpg)

## Models

### AlexLSTM (2D CNN + LSTM)

![clstm](images/clstm.png)


```
[!] Model Summary:
AlexLSTM (
  (conv): Sequential (
    (0): Sequential (
      (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
      (1): ReLU (inplace)
      (2): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1))
      (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (4): ReLU (inplace)
      (5): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1))
      (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU (inplace)
      (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): ReLU (inplace)
      (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU (inplace)
      (12): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1))
    )
  )
  (lstm): LSTM(1280, 256, num_layers=2, dropout=0.2)
  (fc): Sequential (
    (0): Linear (256 -> 64)
    (1): ReLU ()
    (2): Dropout (p = 0.2)
    (3): Linear (64 -> 1)
  )
)
```
## Result

https://github.com/keon/speed/blob/master/predictions.txt

## Future Work

I believe there is a lot of room for improvements. I wish I could:

- Tune the hyperparameters
- Explore deeper and wider models
- Use reinforcement learning
- Also predict steering and test it on GTA or other racing games.
- Explore more on preventing overfitting

## References

* [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)
* [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767)
* [End to End Learning for Self-Driving Cars - Bojarski et al.](https://arxiv.org/abs/1604.07316)
* [Large-scale Video Classification with Convolutional Neural Network - Karpathy et al.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
* [An augmentation based deep neural network approach to learn human driving behavior](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)
