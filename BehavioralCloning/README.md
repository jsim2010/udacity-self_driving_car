# Behavioral Cloning

See <https://github.com/udacity/CarND-Behavioral-Cloning-P3> for installation instructions.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

## Report

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### The submission includes a model.py file, drive.py, model.h5, a writeup report and video.mp4.

- [model.py](model.py)
- [drive.py](drive.py)
- [model.h5](model.h5)
- [video.mp4](video.mp4)
- [writeup report](#report)

### The model provided can be used to successfully operate the simulation.

Using the simulator provided by udacity and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Submission code is usable and readable

[`model.py`](model.py) contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### An appropriate model architecture has been employed

My model uses the architecture of the Nvidea pipeline. It starts with a normalized layer using a Keras lambda layer. Then it has 5 convolutional layers, each of which is followed by a RELU layer to introduce nonlinearity. The convolutional layers are then followed by 4 fully-connected layers.

### Attempts to reduce overfitting in the model

Each fully-connected layer is followed by a dropout layer in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### Appropriate training data

For the training data, I only used the sample data provided by Udacity. For details about how I used the training data, see the next section.

### Solution Design Approach

The overall strategy for deriving a model architecture was to start with a good basic model and add more/improved data as needed.

My first step was to use a convolution neural network model similar to the Nvidea pipeline. I thought this model might be appropriate because it is used by Nvidea to train a real autonomous car and it was recommended by several users on Slack. For the input data I used the center images of my data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. With the included dropout layers, I saw no signs that my model was overfitting.

When I tested my model with the simulator, I tended to find 2 trouble spots. The first was at the end of the first turn, where the car would not turn sharp enough. The second was at or immediately after the bridge where the car would drift to the left, either running into the wall or causing it to veer to the right as it reached the sharp left turn after the bridge.

I resolved the first issue by adding the left and right images of the data. These images provided an orientation that was off from the center. By adjusting the data's respective steering angle, the model was able to guide the car back towards the center of the lane when it got off the center.

For the second issue, I was able to resolve it by taking each of the images (center, left and right) and flipping them across the vertical axis while multiplying each image's steering angle by -1. As a result, every timestamp from the data generated 6 cases of input data. Flipping the images removed a slight bias to steer the car to the left, helping to keep the car in the center as it crossed the bridge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Final Model Architecture

The final model architecture consisted of the following layers:

| Layer | Description |
| --- | --- |
| Lambda | Normalizes the image data |
| Cropping | Crops 50 pixels from the top and 20 pixels from the bottom of each image |
| Convolution | kernel = 5x5; activation layer = RELU; depth = 24 |
| Convolution | kernel = 5x5; activation layer = RELU; depth = 36 |
| Convolution | kernel = 5x5; activation layer = RELU; depth = 48 |
| Convolution | kernel = 3x3; activation layer = RELU; depth = 64 |
| Convolution | kernel = 3x3; activation layer = RELU; depth = 64 |
| Flatten | dropout rate during training = 50%|
| Fully-connected | 100 neurons; dropout rate during training = 50% |
| Fully-connected | 50 neurons; dropout rate during training = 50% |
| Fully-connected | 10 neurons; dropout rate during training = 50% |
| Output | 1 output value |

### Creation of the Training Set & Training Process

For my training data, I only used the sample data provided by Udacity. I found that this data provided good images that generally kept the car in the center of the lane.

![alt text][image1]
![alt text][image2]

I shuffled all of the data and used 20% for validation. The ideal number of epochs was 7, which was roughly the time when there was no discernable difference in the loss rate. I used an adam optimizer so that manually training the learning rate wasn't necessary.

[image1]: ./img/center_2016_12_01_13_30_48_287.jpg "Image1"
[image2]: ./img/center_2016_12_01_13_33_06_411.jpg "Image2"
