# Traffic Sign Recognition

The goals / steps of this project are the following:
* Load the data set (which can be found at <https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip>).
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written [report](#report)

---

## Report
Here I will consider the [rubric criteria](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

### The project submission includes all required files.

- [Ipython notebook](Traffic_Sign_Classifier.ipynb) with code
- [HTML output](Traffic_Sign_Classifier.html) of the code
- A writeup [report](#report)

### The submission includes a basic summary of the data set.

* The size of the training set is `34799`.
* The size of the validation set is `4410`.
* The size of the test set is `12630`.
* The dimensions of a traffic sign image is `32x32x3`.
* The number of unique classes/labels in the data set is `43`.

### The submission includes an exploratory visualization on the dataset.

Below is a bar chart showing how many images each class has in the training data.

![alt text][visualization]

### The submission describes the preprocessing techniques used and why these techniques were chosen.

My preprocessing was to normalize the image data. This is standard for the input data to a neural network because it reduces potential error from floating point calculations and makes it much easier for the weights optimizer.

### The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer.

My final model consisted of the following layers:

| Layer | Description |
|:---|:---|
| Input | 32x32x3 RGB image |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU | |
| Max Pooling | 2x2 stride, valid padding, outputs 14x14x6 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU | |
| Max Pooling | 2x2 stride, valid padding, outputs 5x5x16 |
| Flatten | outputs 400 |
| Fully Connected | ouputs 120 |
| RELU | |
| Dropout | |
| Fully Connected | outputs 84 |
| RELU | |
| Dropout | |
| Fully Connected | outputs 43 (number of classes) |

### The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

To train the model, I used 15 epochs with a batch size of 128. I used the Adam Optimizer provided by TensorFlow with a learning rate of 0.001. While training, the dropout layers had a 0.5 keep probability.

### The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

I started with the LeNet architecture. I choose this architecture because I was familiar with it and it has been used and proved in many cases. Validating using the original LeNet architecture plateaued around 89% accuracy. However, the accuracy of the model with the training data was over 98%. This indicated that the model was overfitting; in order to compensate, I added a dropout layer after each of the first 2 fully-connected layers. Because the dropout layers reduce the amount of data available to the layers following them, the model was able to detect more features that define each class and reduce the amount of overfit.

The accuracy results of my final model were:

| Type | Percentage |
| --- | --- |
| Training | 99.2 % |
| Validation | 94.5 % |
| Test | 94.0% |

### The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3]
![alt text][image4] ![alt text][image5]

The first 3 images are all Speed limit signs and thus have very similar qualities. Classifying them correctly relies on the model correctly identifying the number in the sign. Image 4, the Yield sign, has a very distinguishable shape and colors so it should be relatively easy to classify. Image 5, the Stop sign, is very blurry when compressed to 32x32 pixels and it may be hard for the model to detect the STOP letters in order to differentiate it from other red circles with white symbols.

### The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h) | Speed limit (50km/h) |
| Speed limit (70km/h) | Speed limit (20km/h) |
| Speed limit (100km/h) | Keep right |
| Yield | Yield |
| Stop | No entry |

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This accuracy differs greatly from the test accuracy of 94%.

### The top five softmax probabilities of the predictions on the captured images are outputted. The submission dicusses how certain or uncertain the model is of its predictions.

For the first image, the model is very confident the sign is a Speed limit (50km/h) sign, which is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .825         			| Speed limit (50km/h) |
| .151     				| Speed limit (80km/h) |
| .017					| Speed limit (60km/h) |
| .005	      			| Speed limit (100km/h) |
| .000				    | Speed limit (120km/h) |


For the second image, the model is very confident that the sign is a Speed limit (20km/h) sign. However, this is incorrect as the sign is actually a Speed limit (70km/h) sign. The model does rank the correct sign of Speed limit (70km/h) as second, although its probability is rather small (6.5%).

| Probability | Prediction |
|:---:|:---:|
| 0.884 | Speed limit (20km/h) |
| 0.065 | Speed limit (70km/h) |
| 0.049 | Speed limit (30km/h) |
| 0.001 | Go straight or right |
| 0.000 | End of all speed and passing limits |

For the third sign, the model is very closely split betweeen 2 options: Keep right (49.6%) and Dangerous curve to the right (44.9%). Neighter of these is the correct Speed limit (100km/h), which does not appear in the top 5 probabilities.

| Probability | Prediction |
|:---:|:---:|
| 0.496 | Keep right |
| 0.449 | Dangerous curve to the right |
| 0.026 | Road work |
| 0.010 | Ahead only |
| 0.007 | Slippery road |

For the fourth sign, the model has complete confidence that it is a Yield sign (100% when calculated to 6 significant figures), which is correct.

| Probability | Prediction |
|:---:|:---:|
| 1.000 | Yield |
| 0.000 | Speed limit (60km/h) |
| 0.000 | No passing |
| 0.000 | Keep right |
| 0.000 | No passing for vehicles over 3.5 metric tons |

For the fifth sign, the model is extremely confident that the sign is a No entry. Instead the sign is a Stop sign, which ranked third.

Probability | Prediction |
|:---:|:---:|
| 0.969 | No entry |
| 0.026 | Bicycles crossing |
| 0.002 | Stop |
| 0.002 | Speed limit (30km/h) |
| 0.000 | Priority road |

[image1]: ./image1.jpg "Traffic Sign 1"
[image2]: ./image2.jpg "Traffic Sign 2"
[image3]: ./image3.jpg "Traffic Sign 3"
[image4]: ./image4.jpg "Traffic Sign 4"
[image5]: ./image5.jpg "Traffic Sign 5"
[visualization]: ./visualization.png "Visualization"
