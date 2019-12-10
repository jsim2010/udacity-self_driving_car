# Vehicle Detection and Tracking Project

See <https://github.com/udacity/CarND-Vehicle-Detection> for installation instructions.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier.
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---
## Writeup

Here I will consider the [rubric](https://review.udacity.com/#!/rubrics/513/view) points individually and describe how I addressed each point in my implementation.

### Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained [here](detect_vehicles.py#L93).

I started by reading in all the `vehicle` and `non-vehicle` images. Here is an example of each one:

| Type | Image |
| --- | --- |
| Vehicle | ![alt text][vehicle] |
| Non-Vehicle | ![alt text][non-vehicle] |

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, `cells_per_block`). I viewed the output from random images returned from `skimage.hog()` to get a feel for what the result looked like.

Here is an example using the `Y` channel of the `YCrCb` color space and HOG paramaters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=8, 8)`:

![alt text][hog]

#### Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and compared their accuracy with a test sample of the provided images. I found the best results occured from using `YCrCb` color space with 8 orienations, 8 pixels per cell, and 8 cells per block.

#### Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them.

The code for this step is located [here](detect_vehicles.py#L55).

I trained a linear SVM using the provided images of vehicles and non-vehicles. For each image, I extracted the HOG and spatial features of the image. These were then fed to the linear SVM along with the classification of the image.

### Sliding Window Search

#### Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search [here](detect_vehicles.py#L109).

I searched at scales 1.25, 1.5 and 2. These matched well with most of the images in the project and also kept the pipeline efficient by keeping the number of windows searched smaller. In order to detect a vehicle at every location, I set the overlap rather high at 87.5% (moving 1 of the 8 cells per window).

#### Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using `YCrCb` 3-channel HOG features plus spatially binned color in the feature vector. For each video frame, I extracted the HOG features once for each scale to avoid calculating HOG features for every window. I also limited the search to only locations where cars could be found in the project video. Below is an example of the detection boxes found for a test image.

### Video Implementation

#### Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal falso positives).

Here's a [link to my video result](./output.mp4).

#### Describe how (and identify where in your code) you implemented some kind of filter for falso positives and some method for combining overlapping bounding boxes.

The heatmap filter was located in code [here](detect_vehicles.py#L155). The method for combining overlapping bounding boxes is [here](detect_vehicles.py#L20).

I recorded the positions of positive detections in each frame of the video. Using positive detections of the 10 most recent frames, I created a heatmap and then applied a threshold to that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I assumed each blob corresponded to a vehicle and constructed bounding boxes to cover the area of each blob detected.

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The main problem I faced with my implementation is that it often detected areas along the median wall as a vehicle. I resolved this by not searching this area of the images. However, this has the negative side effect of making my pipeline not very general. Since my training resulted in over 99% accuracy, it seems likely that my pipeline could benefit from more images of median walls to improve the detection of those types of windows as non-vehicles.

Additionally, the bounding boxes are somewhat shaky. These could potentially be improved by adding a smaller scale that would better detect the edges of vehicles.

[vehicle]: ./images/vehicle.png "Vehicle"
[non-vehicle]: ./images/non-vehicle.png "Non-Vehicle"
[hog]: ./images/hog.jpg "HOG"
[boxes]: ./images/boxes.jpg "Boxes"
