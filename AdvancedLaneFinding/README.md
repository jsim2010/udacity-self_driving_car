# Advanced Lane Finding Project

See <https://github.com/udacity/CarND-Advanced-Lane-Lines> for installation instructions.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
## Writeup

Here I will consider the [rubric](https://review.udacity.com/#!/rubrics/571/view) points individually and describe how I addressed each point in my implementation.  

### Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing `object_points`, which are the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Every time I successfully detect all chessboard corners in a test image, I append those points to `image_points`.

I then use `object_points` and `image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the `calibration13.jpg` test image using the `cv2.undistort()` function and obtained this result:

#### Distorted
![alt text][distorted]

#### Undistorted
![alt text][undistorted]

### Pipeline (single images)

To demonstrate the following steps, I will describe how I applied my pipeline to the following test image.

![alt text][test_image]

#### Provide an example of a distortion-corrected image.

Using `cv2.undistort()` with the previously calculated calibration matrix and distortion value resulted in the following.

![alt text][test_undistorted]


#### Describe how you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of my output for this step.

![alt text][test_binary]

#### Describe how you performed a perspective transform and provide an example of a transformed image.

`LaneDetector.transform_perspective()` uses `self.M` which was calculated in `LaneDetector.__init__()` using source (`src`) and destination (`dst`) points. Based on manual analysis of an image where the lane lines were straight, I chose to hardcode the source and destination points in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 262, 677      | 262, 720      | 
| 601, 447      | 262, 0        |
| 676, 447      | 1045, 0       |
| 1045, 677     | 1045, 720     |

I verified that my perspective transform was working as expected by drawing the source and destination points onto a second test image with straight lines and its warped counterpart to verify that the lines appear parallel in the warped image.

##### Source
![alt text][test_src]

##### Destination (Warped)
![alt text][test_dst]

#### Describe how you identified lane-line pixels and fit their positions with a polynomial?

I used `LaneDetector.detect_lines_points()` to detect the coordinates of the left and right lines. I did this by implementing the sliding window approach described in the "Advanced Techniques for Lane Finding" lesson. Then I used the coordinates to determine the best fit 2nd order polynomials for both the left and right line using `LaneDetector.get_left_line()` and `LaneDetector.get_right_line()`. Finally, `LaneDetector.draw_lane()` drew a polygon from the calculted lines onto the undistorted image as seen in the following image:

![alt text][test_lane]

#### Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To detect the radius of curvature of the lane, I used `LaneDetector.get_r_curve()` to convert the points to meters and calculate the radius of the resulting best fit curve. The position of the vehicle with respect to center was calculated by `LaneDetector.get_center_offset()`. Both of these functions return the average of their last 25 values respectively in order to lessen jumpiness.

#### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The result of plotting the lane and calculating the radius and offset is shown below:
![alt text][test_output]

### Pipeline (video)

#### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the main flaws of my implementation is that it does not do anything to attempt to smooth the lane detection. This means that changes in the image (such as when the car enters or exits a shadow) can cause large changes in the lane detected. This could be remedied by comparing a calculated line to the previous one and only using the calculated line if it is within a reasonable deviation.

Another flaw is that the detection of the right/white-dotted line is not very accurate at the farthest point away from the car. This is especially noticable when the lane turns to the right. Likely this is because the line turns too sharply and runs off the right edge of the warped image before hitting the top of the image. This is a case my current implementation does not handle. A more robust implementation would include a solution for this case.

[distorted]: ./images/distorted.jpg "Distorted"
[undistorted]: ./images/undistorted.jpg "Undistorted"
[test_image]: ./images/test_original.jpg "Test Image"
[test_undistorted]: ./images/test_undistorted.jpg "Undistorted Example"
[test_binary]: ./images/test_binary.jpg "Binary Example"
[test_src]: ./images/test_src.jpg "Test Straight Lines"
[test_dst]: ./images/test_dst.jpg "Test Straight Lines - Warped"
[test_fit]: ./image/test_fit.jpg "Test Fit"
[test_lane]: ./images/test_lane.jpg "Test Lane"
[test_output]: ./images/test_output.jpg "Test Output"
