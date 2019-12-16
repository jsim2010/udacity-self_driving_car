# Finding Lane Lines on the Road

## Installation

Set up the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md).

## Goals

* Detect lane lines in images (taken from a video stream) using Python and OpenCV.

## Outcome

Running `find_lines.py` takes `input/solidWhiteRight.mp4` and outputs `output/solidWhiteRight.gif` with the line lines marked. The output is displayed below:

![Output video](output/solidWhiteRight.gif)

## Reflection

### Description of the Pipeline

My pipeline consisted of the following steps:

1. Convert the given image to grayscale.
2. Perform gaussian blur on the image.
3. Run the resulting image through a Canny edge detector.
4. Mask the image to only show a region of interest.
5. Use the Hough transform to generate lines and draw lines.

In order to draw a single line on the left and right lanes, I modified the provided `draw_lines()` function by first splitting the lines into left and right lines by determining the slope (positive or negative) of each line. Then for each line, I calculated the x values if the line was extended to the top and bottom of the region of interest. All valid x values were averaged, resulting in 4 points, 2 to form the left line and 2 to form the right line.

### Potential Shortcomings

One potential shortcoming would be what would happen when the orientation of the image does not match that of the chosen videos. For example, when my pipeline is used for the optional challenge, the region of interest is too large to be able to fully understand the sharp curves.

Another shortcoming could be that all lines in the region of interest are assumed to be a lane line, with no filter to remove lines that are obviously invalid.

### Possible Improvements

A possible improvement would be to remove slight jumpiness of the lane lines. This could be done by filtering any extraneous edges as well as by adding historesis to the pipeline so that previous images are factored into the calculation.

Another potential improvement could be to make the shape of the region of interest more dynamic so that the pipeline can additionally handle cases such as the one given by the optional challenge.
