# importing some useful packages
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

# helper functions
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlue(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2] # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by vertices with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def calc_x(slope, intercept, y):
    """
    Calculates x coordinate from `y` given a line defined by `slope` and `intercept`.
    """
    return (y - intercept) / slope

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane.

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line. Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    positive_slope_bottom_x = 0
    positive_slope_top_x = 0
    positive_slope_count = 0
    negative_slope_bottom_x = 0
    negative_slope_top_x = 0
    negative_slope_count = 0

    # Set y values the defined top and bottom of lane lines.
    # Setting top_limit slightly under the top of the region_of_interest performs better than setting top_limit equal to region_of_interest
    top_limit = 325
    bottom_limit = img.shape[0]

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            bottom_x = calcx(slope, intercept, bottom_limit)
            top_x = calc_x(slope, intercept, top_limit)

            # Ensure that calculated x's are valid
            if 0 < bottom_x < img.shape[1] and 0 < top_x < img.shape[1]:
                if slope > 0:
                    positive_slope_count += 1
                    positive_slope_bottom_x += bottom_x
                    positive_slope_top_x += top_x
                elif slope < 0:
                    negative_slope_count += 1
                    negative_slope_bottom_x += bottom_x
                    negative_slope_top_x += top_x

    if positive_slope_count:
        cv2.line(img, (int(positive_slope_bottom_x / positive_slope_count), bottom_limit), (int(positive_slope_top_x / positive_slope_count), top_limit), color, thickness=10)

    if negative_slope_count:
        cv2.line(img, (int(negative_slope_bottom_x / negative_slope_count), bottom_limit), (int(negative_slope_top_x / negative_slope_count), top_limit), color, thickness=10)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a=0.8, B=1., y=0.):
    """
    `img` is the output of the hough_lines(), An image iwth lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * a + img * B + y
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, B, y)

def pipeline(image):
    gray_image = grayscale(image)
    blur_image = gaussian_blue(gray_image, 5)
    edge_image = canny(blur_image, 50, 150)
    interesting_edges = region_of_interest(edge_image, np.array([[
        (150, image.shape[0]),
        (450, 300),
        (525, 300),
        (image.shape[1], image.shape[0])
    ]]))
    lines = hough_lines(interesting_edges, 1, np.pi / 180, 15, 40, 40)
    image_with_lines = weighted_img(lines, image)

    return image_with_lines

clip = VideoFileClip("input/solidWhiteRight.mp4")
white_clip = clip.fl_image(pipeline)
white_clip.write_gif("output/solidWhiteRight.gif")
