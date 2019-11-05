import cv2
import glob
from moviepy.editor import VideoFileClip
import numpy as np


class LaneDetector:
    def __init__(self):
        self.calibration_matrix = None
        self.distortion = None

        # Calculate perspective transform matrices
        src = np.float32([(262, 677), (601, 447), (676, 447), (1045, 677)])
        dst = np.float32([(262, 720), (262, 0), (1045, 0), (1045, 720)])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        self.meter_per_pixel_x = 3.7 / (1045 - 262)
        self.meter_per_pixel_y = 3 / 70
        self.r_curves = []
        self.center_offsets = []
        self.avg_count = 25

    def calibrate(self):
        calibration_image_filenames = glob.glob('camera_cal/calibration*.jpg')
        object_points = []
        image_points = []
        columns = 9
        rows = 6

        points_3d = np.zeros((rows * columns, 3), np.float32)
        points_3d[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

        for image_filename in calibration_image_filenames:
            calibration_image = cv2.imread(image_filename)
            gray_image = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image, (columns, rows), None)

            if ret:
                image_points.append(corners)
                object_points.append(points_3d)

        ret, self.calibration_matrix, self.distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray_image.shape[::-1], None, None)


    def correct_distortion(self, image):
        return cv2.undistort(image, self.calibration_matrix, self.distortion, None, self.calibration_matrix)

    def generate_binary(self, image):
        sobel_threshold_min = 20
        sobel_threshold_max = 100
        saturation_threshold_min = 170
        saturation_threshold_max = 255

        # Red Sobel x
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
        absolute_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * absolute_sobel / np.max(absolute_sobel))
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= sobel_threshold_min) & (scaled_sobel <= sobel_threshold_max)] = 1

        # Saturation
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        saturation_channel = hls[:, :, 2]
        saturation_binary = np.zeros_like(saturation_channel)
        saturation_binary[(saturation_channel >= saturation_threshold_min) & (saturation_channel < saturation_threshold_max)] = 1

        # Combine the two binary thresholds
        binary = np.zeros_like(sobel_binary)
        binary[(saturation_binary == 1) | (sobel_binary == 1)] = 1
        return binary

    def transform_perspective(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def find_line_starts(self, image):
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        left_start = np.argmax(histogram[:midpoint])
        right_start = np.argmax(histogram[midpoint:]) + midpoint
        return left_start, right_start

    def get_activated_indexes(self, image):
        activated = image.nonzero()
        activated_y = np.array(activated[0])
        activated_x = np.array(activated[1])
        return activated_x, activated_y

    def find_line_indexes(self, image, activated_x, activated_y):
        num_windows = 9
        margin = 100
        min_pixels = 50
        window_height = np.int(image.shape[0] // num_windows)
        left_line, right_line = self.find_line_starts(image)
        left_indexes = []
        right_indexes = []

        for window_index in range(num_windows):
            window_top = image.shape[0] - (window_index + 1) * window_height
            window_bottom = image.shape[0] - window_index * window_height
            left_window_left = left_line - margin
            left_window_right = left_line + margin
            right_window_left = right_line - margin
            right_window_right = right_line + margin

            left_indexes.append(((activated_y >= window_top) & (activated_y < window_bottom) & (activated_x >= left_window_left) & (activated_x < left_window_right)).nonzero()[0])
            right_indexes.append(((activated_y >= window_top) & (activated_y < window_bottom) & (activated_x >= right_window_left) & (activated_x < right_window_right)).nonzero()[0])

            if len(left_indexes[-1]) > min_pixels:
                left_line = np.int(np.mean(activated_x[left_indexes[-1]]))

            if len(right_indexes[-1]) > min_pixels:
                right_line = np.int(np.mean(activated_x[right_indexes[-1]]))

        return np.concatenate(left_indexes), np.concatenate(right_indexes)

    def detect_line_points(self, image):
        activated_x, activated_y = self.get_activated_indexes(image)
        left_indexes, right_indexes = self.find_line_indexes(image, activated_x, activated_y)

        return (activated_x[left_indexes], activated_y[left_indexes]), (activated_x[right_indexes], activated_y[right_indexes])

    def get_left_line(self, image, line_points):
        lane_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        line_fit = np.polyfit(line_points[1], line_points[0], 2)
        lane_x = line_fit[0] * lane_y ** 2 + line_fit[1] * lane_y + line_fit[2]
        return np.array([np.transpose(np.vstack([lane_x, lane_y]))])

    def get_right_line(self, image, line_points):
        lane_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        line_fit = np.polyfit(line_points[1], line_points[0], 2)
        lane_x = line_fit[0] * lane_y ** 2 + line_fit[1] * lane_y + line_fit[2]
        return np.array([np.flipud(np.transpose(np.vstack([lane_x, lane_y])))])

    def draw_lane(self, image, warped, left_line, right_line):
        blank_warped = np.zeros_like(warped).astype(np.uint8)
        lane_image = np.dstack((blank_warped, blank_warped, blank_warped))
        cv2.fillPoly(lane_image, np.int_([np.hstack((left_line, right_line))]), (0, 255, 0))
        unwarped_lane_image = cv2.warpPerspective(lane_image, self.Minv, (image.shape[1], image.shape[0]))
        return cv2.addWeighted(image, 1, unwarped_lane_image, 0.3, 0)

    def get_r_curve(self, image, left_points, right_points):
        # Calculate fit in terms of meters
        left_fit = np.polyfit(left_points[1] * self.meter_per_pixel_y, left_points[0] * self.meter_per_pixel_x, 2)
        right_fit = np.polyfit(right_points[1] * self.meter_per_pixel_y, right_points[0] * self.meter_per_pixel_x, 2)
        y_value = image.shape[0] * self.meter_per_pixel_y

        left_r_curve = ((1 + (2 * left_fit[0] * y_value + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_r_curve = ((1 + (2 * right_fit[0] * y_value + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        self.r_curves.append((left_r_curve + right_r_curve) / 2)
        return sum(self.r_curves[-self.avg_count:]) / min(len(self.r_curves), self.avg_count)

    def get_center_offset(self, image, left_line, right_line):
        car_center = image.shape[1] / 2
        # bottom of left_line is at end
        # bottom of right_line is at beginning
        lane_center = (left_line[0][-1][0] + right_line[0][0][0]) / 2
        self.center_offsets.append((car_center - lane_center) * self.meter_per_pixel_x)
        return sum(self.center_offsets[-self.avg_count:]) / min(len(self.center_offsets), self.avg_count)

    def pipeline(self, image):
        image = self.correct_distortion(image)
        binary = self.generate_binary(image)
        warped = self.transform_perspective(binary)
        left_points, right_points = self.detect_line_points(warped)
        left_line = self.get_left_line(warped, left_points)
        right_line = self.get_right_line(warped, right_points)
        lane_image = self.draw_lane(image, warped, left_line, right_line)
        r_curve = self.get_r_curve(warped, left_points, right_points)
        center_offset = self.get_center_offset(warped, left_line, right_line)
        cv2.putText(lane_image, "r: {:f}".format(r_curve), (0, image.shape[0]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        cv2.putText(lane_image, "offset: {:f}".format(center_offset), (image.shape[1] // 2, image.shape[0]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        return lane_image

lane_detector = LaneDetector()

lane_detector.calibrate()
video = VideoFileClip('project_video.mp4')
output = video.fl_image(lane_detector.pipeline)
output.write_videofile('output.mp4', audio=False)
