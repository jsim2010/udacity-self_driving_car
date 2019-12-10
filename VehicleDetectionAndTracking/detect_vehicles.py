import collections
import cv2
import glob
from moviepy.editor import VideoFileClip
import numpy as np
import pickle
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def convert_color(image):
    color_conversion = cv2.COLOR_BGR2YCrCb

    return cv2.cvtColor(image, color_conversion)


def draw_labeled_boxes(image, labels):
    draw_image = np.copy(image)

    for car in range(1, labels[1] + 1):
        activated_pixels = (labels[0] == car).nonzero()
        x_values = np.array(activated_pixels[1])
        y_values = np.array(activated_pixels[0])
        cv2.rectangle(draw_image, (np.min(x_values), np.min(y_values)), (np.max(x_values), np.max(y_values)), (0, 0, 255), 6)

    return draw_image


class VehicleDetector:
    def __init__(self):
        # Hyperparameters
        self.orientations = 8
        self.pixels_per_cell = 8
        self.cells_per_block = 8
        self.scales = [1.25, 1.5, 2]
        self.heat_threshold = 55
        self.heat_memory = 10
        self.cells_per_step = 1
        self.search_top = 400
        self.search_bottom = 650
        self.search_left = 725

        self.sampling_rate = 64
        self.previous_heats = collections.deque(maxlen=self.heat_memory)
        self.svc, self.scaler = self.get_svc_and_scaler()
        self.blocks_per_window = (self.sampling_rate // self.pixels_per_cell) - self.cells_per_block + 1

    def get_svc_and_scaler(self):
        try:
            trained_info_file = open("trained_info.p", "rb")
        except FileNotFoundError:
            trained_info = {}

            vehicle_image_files = glob.glob('vehicles/**/*.png')
            non_vehicle_image_files = glob.glob('non-vehicles/**/*.png')
            vehicle_features = self.extract_features(vehicle_image_files)
            non_vehicle_features = self.extract_features(non_vehicle_image_files)

            X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
            y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

            trained_info["scaler"] = StandardScaler().fit(X_train)
            X_train = trained_info["scaler"].transform(X_train)
            X_test = trained_info["scaler"].transform(X_test)
            trained_info["svc"] = LinearSVC(C=0.01)
            trained_info["svc"].fit(X_train, y_train)
            pickle.dump(trained_info, open("trained_info.p", "wb"))
        else:
            trained_info = pickle.load(trained_info_file)

        return trained_info["svc"], trained_info["scaler"]

    def channel_hog(self, channel, feature_vector):
        return hog(channel, orientations=self.orientations, pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell), cells_per_block=(self.cells_per_block, self.cells_per_block), block_norm="L2-Hys", feature_vector=feature_vector, visualise=True)

    def image_hog(self, image, feature_vector):
        hogs = []

        for channel in range(3):
            hogs.append(self.channel_hog(image[:,:,channel], feature_vector))

        return hogs

    def bin_spatial(self, image, size=32):
        return cv2.resize(image, (size, size)).ravel()

    def extract_features(self, image_files):
        features = []

        for image_file in image_files:
            image = cv2.imread(image_file)
            image = convert_color(image)
            hogs = self.image_hog(image, True)

            hog_features = np.ravel(np.hstack((hogs[0], hogs[1], hogs[2])))
            spatial_features = self.bin_spatial(image)


            features.append(np.hstack((hog_features, spatial_features)))

        return features

    def search_image(self, image):
        boxes = []
        image_to_search = convert_color(image[self.search_top:self.search_bottom,self.search_left:,:])

        for scale in self.scales:
            if scale != 1:
                image_shape = image_to_search.shape
                scaled_image = cv2.resize(image_to_search, (np.int(image_shape[1] / scale), np.int(image_shape[0] / scale)))
            else:
                scaled_image = image_to_search

            x_blocks = (scaled_image.shape[1] // self.pixels_per_cell) - self.cells_per_block + 1
            y_blocks = (scaled_image.shape[0] // self.pixels_per_cell) - self.cells_per_block + 1 

            x_steps = (x_blocks - self.blocks_per_window) // self.cells_per_step + 1
            y_steps = (y_blocks - self.blocks_per_window) // self.cells_per_step + 1
            hogs = self.image_hog(scaled_image, False)

            for x_step in range(x_steps):
                for y_step in range(y_steps):
                    top = y_step * self.cells_per_step
                    bottom = top + self.blocks_per_window
                    left = x_step * self.cells_per_step
                    right = left + self.blocks_per_window
                    top_pixel = top * self.pixels_per_cell
                    left_pixel = left * self.pixels_per_cell

                    hog_feat0 = hogs[0][top:bottom, left:right]
                    hog_feat1 = hogs[1][top:bottom, left:right]
                    hog_feat2 = hogs[2][top:bottom, left:right]
                    hog_features = np.ravel(np.hstack((hog_feat0, hog_feat1, hog_feat2)))


                    patch_image = cv2.resize(scaled_image[top_pixel:top_pixel+self.sampling_rate, left_pixel:left_pixel+self.sampling_rate], (self.sampling_rate, self.sampling_rate))
                    spatial_features = self.bin_spatial(patch_image)

                    features = self.scaler.transform(np.hstack((hog_features, spatial_features)).reshape(1, -1))

                    if self.svc.predict(features) == 1:
                        box_left = np.int(left_pixel * scale) + self.search_left
                        box_top = np.int(top_pixel * scale) + self.search_top
                        box_size = np.int(self.sampling_rate * scale)
                        boxes.append(((box_left, box_top), (box_left + box_size, box_top + box_size)))

        return boxes

    def build_heatmap(self, image, boxes):
        heat = np.zeros_like(image)
        heatmap = np.zeros_like(image)

        for box in boxes:
            heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        self.previous_heats.append(heat)

        for heat in self.previous_heats:
            heatmap += heat

        heatmap[heatmap <= self.heat_threshold] = 0
        return heatmap

    def pipeline(self, image):
        boxes = self.search_image(image)
        heatmap = self.build_heatmap(image, boxes)
        draw_image = draw_labeled_boxes(image, label(heatmap))
        return draw_image


detector = VehicleDetector()

video = VideoFileClip('project_video.mp4')
output = video.fl_image(detector.pipeline)
output.write_videofile('output.mp4', audio=False)
