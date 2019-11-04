import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Parameters
DROP_RATE = 0.5
STEERING_CORRECTION = 0.2
BATCH_SIZE = 32
NUMBER_EPOCHS = 7

input_data = []


def add_log_data(data, directory):
    """Adds log data from driving_log.csv file located in directory."""
    with open('{:s}/driving_log.csv'.format(directory)) as csv_file:
        reader = csv.reader(csv_file)

        # Skip header row
        next(reader)

        for line in reader:
            data.append(line)

    return data


def get_image(image_file):
    return cv2.imread('IMG/' + image_file.split('/')[-1])


def generator(data):
    num_data = len(data)

    while True:
        shuffle(data)

        for offset in range(0, num_data, BATCH_SIZE):
            batch_data = data[offset:offset + BATCH_SIZE]
            images = []
            steering_measurements = []

            for datum in batch_data:
                center_steering = float(datum[3])

                # Center image
                image = get_image(datum[0])
                images.append(image)
                steering_measurements.append(center_steering)

                images.append(cv2.flip(image, 1))
                steering_measurements.append(-center_steering)

                # Left image
                image = get_image(datum[1])
                left_steering = center_steering + STEERING_CORRECTION
                images.append(image)
                steering_measurements.append(left_steering)

                images.append(cv2.flip(image, 1))
                steering_measurements.append(-left_steering)

                # Right image
                image = get_image(datum[2])
                right_steering = center_steering - STEERING_CORRECTION
                images.append(image)
                steering_measurements.append(right_steering)

                images.append(cv2.flip(image, 1))
                steering_measurements.append(-right_steering)

            X_train = np.array(images)
            y_train = np.array(steering_measurements)
            yield shuffle(X_train, y_train)


input_data = add_log_data(input_data, 'data')
train_data, validation_data = train_test_split(input_data, test_size=0.2)
train_generator = generator(train_data)
validation_generator = generator(validation_data)

model = Sequential()

# Normalize the data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# Crop the image
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
# 5x5 Convolution layer with depth of 24
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
# 5x5 Convolutional layer with depth of 36
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
# 5x5 Convolutional layer with depth of 48
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
# 3x3 Convolution layer with depth of 64
model.add(Convolution2D(64, 3, 3, activation='relu'))
# 3x3 Convolution layer with depth of 64
model.add(Convolution2D(64, 3, 3, activation='relu'))
# Flatten
model.add(Flatten())
model.add(Dropout(DROP_RATE))
# Fully connected layer with 100 neurons
model.add(Dense(100))
model.add(Dropout(DROP_RATE))
# Fully connected layer with 50 neurons
model.add(Dense(50))
model.add(Dropout(DROP_RATE))
# Fully connected layer with 10 neurons
model.add(Dense(10))
model.add(Dropout(DROP_RATE))
# Output neuron
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# Multiply data len by 6 because generator produces 6 outputs from every 1 line:
# Center, center flipped, left, left flipped, right, right flipped
model.fit_generator(train_generator, samples_per_epoch=len(train_data) * 6, validation_data=validation_generator, nb_val_samples=len(validation_data) * 6, nb_epoch=NUMBER_EPOCHS)

model.save('model.h5')
