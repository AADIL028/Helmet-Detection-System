from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob

xml_ = glob.glob('DataSet/annotations/*.xml')

lstData = []
for data in xml_:
    # read content of xml file
    with open(data) as f:
        d = f.read()
        xml_string = d

    # parse xml content
    root = ET.fromstring(xml_string)

    for child in root:
        # check for class label 'With Helmet' or 'Without Helmet'
        if child.tag == 'object' and (
                child.find('name').text == 'With Helmet' or child.find('name').text == 'Without Helmet'):
            # add it to list
            lstData.append(child.find('name').text)
            break

images = os.listdir('DataSet/images/')
images = pd.DataFrame(images)
image = list(np.array(images).flatten())

# create a DataFrame with filename and corresponding class label
data = pd.DataFrame({'filename': image[:761], 'classes': lstData})

# data augmentation
data_gen = ImageDataGenerator(rotation_range=0.2, zoom_range=0.2,
                              horizontal_flip=True, rescale=1 / 255)

# create a data generator for training data from the DataFrame
train_data = data_gen.flow_from_dataframe(data, 'DataSet/images/', x_col='filename', y_col='classes',
                                          target_size=(100, 100))

# build a sequential model for a CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(120, 120, 3)))
model.add(MaxPool2D())
model.add(Conv2D(64, (3, 3)))
model.add(MaxPool2D())
model.add(Conv2D(128, (3, 3)))
model.add(MaxPool2D())
model.add(Conv2D(128, (3, 3)))
model.add(MaxPool2D())
model.add(Conv2D(32, (3, 3)))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(train_data, epochs=5, batch_size=5)

# save model
# model.save("helmet_detector.model", save_format="h5")
