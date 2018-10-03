# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:00:21 2018

@author: hp
"""


#1 CREATING CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing CNN

classifier = Sequential()

#Adding Convolutional Layers

classifier.add(Convolution2D(32,( 3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Max Pooling

classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten

classifier.add(Flatten())

#Full Connection

classifier.add(Dense(output_dim = 64, activation = "relu"))

classifier.add(Dense(output_dim = 1, activation = "sigmoid"))

#compiling CNN

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


#2 Fitting Images to CNN

from keras.preprocessing.image import ImageDataGenerator

Train_data_gen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

Test_data_gen = ImageDataGenerator(rescale = 1./255)

Train = Train_data_gen.flow_from_directory("New folder",
                                           target_size = (64,64),
                                           batch_size = 32,
                                           class_mode = "binary")

Test = Test_data_gen.flow_from_directory("test",
                                         target_size = (64,64),
                                         batch_size = 32,
                                         class_mode = "binary")

classifier.fit_generator(Train, samples_per_epoch = 71,
                         nb_epoch = 10,
                         validation_data = Test,
                         nb_val_samples = 9)