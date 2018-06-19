from __future__ import print_function
from keras.layers import Permute, Conv2D, MaxPooling2D, Input, Multiply, Reshape, Dense, Flatten, Activation, Dropout, Lambda
from keras.models import Model
from keras.datasets import cifar10
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import os
import numpy as np

input_dim = (32,32,3)

img_inputs = Input(shape=input_dim)
conv1 = Conv2D(1, (1,1), padding='same', activation='relu')(img_inputs)     #1
conv64 = Conv2D(9, (3,3), padding='same', activation='relu')(conv1)        #2

"""
This found the mistake of softmax returning array on 1s. This solution will
basically find the row importance and then the features' importance.
"""
y = Conv2D(1, (1,1))(conv64) # 32x32x1 ?
y = Permute((3, 2, 1))(y)
y = Dense(32, activation='softmax')(y)
y = Permute((1, 3, 2))(y)
y = Dense(32, activation='softmax')(y)

#now permute back
y = Permute((1, 3, 2))(y)
y = Permute((3, 2, 1))(y)


# mult = Multiply()([conv64,y])
#
# summed = Lambda(lambda x: K.sum(x, axis=(1,2)), output_shape=lambda s: (s[0], s[3]))(mult)
#
# dense5 = Dense(64, activation='relu')(summed)
# final = Dense(10, activation='softmax')(dense5)

#Finalization
model = Model(inputs=img_inputs, outputs=y)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x = np.array([x_train[0]])
layer_outputs = [layer.output for layer in model.layers]
viz_model = Model(input=model.input, output=layer_outputs)
features = viz_model.predict(x)
for feature_map in features:
    print(feature_map.shape)

# print(features[2] == features[5])
print(features[9])
