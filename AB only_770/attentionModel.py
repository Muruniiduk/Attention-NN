from __future__ import print_function
from keras.layers import BatchNormalization, Permute, Conv2D, MaxPooling2D, Input, Multiply, Reshape, Dense, Flatten, Activation, Dropout, Lambda
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

#setup Keras
batch_size = 32 #64?
num_classes = 10
epochs = 50
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'
#TensorBoard
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
histogram_freq=0, write_graph=True, write_images=True)

#prepare data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_dim = (32,32,3)

#Model
"""
I am trying to build a model that wouldn't use maxpooling. For that I will sum
the values in the layer after the filtering (multiplying with the Attention
Block) and after that using a dense network that is reshaped.

32x32x3 -> (1x1) conv 32x32x1 -> (3x3)x64 conv(same padding) 32x32x64 ->
---> Attention block will be small dense network with conv and softmax
---> Attention block * last layer ---> sum over 32x32 values and get 64
neurons -> Bigger dense network -> softmax -> prediction
"""

img_inputs = Input(shape=input_dim)
conv1 = Conv2D(1, (1,1), padding='same', activation='relu')(img_inputs)     #1
conv = Conv2D(128, (4,4), padding='same', activation='relu')(conv1)        #2
conv = BatchNormalization()(conv)


# #Attention 1
# y = Conv2D(1, (1,1))(conv) # 32x32x1 ?
# y = Permute((3, 2, 1))(y)
# y = Dense(32, activation='softmax')(y)
# y = Permute((1, 3, 2))(y)
# y = Dense(32, activation='softmax')(y)
# y = Permute((1, 3, 2))(y) #now permute back
# y = Permute((3, 2, 1))(y) #end attention
#
#
# mult = Multiply()([conv,y])
# pooled = MaxPooling2D(pool_size=(2,2))(mult)
pooled = MaxPooling2D(pool_size=(2,2))(conv)
pooled = Dropout(0.2)(pooled)
conv = Conv2D(128, (3,3), padding='same', activation='relu')(pooled)
conv = BatchNormalization()(conv)

# #Attention 2
# y = Conv2D(1, (1,1))(conv) # 32x32x1 ?
# y = Permute((3, 2, 1))(y)
# y = Dense(16, activation='softmax')(y)
# y = Permute((1, 3, 2))(y)
# y = Dense(16, activation='softmax')(y)
# y = Permute((1, 3, 2))(y) #now permute back
# y = Permute((3, 2, 1))(y) #end attention
#
# mult = Multiply()([conv,y])
# pooled = MaxPooling2D(pool_size=(2,2))(mult)
pooled = MaxPooling2D(pool_size=(2,2))(conv)
pooled = Dropout(0.3)(pooled)
conv = Conv2D(32, (3,3), padding='same', activation='relu')(pooled) # 32 -> 256
conv = BatchNormalization()(conv)

# #Attention 3
# y = Conv2D(1, (1,1))(conv) # 32x32x1
# y = Permute((3, 2, 1))(y)
# y = Dense(8, activation='softmax')(y)
# y = Permute((1, 3, 2))(y)
# y = Dense(8, activation='softmax')(y)
# y = Permute((1, 3, 2))(y) #now permute back
# y = Permute((3, 2, 1))(y) #end attention
#
# mult = Multiply()([conv,y])

# summed = Lambda(lambda x: K.sum(x, axis=(1,2)), output_shape=lambda s: (s[0], s[3]))(mult)
summed = Lambda(lambda x: K.sum(x, axis=(1,2)), output_shape=lambda s: (s[0], s[3]))(conv)
# pooled = MaxPooling2D(pool_size=(2,2))(conv)
# summed = Flatten()(conv)

#Dense network with input of 64 neurons -> hidden -> 10 neurons w/ softmax
dense = Dense(256, activation='relu')(summed)
dense = Dense(64, activation='relu')(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dense(64, activation='relu')(dense)
final = Dense(10, activation='softmax')(dense)

#Finalization
model = Model(inputs=img_inputs, outputs=final)
#print shapes
x = np.array([x_train[0]])
layer_outputs = [layer.output for layer in model.layers]
viz_model = Model(input=model.input, output=layer_outputs)
features = viz_model.predict(x)
for feature_map in features:
    print(feature_map.shape)


model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy']) #DEFAULT: epochs=1, batch_size=32, #novalidation
model.fit(x_train, y_train, epochs=epochs, callbacks=[tbCallBack])

scores = model.evaluate(x_test, y_test, verbose=1)
model.save('best_model.h5')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
