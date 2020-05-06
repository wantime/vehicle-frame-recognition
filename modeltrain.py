# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:33:30 2020

@author: Wang
"""

#!/usr/bin/env python
# coding: utf-8

import skimage.io
import skimage.color
import skimage.transform

import numpy as np
import os
import random

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

DATASET_DIR = 'E:/code/python/chepai/dataset/carplate'#数据集的位置


classes = os.listdir(DATASET_DIR + "/ann/")#类别
data = []
for cls in classes: #这一段循环是给每一个图片增加类别
    files = os.listdir(DATASET_DIR + "/ann/"+cls)
    for f in files:
        img = skimage.io.imread(DATASET_DIR + "/ann/"+cls+"/"+f)
        img = skimage.color.rgb2gray(img)
        data.append({
            'x': img,
            'y': cls
        })
#这里增加自己的测试数据
#classes1 = os.listdir(TEST_DIR + "/ann/")#类别
#data1 = []


random.shuffle(data)    #打乱顺序


X = [d['x'] for d in data]  #获取图片
y = [d['y'] for d in data]  #获取图片类别

ys = list(np.unique(y))     #这里是获取图片的类别总数，0~9，A~Z
y = [ys.index(v) for v in y]    #这里把每个图片的类别用数字表示

x_train = np.array(X[:]) #训练集的图片
y_train = np.array(y[:]) #训练集的类别


x_test = np.array(X[int(len(X)*0.8):])  #测试集的图片
y_test = np.array(y[int(len(X)*0.8):])  #测试集的类别




batch_size = 128                    
num_classes = len(classes)      #类别的个数
epochs = 30

# input image dimensions
img_rows, img_cols = 20, 20     #输入图片的维度

def extend_channel(data):   #这里好像是对图片进行标准化，规范大小，但是后面又加上了一维，赋值为1
    if K.image_data_format() == 'channels_first':   #这一句判断不太明白
        data = data.reshape(data.shape[0], 1, img_rows, img_cols)
    else:
        data = data.reshape(data.shape[0], img_rows, img_cols, 1)
        
    return data

x_train = extend_channel(x_train)
x_test = extend_channel(x_test)

input_shape = x_train.shape[1:]

x_train = x_train.astype('float32')     #转换类型
x_test = x_test.astype('float32')       #转换类型
x_train /= 255                          #归一化
x_test /= 255                           #归一化
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train2 = keras.utils.to_categorical(y_train, num_classes)
y_test2 = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train2,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test2))

model.save('char_cnn_6G.h5')
score = model.evaluate(x_test, y_test2, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('char_cnn_6G.h5')
#model.save_weights('char_cnn.h5')
