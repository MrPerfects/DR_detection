# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img
datagen = ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    rescale=1./255
)
img = load_img("F:\\data\\0\\10_left.jpeg")
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 10:
        break  # 否则生成器会退出循环