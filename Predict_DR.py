# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing import image
from PIL import Image as pil_image
import numpy as np
import cv2
import h5py
import os
import fttlutils
import sys
sys.path.append("./pre_")
from cropblack import crop
from keras.applications.inception_v3 import preprocess_input
class Predict:

    def __init__(self,path,img):
        self.Model_path = path
        self.img = img
        # self.xs = []
    def Load_img(self):
        xs = []
        img = image.load_img(self.img,target_size=None)
        img_ = crop(img)
        img = img_.resize((512,512),pil_image.NEAREST)
        img0 = image.img_to_array(img)
        img4d1 = np.expand_dims(img0, axis=0)

        # img4d1 /=255
        img4d = preprocess_input(img4d1)
        img4d1 = fttlutils.dataset_normalized(img4d)
        # xs.append(img4d[0])
        # X = np.array(xs)
        X = img4d1
        # X = np.array(X)
        # print(X.shape)
        return X

    def load_hdf5(self,infile):
        with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
            return f["image"][()]

    def Evaluate(self,x):

        class_ = Model.predict(x)

        # print(class_)
        return class_

if __name__ == "__main__":
    # pre_class = 3
    path = "F:\\wzz_code\\DR_detection\\v3\\v3_model_h5\\best_model.h5"
    img = "F:\\data\\TL_data_plus_balance\\3\\"
    Model = load_model(path)
    print("Model is load done..................")
    count = 0
    total = len(os.listdir(img))
    print("total = :", total)
    for test in os.listdir(img):
        img0 = os.path.join(img,test)
        P = Predict(path, img0)
        x = P.Load_img()
        # print(x.shape)
        # cv2.imshow("test",x[0])
        # cv2.waitKey(0)
        class_ = P.Evaluate(x)
        # print(class_,class_[0])
        pre = np.argmax(class_[0])
        print("预测：",pre)
        if pre!=3:
            count = count+1
            print(img0)
        print("error",count)


    # test_img = "F:\\wzz_code\\DR_detection\\v3\\test_img.h5"
    # test_label = "F:\\wzz_code\\DR_detection\\v3\\test_label.h5"
    # X = P.load_hdf5(test_img)
    # Y = P.load_hdf5(test_label)
    # Y_ture = np.argmax(Y,axis=1)
    # print("one-hot_before",Y)
    # print("label__",Y_ture)
    # Y_pre = P.Evaluate(X)
    # Y_pre = np.argmax(Y_pre,axis=1)
    # print(Y_pre)




