# -*- coding: utf-8 -*-
# 直方图均衡化对图像预处理
import os
import cv2
import csv
from PIL import Image
import numpy as np
import cv2

def histogram(img):
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = cv2.equalizeHist(lab_planes[0])   #仅对L通道均衡化
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    return bgr


def clahe(img,cliplimit=None,gridsize = 8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr
if __name__ == "__main__":
    test = cv2.imread("F:\\data\\2\\1469_right.jpeg")

    test = cv2.resize(test,(612,512))
    hist_img = histogram(test)
    clahe_img = clahe(test, cliplimit=None, gridsize=8)


    show = np.concatenate((test,hist_img,clahe_img),axis=1)
    cv2.imwrite("show.jpg",show)
    cv2.imshow("show",show)
    cv2.waitKey(0)
