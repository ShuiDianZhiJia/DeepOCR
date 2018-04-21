# -*- coding:utf-8 -*-
"""
@File      : test.py
@Software  : DeepOCR
@Time      : 2018/4/20 8:05
@Author    : yubb
"""
import os
import numpy as np
import cv2


img = cv2.imread('./exam1.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 0, 255))
count = 0
for contour in contours:
    x, y, z = contour.shape
    tmp = contour.reshape(1, x * z)[0]
    x_tmp = np.array([tmp[idx] for idx in range(len(tmp)) if idx % 2 == 0])
    y_tmp = np.array([tmp[idx] for idx in range(len(tmp)) if idx % 2 != 0])
    width = np.max(x_tmp) - np.min(x_tmp)
    height = np.max(y_tmp) - np.min(y_tmp)
    print("X={}, Y={}, Width={}, Height={}".format(x_tmp[0], y_tmp[0], width, height))
    if width > 10 and height > 18:
        im = img[y_tmp[0]:y_tmp[0] + height, x_tmp[0]:x_tmp[0] + width]
        # cv2.rectangle(img, (x_tmp[0], y_tmp[0]), (x_tmp[0] + width, y_tmp[0] + height), (0, 0, 255), 2)
        cv2.imwrite('./tmp/rs/{}.png'.format(count), im)
    count += 1
cv2.imshow('img', img)
cv2.waitKey(0)