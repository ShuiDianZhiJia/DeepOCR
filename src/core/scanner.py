# -*- coding:utf-8 -*-
"""
@File      : scanner.py
@Software  : OCR
@Time      : 2018/4/15 18:11
@Author    : yubb
"""
import cv2
import numpy as np


class Scanner:

    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.cur_h = 0
        self.cur_w = 0

    def split_image(self, path):
        img_matrix = self.prepare(path)
        self.scanning(img_matrix)

    def scanning(self, img_matrix):
        print(self.width, img_matrix)

    def prepare(self, path):
        """
        preprocessor:
            1、Binaryzation
            2、Remove blank edges
        :param path: path of image
        :return: numpy.array
        """
        img = cv2.imread(path, 0)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        self.cur_h, self.cur_w = img.shape
        return img

