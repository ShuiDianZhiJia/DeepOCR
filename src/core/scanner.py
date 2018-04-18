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

    """
    Vertical edge:     Horizontal edge:
        [[-1, 0, 1],      [[1, 1, 1],
        [-1, 0, 1],       [0, 0, 0],
        [-1, 0, 1]]       [-1, -1, -1]]
    """

    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.cur_h, self.cur_w = 0, 0
        self.ver_edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        self.hor_edge = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

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


def test():
    img = cv2.imread('./timg.jpg', 0)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ver_edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    hor_edge = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    height, width = img.shape
    x_axis, y_axis, x_span, y_span, num = 0, 0, 3, 3, 0
    h_rs, h_row_rs, v_rs, v_row_rs = None, None, None, None
    for h_idx in range(height - 2):
        for w_idx in range(width - 2):
            tmp_arr = img[h_idx:h_idx + y_span, w_idx:w_idx + x_span]
            v_tmp = np.sum(ver_edge * tmp_arr)
            h_tmp = np.sum(hor_edge * tmp_arr)
            if h_row_rs is None:
                h_row_rs = h_tmp
            else:
                h_row_rs = np.hstack((h_row_rs, h_tmp))
            if v_row_rs is None:
                v_row_rs = v_tmp
            else:
                v_row_rs = np.hstack((v_row_rs, v_tmp))

        if h_rs is None:
            h_rs = h_row_rs
        else:
            h_rs = np.vstack((h_rs, h_row_rs))
        if v_rs is None:
            v_rs = v_row_rs
        else:
            v_rs = np.vstack((v_rs, v_row_rs))

        v_row_rs, h_row_rs = None, None
    cv2.imwrite('./tmp/h.png', h_rs)
    # h, w = h_rs.shape
    # sum, pre_sum = 0, 0
    # for h_ in range(h):
    #     if sum == 0 and np.sum(h_rs[h_, :]) != 0:
    #         cv2.imwrite('./tmp/{}.png'.format(h_), h_rs[0:h_, :])
    #     sum = np.sum(h_rs[h_, :])
    cv2.imwrite('./tmp/v.png', v_rs)
    cv2.imwrite('./tmp/all.png', h_rs * v_rs)
    return v_rs, h_rs

print(test())