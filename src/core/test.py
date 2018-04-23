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
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def analysed(path):
    img = cv2.imread(path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255))
    rects = []
    for contour in contours:
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        rects.append((x1, y1, w1, h1))
    return rects, img


def find(x_arr, h_max):
    rs = []
    cur = None
    for idx in range(len(x_arr)):
        begin = x_arr[idx]
        vir_end = begin + h_max
        for i in range(len(x_arr)):
            if i == len(x_arr) - 1:
                break
            if x_arr[i] < vir_end < x_arr[i + 1]:
                end = (x_arr[i] + x_arr[i + 1]) // 2
                if cur and begin < cur:
                    break
                if x_arr[i + 1] - x_arr[i] < h_max:
                    rs.append((begin, end))
                cur = end
                break
    rs.append((x_arr[-1], x_arr[-1] + h_max))
    return rs


def show_img(img, rs, height):
    fig, axes = plt.subplots(1, len(rs), figsize=(height, height))
    idx = 0
    for ax in axes.ravel():
        ax.imshow(img[0:height, rs[idx][0]:rs[idx][1]])
        idx += 1
    fig.tight_layout()
    plt.show()


def main():
    rects, img = analysed('./tmp/240.png')
    height, width, _ = img.shape
    idx = 0
    for rect in rects:
        tmp_img = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)
        # cv2.imwrite('./tmp/rs/{}.png'.format(idx), tmp_img)
        plt.scatter(rect[0], rect[1], c='b', marker='*')
        idx += 1
    plt.show()
    cv2.imshow('Image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
