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
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


def analysed(path):
    img = cv2.imread(path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255))
    rects = []
    for contour in contours:
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        rects.append((x1, y1, w1, h1))
    rects.sort()
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


def show(img, rs, height):
    fig, axes = plt.subplots(1, len(rs), figsize=(height, height))
    idx = 0
    for ax in axes.ravel():
        ax.imshow(img[0:height, rs[idx][0]:rs[idx][1]])
        idx += 1
    fig.tight_layout()
    plt.show()


def show_rects(rects):
    idx = 0
    fig, axis = plt.subplots()
    boxes = []
    for rect in rects:
        if idx > 0:
            box = mpatches.Rectangle(rect[0:2], rect[2], rect[3], ec='none')
            boxes.append(box)
        idx += 1
    pc = PatchCollection(boxes, facecolor='r', alpha=0.3, edgecolor='None')
    axis.add_collection(pc)
    axis.errorbar([x[0] for x in rects], [x[1] for x in rects], fmt='None', ecolor='b')
    plt.tight_layout()
    plt.show()


def what_rests(rects):
    for rect in rects:
        print(rect)


def analyse(path):
    img = cv2.imread(path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(imgray, (3, 3), 0)
    canny = cv2.Canny(imgray, 50, 150)
    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = []
    for contour in contours:
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        if w1 > 3:
            rects.append((x1, y1, w1, h1))
    rects.sort()

    return rects, img


def main():
    rects, img = analyse('./tmp/504.png')
    h, w, _ = img.shape
    idx = 0
    print(x[0] for x in rects)
    for rect in rects:
        tmp_img = img[0:h, rect[0]:rect[0] + rect[2]]
        print(rect)
        cv2.imwrite('./tmp/rs/{}.png'.format(idx), tmp_img)
        cv2.rectangle(img, rect[0:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)
        idx += 1
    cv2.imshow('Image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
