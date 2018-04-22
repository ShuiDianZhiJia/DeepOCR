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


def analysed(path):
    img = cv2.imread(path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = thresh.shape
    cv2.drawContours(img, contours, -1, (0, 0, 255))
    count = 0
    x_arr, y_arr = [], []
    for contour in contours:
        x, y, z = contour.shape
        tmp = contour.reshape(1, x * z)[0]
        x_tmp = np.array([tmp[idx] for idx in range(len(tmp)) if idx % 2 == 0])
        y_tmp = np.array([tmp[idx] for idx in range(len(tmp)) if idx % 2 != 0])
        w = np.max(x_tmp) - np.min(x_tmp)
        h = np.max(y_tmp) - np.min(y_tmp)
        x_arr.append(x_tmp[0])
        y_arr.append(y_tmp[0])
        # print("X={}, Y={}, Width={}, Height={}".format(x_tmp[0], y_tmp[0], w, h))
        # im = img[0:height, x_tmp[0]:x_tmp[0] + 18]
        # cv2.imwrite('./tmp/rs/{}.png'.format(count), im)
        # if width > 10 and height > 18:
        #     im = img[y_tmp[0]:y_tmp[0] + height, x_tmp[0]:x_tmp[0] + width]
        #     # cv2.rectangle(img, (x_tmp[0], y_tmp[0]), (x_tmp[0] + width, y_tmp[0] + height), (0, 0, 255), 2)
        #     cv2.imwrite('./tmp/rs/{}.png'.format(count), im)
        count += 1
    # plt.bar(x_arr, y_arr, 1, color='r')
    # plt.show()
    x_arr, y_arr = list(set(x_arr)), list(set(y_arr))
    x_arr.sort()
    y_arr.sort()
    cv2.imshow('img', img)
    cv2.waitKey(0)
    return x_arr, y_arr, img


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
    x_arr, y_arr, img = analysed('./tmp/240.png')
    height, width, _ = img.shape
    print("X:", x_arr)
    print("MAX_H:", y_arr[-1])
    rs = []
    for idx in range(len(x_arr)):
        if idx != len(x_arr) - 1:
            rs.append(x_arr[idx+1] - x_arr[idx])

    print(rs, sum(rs) // len(rs))
    # for x in x_arr:
    #     plt.scatter(x, 0, c='b', marker='|')
    # show_img(img, find(x_arr, y_arr[-1]), height)
    # cv2.imwrite('./tmp/rs/{}.png'.format(4), img[0:height, 136:155])


if __name__ == '__main__':
    main()
