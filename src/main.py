# -*- coding:utf-8 -*-
"""
@File      : main.py
@Software  : OCR
@Time      : 2018/3/30 17:23
@Author    : yubb
"""
import tensorflow as tf
import pytesseract
from PIL import Image


def main(argv=None):
    text = pytesseract.image_to_string(Image.open('./2.png'), lang='eng')
    print(text)


if __name__ == '__main__':
    main()