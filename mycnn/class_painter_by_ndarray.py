# -*- coding: utf-8 -*-
# !/usr/bin/python
################################### PART0 DESCRIPTION #################################
# Filename: class_painter_by_ndarray.py
# Description:
#


# Author: Shuai Yuan
# E-mail: ysh329@sina.com
# Create: 2016-02-18 17:05:35
# Last:
__author__ = 'yuens'

################################### PART1 IMPORT ######################################
import logging
import time
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

import decorator_of_function

################################### PART2 CLASS && FUNCTION ###########################
class PaintNDarray(object):

    Decorator = decorator_of_function.CreateDecorator()

    @Decorator.log_of_function
    def __init__(self):
        self.start = time.clock()

        logging.basicConfig(level = logging.INFO,
                            format = '%(asctime)s  %(levelname)5s %(filename)19s[line:%(lineno)3d] %(funcName)s %(message)s',
                            datefmt = '%y-%m-%d %H:%M:%S',
                            filename = './my-first-cnn.log',
                            filemode = 'a')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s  %(levelname)5s %(filename)19s[line:%(lineno)3d] %(funcName)s %(message)s')
        console.setFormatter(formatter)

        logging.getLogger('').addHandler(console)
        logging.info("START CLASS {class_name}.".format(class_name = PaintNDarray.__name__))



    @Decorator.log_of_function
    def __del__(self):
        logging.info("END CLASS {class_name}.".format(class_name = PaintNDarray.__name__))
        self.end = time.clock()
        logging.info("The class {class_name} run time is : {delta_time} seconds".format(class_name = PaintNDarray.__name__, delta_time = self.end))



    @Decorator.log_of_function
    def paint_one_img(self, img_ndarray, img_shape_tuple = (28, 28)):
        plt.figure(figsize = img_shape_tuple)
        # show image
        plt.imshow(img_ndarray, cmap = 'gray', interpolation = 'nearest')
        # close axis of image
        plt.axis('off')
        plt.show()



    @Decorator.log_of_function
    def save_one_img(self, img_ndarray, img_save_dir, img_filename, img_shape_tuple = (28, 28)):
        plt.figure(figsize = img_shape_tuple)
        # show image
        plt.imshow(img_ndarray, cmap = 'gray', interpolation = 'nearest')
        # close axis of image
        plt.axis('off')
        plt.savefig(join(img_save_dir, img_filename))


    @Decorator.log_of_function
    def save_img_sequence(self, img_2d_ndarray, img_save_dir, img_shape_tuple = (28, 28)):
        img_num = len(img_2d_ndarray)
        for img_idx in xrange(img_num):
            cur_img_ndarray = img_2d_ndarray[img_idx]
            cur_img_filename = str(img_idx) + ".jpg"
            self.save_one_img(img_ndarray = cur_img_ndarray,\
                              img_save_dir = img_save_dir,\
                              img_filename = cur_img_filename,\
                              img_shape_tuple = img_shape_tuple)

################################### PART3 CLASS TEST ##################################
'''
# Initialization
train_sample_data_dir = "..//data//input//train-images-idx3-ubyte"
train_label_data_dir = "..//data//input//train-labels-idx1-ubyte"

img_save_dir = "..//data//output"
test_img_filename = "test.jpg"

# test_img_ndarray
img_ndarray = np.ndarray(shape = (30, 20), dtype = float)


Painter = PaintNDarray()
Painter.save_one_img(img_ndarray = img_ndarray, img_save_dir = img_save_dir, img_filename = test_img_filename)
#Painter.paint_one_img(img_ndarray = img_ndarray)

'''