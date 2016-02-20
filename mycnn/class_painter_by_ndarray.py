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
    def paint_one_img(self, img_ndarray, dpi, img_shape_tuple = (28, 28)):
        plt.figure(figsize = img_shape_tuple, dpi = dpi)
        # show image
        plt.imshow(img_ndarray, cmap = 'gray', interpolation = 'nearest')
        # close axis of image
        plt.axis('off')
        plt.show()



    @Decorator.log_of_function
    def paint_img_sequence(self, img_ndarray_list, img_name_list, dpi, max_img_num_in_plot = 9, img_shape_tuple = (28, 28)):
        # subplot_format: such as 211
        if (len(img_ndarray_list) == len(img_name_list)) and (len(img_name_list) <= max_img_num_in_plot):
            pass
            #logging.info("The length of variable 'img_ndarray_list' equals to length of variable 'img_name_list'.")
        elif len(img_name_list) > max_img_num_in_plot:
            logging.error("The images number(length of variable 'img_ndarray_list') bigger than variable 'max_img_num_in_plot'.")
            return -1
        else:
            logging.error("The length of variable 'img_ndarray_list' differs from length of variable 'img_name_list'.")
            return -1

        plt.figure(figsize = img_shape_tuple, dpi = dpi)

        # get subplot format according to subplot number
        ############## sub-function ##############
        def get_subplot_format(img_name_list):
            # get all prime numbers of number len(img_name_list).
            prime_num_list = map(lambda num: num if len(img_name_list)%num==0 else -1,\
                               xrange(1, len(img_name_list)+1)\
                               )
            prime_num_list = filter(lambda sub_num: sub_num > 0, prime_num_list)

            #logging.info("prime_num_list:{0}".format(prime_num_list))

            # initial a tuple list,
            # qualified with 'prime_num1*prime_num2 == len(img_name_list)'
            prime_tuple_list = []

            for prime_num1_idx in xrange(len(prime_num_list)):
                prime_num1 = prime_num_list[prime_num1_idx]
                #logging.info("prime_num1:{0}".format(prime_num1))

                prime_num2_list = prime_num_list[prime_num1_idx+1:]
                #logging.info("prime_num2_list:{0}".format(prime_num2_list))

                for prime_num2_idx in xrange(len(prime_num2_list)):# except last redundant index
                    prime_num2 = prime_num2_list[prime_num2_idx]
                    #logging.info("prime_num2:{0}".format(prime_num2))
                    #logging.info("(len(img_name_list) - prime_num1*prime_num2):{0}".format(len(img_name_list) - prime_num1*prime_num2))

                    if (len(img_name_list) - prime_num1*prime_num2) == 0:
                        prime_tuple_list.append((prime_num1, prime_num2))

            #logging.info("len(prime_tuple_list):{0}".format(len(prime_tuple_list)))
            #logging.info("prime_tuple_list:{0}".format(prime_tuple_list))

            # Choose the best subplot format from prime_tuple_list
            # by 2 rules below:
            # minus first
            # second order by minus result
            prime_num_triple_list = map(lambda (prime_num1, prime_num2):\
                                            (prime_num1, prime_num2, prime_num2-prime_num1),\
                                        prime_tuple_list)
            prime_num_triple_list.sort(key = lambda prime_num_tripe: prime_num_tripe[2],\
                                       reverse = False)
            #logging.info("prime_num_triple_list:{0}".format(prime_num_triple_list))
            proper_prime_num_tuple = prime_num_triple_list[0][:2]

            # Adjust the subplot number of width and height
            # first, adjust the type
            proper_prime_num_list = list(proper_prime_num_tuple)
            if proper_prime_num_list[1] < proper_prime_num_list[0]: # column number < row number
                proper_prime_num_list[0], proper_prime_num_list[1] = proper_prime_num_list[1], proper_prime_num_list[0]

            #logging.info("proper_prime_num_list:{0}".format(proper_prime_num_list))
            return proper_prime_num_list
        ############## sub-function ##############

        # get subplot format number
        # according to function "get_subplot_format"
        proper_prime_num_list = get_subplot_format(img_name_list = img_name_list)
        first_subplot_num = proper_prime_num_list[0]
        second_subplot_num = proper_prime_num_list[1]
        subplot_figure = plt.figure()

        # calculate all images'subplot
        """
        subplot_plt_list = []
        for img_idx in xrange(len(img_ndarray_list)):
            third_subplot_num = img_idx + 1
            subplot_plt_list.append(subplot_figure.\
                                    add_subplot(first_subplot_num, second_subplot_num, third_subplot_num)\
                                    )
        """
        subplot_plt_list = map(lambda img_idx:\
                                   subplot_figure.add_subplot(first_subplot_num,\
                                                              second_subplot_num,\
                                                              (img_idx+1),\
                                                              ),\
                               xrange(len(img_ndarray_list)),\
                               )

        # imshow all images'subplot
        """
        for img_idx in xrange(len(img_ndarray_list)):
            img_subplot_plt = subplot_plt_list[img_idx]
            img_title = img_name_list[img_idx]
            img_ndarray = img_ndarray_list[img_idx]

            img_subplot_plt.imshow(img_ndarray, cmap = 'gray', interpolation = 'nearest')
            img_subplot_plt.set_title(img_title)
        """
        map(lambda img_subplot_plt, img_ndarray, img_name:\
                img_subplot_plt.imshow(img_ndarray, cmap = 'gray', interpolation = 'nearest'),\
            subplot_plt_list,\
            img_ndarray_list,\
            img_name_list\
            )

        # set titles
        map(lambda img_subplot_plt, img_name:\
                img_subplot_plt.set_title(img_name),\
            subplot_plt_list,\
            img_name_list)

        plt.show()
        #subplot_plt_list = map()



    @Decorator.log_of_function
    def save_one_img(self, img_ndarray, img_save_dir, img_filename, dpi, img_shape_tuple = (28, 28)):
        plt.figure(figsize = img_shape_tuple)
        # show image
        plt.imshow(img_ndarray, cmap = 'gray', interpolation = 'nearest')
        # close axis of image
        plt.axis('off')
        plt.savefig(join(img_save_dir, img_filename), dpi = dpi)



    @Decorator.log_of_function
    def save_img_sequence(self, img_2d_ndarray, img_save_dir, dpi = 100, img_shape_tuple = (28, 28)):
        img_num = len(img_2d_ndarray)
        for img_idx in xrange(img_num):
            cur_img_ndarray = img_2d_ndarray[img_idx]
            cur_img_filename = str(img_idx) + ".jpg"
            self.save_one_img(img_ndarray = cur_img_ndarray,\
                              img_save_dir = img_save_dir,\
                              img_filename = cur_img_filename,\
                              dpi = dpi,\
                              img_shape_tuple = img_shape_tuple)

################################### PART3 CLASS TEST ##################################
#'''
# Initialization
train_sample_data_dir = "..//data//input//train-images-idx3-ubyte"
train_label_data_dir = "..//data//input//train-labels-idx1-ubyte"

img_save_dir = "..//data//output"
test_img_filename = "test.jpg"

# test_img_ndarray
img_ndarray = np.ndarray(shape = (30, 20), dtype = float)
img_ndarray_list = [img_ndarray, img_ndarray, img_ndarray, img_ndarray, img_ndarray, img_ndarray]
img_name_list = ["1", "2", "3", "4", "5", "6"]


Painter = PaintNDarray()
Painter.paint_img_sequence(img_ndarray_list = img_ndarray_list,\
                           img_name_list = img_name_list,\
                           dpi = 1,)
#Painter.save_one_img(img_ndarray = img_ndarray, img_save_dir = img_save_dir, img_filename = test_img_filename)
#Painter.paint_one_img(img_ndarray = img_ndarray)

#'''