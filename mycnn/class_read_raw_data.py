# -*- coding: utf-8 -*-
# !/usr/bin/python
################################### PART0 DESCRIPTION #################################
# Filename: class_read_raw_data.py
# Description:
#


# Author: Shuai Yuan
# E-mail: ysh329@sina.com
# Create: 2016-02-18 11:24:12
# Last:
__author__ = 'yuens'

################################### PART1 IMPORT ######################################
import logging
import time
import numpy as np
import struct

import decorator_of_function

################################### PART2 CLASS && FUNCTION ###########################
class ReadRawData(object):

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
        logging.info("START CLASS {class_name}.".format(class_name = ReadRawData.__name__))



    @Decorator.log_of_function
    def __del__(self):
        logging.info("END CLASS {class_name}.".format(class_name = ReadRawData.__name__))
        self.end = time.clock()
        logging.info("The class {class_name} run time is : {delta_time} seconds".format(class_name = ReadRawData.__name__, delta_time = self.end))



    @Decorator.log_of_function
    def load_image_data_set(self, img_data_dir):
        logging.info("Load image data set from {0}.".format(img_data_dir))

        with open(img_data_dir, "rb") as binary_file_handle:
            image_data_buffer = binary_file_handle.read()

        # '>IIII'是说使用大端法读取4个unsigned int32
        # unpack_from(...)
        # Unpack the buffer, containing packed C structure data, according to
        # fmt, starting at offset. Requires len(buffer[offset:]) >= calcsize(fmt).
        head = struct.unpack_from('>IIII' , image_data_buffer ,0)
        logging.info("head:{0}".format(head))

        magic_num = struct.calcsize('>IIII')
        img_num = head[1]
        img_width = head[2]
        img_height = head[3]
        logging.info("magic_num:{0}".format(magic_num))
        logging.info("img_num:{0}".format(img_num))
        logging.info("img_width:{0}".format(img_width))
        logging.info("img_height:{0}".format(img_height))

        #[60000]*28*28
        all_img_bit = img_num * img_width * img_height
        all_img_bit_string = '>' + str(all_img_bit) + 'B' #like '>47040000B'
        logging.info("all_img_bit_string:{0}".format(all_img_bit_string))

        all_image_2d_ndarray = struct.unpack_from(all_img_bit_string, image_data_buffer, magic_num)
        all_image_2d_ndarray = np.reshape(all_image_2d_ndarray, [img_num, img_width, img_height])
        return all_image_2d_ndarray



    @Decorator.log_of_function
    def load_label_data_set(self, label_data_dir):
        logging.info("load label set from {0}.".format(label_data_dir))

        with open(label_data_dir, "rb") as binary_file_handle:
            label_data_buffer = binary_file_handle.read()

        head = struct.unpack_from('>II' , label_data_buffer ,0)
        logging.info("head:{0}".format(head))

        label_num = head[1]
        logging.info("img_num:{0}".format(label_num))

        offset = struct.calcsize('>II')
        logging.info("offset:{0}".format(offset))

        img_num_string='>'+str(label_num)+"B"
        logging.info("img_num_string:{0}".format(img_num_string))

        all_label_2d_ndarray = struct.unpack_from(img_num_string, label_data_buffer, offset)
        all_label_2d_ndarray = np.reshape(all_label_2d_ndarray, [label_num, 1])
        logging.info("len(all_label_2d_ndarray):{0}".format(len(all_label_2d_ndarray)))
        logging.info("type(all_label_2d_ndarray):{0}".format(type(all_label_2d_ndarray)))

        logging.info("all_label_2d_ndarray[0]:{0}".format(all_label_2d_ndarray[0]))
        logging.info("type(all_label_2d_ndarray[0]):{0}".format(type(all_label_2d_ndarray[0])))

        logging.info("all_label_2d_ndarray[0][0]:{0}".format(all_label_2d_ndarray[0][0]))
        logging.info("type(all_label_2d_ndarray[0][0]):{0}".format(type(all_label_2d_ndarray[0][0])))

        #print labels
        logging.info("Load label finished.")
        return all_label_2d_ndarray



################################### PART3 CLASS TEST ##################################
#'''
# Initialization
train_sample_data_dir = "..//data//input//train-images-idx3-ubyte"
train_label_data_dir = "..//data//input//train-labels-idx1-ubyte"



DataReader = ReadRawData()
image_2d_ndarray = DataReader.load_image_data_set(img_data_dir = train_sample_data_dir)
image_label = DataReader.load_label_data_set(label_data_dir = train_label_data_dir)
#'''