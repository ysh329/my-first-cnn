# -*- coding: utf-8 -*-
# !/usr/bin/python
################################### PART0 DESCRIPTION #################################
# Filename: class_create_network.py
# Description:
#


# Author: Shuai Yuan
# E-mail: ysh329@sina.com
# Create: 2016-02-18 21:24:12
# Last:
__author__ = 'yuens'

################################### PART1 IMPORT ######################################
import logging
import time
import numpy as np
import struct

import decorator_of_function

from class_painter_by_ndarray import *

################################### PART2 CLASS && FUNCTION ###########################
class CreateNetwork(object):

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
        logging.info("START CLASS {class_name}.".format(class_name = CreateNetwork.__name__))



    @Decorator.log_of_function
    def __del__(self):
        logging.info("END CLASS {class_name}.".format(class_name = CreateNetwork.__name__))
        self.end = time.clock()
        logging.info("The class {class_name} run time is : {delta_time} seconds".format(class_name = CreateNetwork.__name__, delta_time = self.end))



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
    def sigmoid_activation_function(self, input_matrix):
        return 1.0 / (1.0 + np.exp(-input_matrix))



    @Decorator.log_of_function
    def tanh_activation_function(self, input_matrix):
        sub_operator1 = np.exp(input_matrix)
        sub_operator2 = np.exp(-input_matrix)
        return (sub_operator1 - sub_operator2)/(sub_operator1 + sub_operator2)



    @Decorator.log_of_function
    def relu_activation_function(self, input_matrix):
        return self.leaky_relu_activation_function(input_matrix = input_matrix,\
                                                   leaky_coefficient = 1)


    @Decorator.log_of_function
    def leaky_relu_activation_function(self, input_matrix, leaky_coefficient):
        ################# start #################
        def relu(x, leak_coefficient):
            if leaky_coefficient == 1:
                return 0 if x <= 0 else x
            else:
                return 0 if x <= 0 else float(x)/leaky_coefficient
        #################  end  #################
        vectorized_relu = np.vectorize(relu)
        return vectorized_relu(input_matrix, leak_coefficient = leaky_coefficient)


    @Decorator.log_of_function
    def max_out_activation_function(self, input_matrix):
        pass


    @Decorator.log_of_function
    def softplus_activation_function(self, input_matrix):
        return np.log(1 + input_matrix.dot(np.e))



    @Decorator.log_of_function
    def convolution(self, input_matrix, conv_operator_array = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])):
        conv_operator_shape_tuple = conv_operator_array.shape
        conv_operator_height = conv_operator_shape_tuple[0]
        conv_operator_width = conv_operator_shape_tuple[1]

        input_matrix_shape_tuple = input_matrix.shape
        input_matrix_height = input_matrix_shape_tuple[0]
        input_matrix_width = input_matrix_shape_tuple[1]

        if input_matrix_height <= conv_operator_height or input_matrix_width <= conv_operator_width:
            logging.error("The input matrix({0}) can't execute convolution({1}) progress.".format(input_matrix_shape_tuple, conv_operator_shape_tuple))
            unconv_input_matrix = input_matrix
            return unconv_input_matrix

        # initialization
        new_conv_matrix_height = input_matrix_height - conv_operator_height + 1
        new_conv_matrix_width = input_matrix_width - conv_operator_width + 1

        # sub-function
        def calculate_start_conv_coordinate_in_origin_matrix(new_conv_matrix_height, new_conv_matrix_width):
            start_conv_coordinate_tuple_list = []
            start_height_xrange = xrange(new_conv_matrix_height)
            start_width_xrange = xrange(new_conv_matrix_width)
            for start_height in start_height_xrange:
                cur_conv_coordinate_tuple_list = map(lambda start_width: (start_height, start_width), start_width_xrange)
                start_conv_coordinate_tuple_list.extend(cur_conv_coordinate_tuple_list)
            return start_conv_coordinate_tuple_list

        start_conv_coordinate_tuple_list_in_origin_matrix = calculate_start_conv_coordinate_in_origin_matrix(new_conv_matrix_height = new_conv_matrix_height,\
                                                                                                             new_conv_matrix_width = new_conv_matrix_width)
        logging.info("start_conv_coordinate_tuple_list_in_origin_matrix:{0}".format(start_conv_coordinate_tuple_list_in_origin_matrix))
        logging.info("len(start_conv_coordinate_tuple_list_in_origin_matrix):{0}".format(len(start_conv_coordinate_tuple_list_in_origin_matrix)))

        def calculate_conv_matrix_according_to_start_conv_coordinate(start_conv_coordinate_tuple_list_in_origin_matrix, input_matrix, conv_operator_array):
            input_matrix = np.mat(input_matrix)
            conv_operator_matrix = np.mat(conv_operator_array)
            """
            logging.info("conv_operator_matrix:{0}".format(conv_operator_matrix))
            logging.info("input_matrix:{0}".format(input_matrix))
            """

            conv_operator_height, conv_operator_width = conv_operator_matrix.shape
            """
            logging.info("conv_operator_matrix.height:{0}".format(conv_operator_height))
            logging.info("conv_operator_width:{0}".format(conv_operator_width))
            """

            # Initialize a convolution matrix
            #conv_input_matrix = np.ones((new_conv_matrix_height, new_conv_matrix_width))
            conv_input_list = list()

            for start_coord_tuple_idx in xrange(len(start_conv_coordinate_tuple_list_in_origin_matrix)):
                start_coord_tuple = start_conv_coordinate_tuple_list_in_origin_matrix[start_coord_tuple_idx]
                #logging.info("start_coord_tuple:{0}".format(start_coord_tuple))

                start_coord_height = start_coord_tuple[0]
                end_coord_height = start_coord_height+conv_operator_height #+1(dont need plus 1, because operator starts from 1)
                start_coord_width = start_coord_tuple[1]
                end_coord_width = start_coord_width+conv_operator_width #+1(dont need plus 1, because operator starts from 1)
                """
                conv_value = input_matrix[start_coord_height:end_coord_height, start_coord_width:end_coord_width]
                logging.info("conv_value:{0}".format(conv_value))
                break
                """
                conv_value = input_matrix[start_coord_height:end_coord_height, start_coord_width:end_coord_width]\
                    .dot(conv_operator_matrix)\
                    .sum()

                conv_input_list.append(conv_value)

            conv_input_matrix = np\
                .mat(conv_input_list)\
                .reshape(new_conv_matrix_height,\
                         new_conv_matrix_width)

            return conv_input_matrix


        conv_input_matrix = calculate_conv_matrix_according_to_start_conv_coordinate(\
            start_conv_coordinate_tuple_list_in_origin_matrix = start_conv_coordinate_tuple_list_in_origin_matrix,\
            input_matrix = input_matrix,\
            conv_operator_array = conv_operator_array)

        return conv_input_matrix




    @Decorator.log_of_function
    def general_pooling(self, input_matrix, pooling_mode = "max", pooling_operator_shape_tuple = (2, 2)):
        # First, Compare the scale of variable "input_matrix" with pooling operator("pooling_operator_shape_tuple")
        input_matrix = np.mat(input_matrix)
        if input_matrix.shape[0] <= pooling_operator_shape_tuple[0] or input_matrix.shape[1] <= pooling_operator_shape_tuple[1]:
            logging.error("The scale of pooling operator is bigger than input matrix.")
            return input_matrix

        #logging.info("input_matrix.shape:{0}".format(input_matrix.shape))

        # Second, Find all start pooling coordinate in origin matrix("input_matrix")
        new_pooled_matrix_height = int(input_matrix.shape[0]/pooling_operator_shape_tuple[0])
        new_pooled_matrix_width = int(input_matrix.shape[1]/pooling_operator_shape_tuple[1])

        #logging.info("new_pooled_matrix_height:{0}".format(new_pooled_matrix_height))
        #logging.info("new_pooled_matrix_width:{0}".format(new_pooled_matrix_width))


        ################# start #################
        def calculate_start_pooling_coordinate_in_origin_matrix(input_matrix_shape_tuple, pooling_operator_shape_tuple):
            start_pooling_coordinate_tuple_list = []
            start_height_xrange = xrange(0, input_matrix_shape_tuple[0], pooling_operator_shape_tuple[0])
            start_width_xrange = xrange(0, input_matrix_shape_tuple[1], pooling_operator_shape_tuple[1])
            for start_height in start_height_xrange:
                cur_pooling_coordinate_tuple_list = map(lambda start_width: (start_height, start_width), start_width_xrange)
                start_pooling_coordinate_tuple_list.extend(cur_pooling_coordinate_tuple_list)
            return start_pooling_coordinate_tuple_list
        #################  end  #################


        start_pooling_coordinate_tuple_list = calculate_start_pooling_coordinate_in_origin_matrix(input_matrix_shape_tuple = input_matrix.shape,\
                                                                                                  pooling_operator_shape_tuple = pooling_operator_shape_tuple)
        #logging.info("len(start_pooling_coordinate_tuple_list):{0}".format(len(start_pooling_coordinate_tuple_list)))
        #logging.info("start_pooling_coordinate_tuple_list:{0}".format(start_pooling_coordinate_tuple_list))


        ################# start #################
        def calculate_pooled_matrix_according_to_start_pooling_coordinate(start_pooling_coordinate_tuple_list_in_origin_matrix, input_matrix, pooling_mode, pooling_operator_shape_tuple, new_pooled_matrix_shape_tuple):
            input_matrix = np.mat(input_matrix)
            #logging.info("input_matrix:{0}".format(input_matrix))

            pooling_operator_height, pooling_operator_width = pooling_operator_shape_tuple[0], pooling_operator_shape_tuple[1]
            #logging.info("pooling_operator_matrix.height:{0}".format(pooling_operator_height))
            #logging.info("pooling_operator_width:{0}".format(pooling_operator_width))

            # Initialize a convolution matrix
            pooled_input_list = list()

            for start_coord_tuple_idx in xrange(len(start_pooling_coordinate_tuple_list_in_origin_matrix)):
                start_coord_tuple = start_pooling_coordinate_tuple_list_in_origin_matrix[start_coord_tuple_idx]
                #logging.info("start_coord_tuple:{0}".format(start_coord_tuple))

                start_coord_height = start_coord_tuple[0]
                end_coord_height = start_coord_height+pooling_operator_height #+1(dont need plus 1, because operator starts from 1)
                start_coord_width = start_coord_tuple[1]
                end_coord_width = start_coord_width+pooling_operator_width #+1(dont need plus 1, because operator starts from 1)

                if pooling_mode == "max":
                    pooled_value = input_matrix[start_coord_height:end_coord_height, start_coord_width:end_coord_width]\
                        .max()
                elif pooling_mode == "mean":
                    pooled_value = input_matrix[start_coord_height:end_coord_height, start_coord_width:end_coord_width]\
                        .mean()

                pooled_input_list.append(pooled_value)

            new_pooled_matrix_height, new_pooled_matrix_width = new_pooled_matrix_shape_tuple[0], new_pooled_matrix_shape_tuple[1]
            pooled_input_matrix = np\
                .mat(pooled_input_list)\
                .reshape(new_pooled_matrix_height,\
                         new_pooled_matrix_width)

            return pooled_input_matrix
        #################  end  #################

        pooled_input_matrix = calculate_pooled_matrix_according_to_start_pooling_coordinate(start_pooling_coordinate_tuple_list_in_origin_matrix = start_pooling_coordinate_tuple_list,\
                                                                                            input_matrix = input_matrix,\
                                                                                            pooling_operator_shape_tuple = pooling_operator_shape_tuple,\
                                                                                            pooling_mode = pooling_mode,\
                                                                                            new_pooled_matrix_shape_tuple = (new_pooled_matrix_height, new_pooled_matrix_width))
        return pooled_input_matrix



    @Decorator.log_of_function
    def overlapping_pooling(self, input_matrix):
        pass


    @Decorator.log_of_function
    def spatial_pyramid_pooling(self, input_matrix):
        pass







################################### PART3 CLASS TEST ##################################
#'''
# Initialization
train_sample_data_dir = "..//data//input//train-images-idx3-ubyte"
train_label_data_dir = "..//data//input//train-labels-idx1-ubyte"
img_save_dir = "../data/output"
img_filename = "raw-relu.jpg"

# Get data and one image matrix
Net = CreateNetwork()
all_image_2d_ndarray = Net.load_image_data_set(img_data_dir = train_sample_data_dir)
input_matrix = all_image_2d_ndarray[0]
#logging.info(input_matrix)


# convolution
#input_matrix = Net.tanh_function(input_matrix = input_matrix)
#input_matrix = Net.softplus_activation_function(input_matrix = input_matrix)
#input_matrix = Net.convolution(input_matrix = input_matrix)
input_matrix = Net.relu_activation_function(input_matrix = input_matrix)


'''
# pooling
input_matrix = Net.general_pooling(input_matrix = input_matrix,\
                                   pooling_mode = "max",\
                                   pooling_operator_shape_tuple = (2, 2))
'''

#"""
Painter = PaintNDarray()
Painter.paint_one_img(img_ndarray = input_matrix,\
                      dpi = 1)

Painter.save_one_img(img_ndarray = input_matrix,\
                     img_save_dir = img_save_dir,\
                     img_filename = img_filename,\
                     dpi = 100,\
                     img_shape_tuple = input_matrix.shape)
#"""


#'''