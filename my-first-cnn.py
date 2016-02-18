# -*- coding: utf-8 -*-
# !/usr/bin/python
################################### PART0 DESCRIPTION #################################
# Filename: my-first-cnn.py
# Description:

# Author: Shuai Yuan
# E-mail: ysh329@sina.com
# Create: 2016-02-18 16:12:54
# Last:
__author__ = 'yuens'
################################### PART1 IMPORT ######################################
from mycnn.class_load_parameter_from_config import *
from mycnn.class_read_raw_data import *
from mycnn.class_painter_by_ndarray import *

################################ PART3 MAIN ###########################################
def main():
    # Step1: class_initialization_and_load_parameter
    #######################################
    # Initialization
    config_data_dir = "./config.ini"
    #######################################
    # load parameters
    ParameterLoader = LoadParameter()

    train_sample_date_dir,\
    train_label_data_dir,\
    img_save_dir,\
    test_img_filename = ParameterLoader.load_parameter(config_data_dir = config_data_dir)



    # Step2: class_read_raw_data
    #######################################
    # Initialization
    # train_sample_data_dir = "..//data//input//train-images-idx3-ubyte"
    # train_label_data_dir = "..//data//input//train-labels-idx1-ubyte"
    #######################################
    DataReader = ReadRawData()
    image_2d_ndarray = DataReader.load_image_data_set(img_data_dir = train_sample_date_dir)
    label_2d_ndarray = DataReader.load_label_data_set(label_data_dir = train_label_data_dir)
    img_shape_tuple = DataReader.get_img_size(some_img_ndarray = image_2d_ndarray[0])



    # Step3: class_painter_by_ndarray
    #######################################
    # img_save_dir = ".//data//output"
    # test_img_filename = "test.jpg"
    #######################################
    Painter = PaintNDarray()
    Painter.paint_one_img(img_ndarray = image_2d_ndarray[0])
    """
    Painter.save_one_img(img_ndarray = image_2d_ndarray[0],\
                         img_save_dir = img_save_dir,\
                         img_filename = test_img_filename)
    """

################################ PART4 EXECUTE ##################################
if __name__ == "__main__":
    main()