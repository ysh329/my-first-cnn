# -*- coding: utf-8 -*-
# !/usr/bin/python
################################### PART0 DESCRIPTION #################################
# Filename: class_load_parameter_from_config.py
# Description:
#


# Author: Shuai Yuan
# E-mail: ysh329@sina.com
# Create: 2016-02-18 16:04:55
# Last:
__author__ = 'yuens'
################################### PART1 IMPORT ######################################
import logging
import ConfigParser
import time

import decorator_of_function

################################### PART2 CLASS && FUNCTION ###########################
class LoadParameter(object):

    Decorator = decorator_of_function.CreateDecorator()

    @Decorator.log_of_function
    def __init__(self):
        self.start = time.clock()

        logging.basicConfig(level = logging.INFO,
                  format = '%(asctime)s  %(levelname)5s %(filename)19s[line:%(lineno)3d] %(funcName)s %(message)s',
                  datefmt = '%y-%m-%d %H:%M:%S',
                  filename = 'main.log',
                  filemode = 'a')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s  %(levelname)5s %(filename)19s[line:%(lineno)3d] %(funcName)s %(message)s')
        console.setFormatter(formatter)

        logging.getLogger('').addHandler(console)
        logging.info("START CLASS {class_name}.".format(class_name = LoadParameter.__name__))


    @Decorator.log_of_function
    def __del__(self):
        logging.info("Success in quiting MySQL.")
        logging.info("END CLASS {class_name}.".format(class_name = LoadParameter.__name__))

        self.end = time.clock()
        logging.info("The class {class_name} run time is : {delta_time} seconds".format(class_name = LoadParameter.__name__, delta_time = self.end - self.start))


    @Decorator.log_of_function
    def load_parameter(self, config_data_dir):
        conf = ConfigParser.ConfigParser()
        conf.read(config_data_dir)

        #[data]
        train_sample_data_dir = conf.get("data", "train_sample_data_dir")
        logging.info("train_sample_data_dir:{0}".format(train_sample_data_dir))

        train_label_data_dir = conf.get("data", "train_label_data_dir")
        logging.info("train_label_data_dir:{0}".format(train_label_data_dir))

        #[plot]
        img_save_dir = conf.get("plot", "img_save_dir")
        logging.info("img_save_dir:{0}".format(img_save_dir))

        test_img_filename = conf.get("plot", "test_img_filename")
        logging.info("test_img_filename:{0}".format(test_img_filename))

        return train_sample_data_dir, train_label_data_dir, img_save_dir, test_img_filename


################################### PART3 CLASS TEST ##################################
'''
# Initialization
config_data_dir = "../config.ini"

# load parameters
ParameterLoader = LoadParameter()
train_sample_date_dir,\
train_label_data_dir = ParameterLoader.load_parameter(config_data_dir = config_data_dir)

'''