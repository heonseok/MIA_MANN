import numpy as np
import os
import logging

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


# Metric
def calculate_auc(target, pred):
    right_index = (target != -1.)

    right_target = target[right_index]
    right_pred = pred[right_index]

    try:
        return roc_auc_score(right_target, right_pred)
    except:
        return -1


def calculate_acc(target, pred):
    right_index = (target != -1.)

    right_target = target[right_index]
    right_pred = pred[right_index]

    right_pred[right_pred > 0.5] = 1.0
    right_pred[right_pred <= 0.5] = 0.0

    return accuracy_score(right_target, right_pred)


def calculate_auc_acc(target, pred):
    auc = calculate_auc(target, pred)
    acc = calculate_acc(target, pred)

    return auc, acc


def calculate_metric_for_each_q(target, pred, q, num_q):
    count = list()
    result = list()

    for q_idx in range(num_q):

        filtered_idx = (q == q_idx+1)
        count.append(np.sum(filtered_idx))
        filtered_target = target[filtered_idx]
        filtered_pred = pred[filtered_idx]
        sub_result = calculate_auc_acc(filtered_target, filtered_pred)

        result.append(sub_result)

    return count, result


# ########## LOGGER ##########
# def set_logger(name, path, log_file, logging_level, display=True):
#     logger = logging.getLogger(name)
#
#     if logger.hasHandlers():
#         logger.handlers.clear()
#
#     streamFormatter = logging.Formatter('[%(levelname)s] %(message)s')
#     streamHandler = logging.StreamHandler()
#     streamHandler.setFormatter(streamFormatter)
#     streamHandler.setLevel(logging.INFO)
#
#     if not os.path.exists('log/'+path):
#         os.makedirs('log/'+path)
#
#     fileFormatter = logging.Formatter('%(message)s')
#     fileHandler = logging.FileHandler(os.path.join('log', path, log_file))
#     fileHandler.setFormatter(fileFormatter)
#     fileHandler.setLevel(logging.DEBUG)
#
#     if display == True:
#         logger.addHandler(streamHandler)
#
#     logger.addHandler(fileHandler)
#
#     logger.setLevel(eval('logging.{}'.format(logging_level)))
#
#     return logger


# Customized Logger
class CustomLogger:
    def __init__(self, name, path, log_file, logging_level, display_flag=True, result_flag=True):
        if not os.path.exists('log/'+path):
            os.makedirs('log/'+path)

        self.logger = logging.getLogger(name)
        self.result_flag = result_flag

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(logging.INFO)


        log_formatter = logging.Formatter('%(message)s')
        log_handler = logging.FileHandler(os.path.join('log', path, log_file + '.log'))
        log_handler.setFormatter(log_formatter)
        log_handler.setLevel(logging.DEBUG)

        if display_flag:
            self.logger.addHandler(stream_handler)
        self.logger.addHandler(log_handler)

        self.logger.setLevel(eval('logging.{}'.format(logging_level)))

        self.f = open(os.path.join('log', path, log_file + '.result'), 'w')

    def info(self, msg):
        self.logger.info(msg)

        if self.result_flag:
            self.f.write(msg + '\n')

    def debug(self, msg):
        self.logger.info(msg)


# utility
def float2str_list(input_list):
    return ','.join('{:.4f}'.format(item) for item in input_list)


def int2str_list(input_list):
    return ','.join('{}'.format(int(item)) for item in input_list)


def calc_seq_trend(seq):
    zeros = np.zeros(shape=(seq.shape[0], 1))
    seq_right = np.hstack((seq, zeros))
    seq_left = np.hstack((zeros, seq))

    seq_diff = (seq_right-seq_left)[:,1:-1]
    increase_count = len(np.where(seq_diff>0)[0])
    decrease_count = len(np.where(seq_diff<0)[0])

    return increase_count, decrease_count


def calc_rmse_with_ones(seq):
    ones = np.ones_like(seq)
    return np.sqrt(np.average(np.square(seq-ones), axis=0))


def calc_diff_with_ones(seq):
    ones = np.ones_like(seq)
    return np.average((ones-seq), axis=0)