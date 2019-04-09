'''
@author: Aeolus
@url: x-fei.me
@time: 2019-04-08 21:27
'''
import argparse
import os
import sys
import time
import functools
from os.path import join, exists
from datetime import datetime
from src import config
import traceback
import logging
import logging.handlers


def gen_parser():
    parser = argparse.ArgumentParser(prog=config.PROGRAM,
                                     description=config.DESCRIPTION)

    parser.add_argument('cmd', choices=config.cmd_list, help='what to do')
    parser.add_argument('--desc', type=str, default='', help='description')
    parser.add_argument('--action', type=str, default='base', help='action')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='ImageNet100',
                        help='dataset')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model', type=str, default='res18', help='model')

    # source setting
    parser.add_argument('--cuda', type=str, default='0', help='gpu(s)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers for dataloader')

    # basic setting
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_epoch', type=int, default=35,
                        help='learning rate decay epoch')
    parser.add_argument('--epoch', type=int, default=200, help='epoch')
    parser.add_argument('--pre_train', action='store_true', default=False,
                        help='pre train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch for all')

    # result
    parser.add_argument('--no_eval', action='store_true', default=False,
                        help='no need to eval')

    return parser.parse_args()


def check_mkdir(dir_name):
    if not exists(dir_name):
        os.makedirs(dir_name)