'''
@author: Aeolus
@url: x-fei.me
@time: 2019-04-08 21:26
'''
from os.path import join, dirname, abspath
from src import tool

# Basic project info
AUTHOR = "Felix"
PROGRAM = "Robust"
DESCRIPTION = "Defense against adversarial attacks. " \
              "If you find any bug, please new issue. "

# Main CMDs. This decides what kind of cmd you will use.
cmd_list = ['temp', 'train', 'test']

log_name = 'Robust'

# add parsers to this procedure
globals().update(vars(tool.gen_parser()))

def init_path_config(main_file):
    # global_variables
    gv = globals()
    project_dir = abspath(join(dirname(main_file), '..'))
    gv['project_dir'] = project_dir
    gv['data_dir'] = data_dir = join(project_dir, 'data')
    gv['log_dir'] = join(data_dir, 'log')
    gv['loss_dir'] = join(data_dir, 'loss')
    gv['model_dir'] = join(data_dir, 'model')
    gv['best_model_dir'] = join(data_dir, 'best_model')
    # tensorboard dir
    gv['tb_dir'] = join(data_dir, 'tb')

    # local
    # gv['ImageNet100_dir'] = '/data/DataSets/MyImagenet'
    gv['CIFAR10_dir'] = '/data/DataSets/cifar10'
    # 15
    # gv['ImageNet100_dir'] = '/home/feifei/datasets/MyImagenet'
    # 16
    gv['ImageNet100_dir'] = '/data0/feifei/datasets/MyImagenet'
