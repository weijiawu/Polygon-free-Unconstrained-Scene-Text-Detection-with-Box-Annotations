# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 17:40
# @Author  : zhoujun


# data config
exp_name = "ICDAR17"
trainroot = '/data/glusterfs_cv_04/11121171/data/ICDAR2017/'
testroot = '/data/glusterfs_cv_04/11121171/data/ICDAR2017/'
workspace_dir = '/data/glusterfs_cv_04/11121171/CVPR_Text/PSENet/workspace'
workspace = ""
gt_name = "icd17_gt.zip"
data_shape = 800

# train config
gpu_id = '0,1'
workers = 10
start_epoch = 0
epochs = 600

train_batch_size = 24

lr = 1e-4
end_lr = 1e-7
lr_gamma = 0.1
lr_decay_step = [100,200]
weight_decay = 5e-4
warm_up_epoch = 6
warm_up_lr = lr * lr_gamma

display_input_images = False
display_output_images = False
display_interval = 10
show_images_interval = 50

pretrained = True
restart_training = False
checkpoint = ''

# net config
backbone = 'resnet50'
Lambda = 0.7
n = 6
m = 0.5
OHEM_ratio = 3
scale = 1
# random seed
seed = 2


def print():
    from pprint import pformat
    tem_d = {}
    for k, v in globals().items():
        if not k.startswith('_') and not callable(v):
            tem_d[k] = v
    return pformat(tem_d)
