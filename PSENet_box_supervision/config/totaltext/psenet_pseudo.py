# -*- coding: utf-8 -*-

# data config
exp_name = "totaltext/psenet_pseudo"

train_data_dir = '/home/xjc/Dataset/total-text/Images/Train/'
train_gt_dir = '/home/xjc/Dataset/total-text/pseudolabel_attention/Train_pseudo/'
test_data_dir = '/home/xjc/Dataset/total-text/Images/Test/'
test_gt_dir = '/home/xjc/Dataset/total-text/gt/Test/'

workspace_dir = '/home/xjc/Desktop/CVPR_SemiText/SemiText/PSENet/workdirs/'
workspace = "/home/xjc/Desktop/CVPR_SemiText/SemiText/PSENet/workdirs/eval"
data_shape = 640

# train config
gpu_id = '2,3'
workers = 10
start_epoch = 0
epochs = 150
train_batch_size = 8

lr = 1e-4
lr_gamma = 0.1
lr_decay_step = [80,120]

display_input_images = False
display_output_images = False
visualization = False
display_interval = 10
show_images_interval = 50

is_pseudo = True
pretrained = True
restart_training = False
checkpoint = ''
save_interval = 5

# net config
backbone = 'resnet50'
Lambda = 0.7
kernel_num = 7
min_scale = 0.4
OHEM_ratio = 3
scale = 1
min_kernel_area = 10.0
# random seed
seed = 2
binary_th = 1.0


def pprint():
    from pprint import pformat
    tem_d = {}
    for k, v in globals().items():
        if not k.startswith('_') and not callable(v):
            tem_d[k] = v
    return pformat(tem_d)
