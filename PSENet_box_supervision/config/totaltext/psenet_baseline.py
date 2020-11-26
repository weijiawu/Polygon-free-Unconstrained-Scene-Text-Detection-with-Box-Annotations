# -*- coding: utf-8 -*-

# data config
exp_name = "totaltext/psenet_baseline"

train_data_dir = '/data/glusterfs_cv_04/11121171/data/Total/Images/Train/'
train_gt_dir = '/data/glusterfs_cv_04/11121171/data/Total/gt/Train/'
test_data_dir = '/data/glusterfs_cv_04/11121171/data/Total/Images/Test/'
test_gt_dir = '/data/glusterfs_cv_04/11121171/data/Total/gt/Test/'

workspace_dir = '/data/glusterfs_cv_04/11121171/CVPR_Text/SemiText/PSENet/workdirs/'
workspace = "/data/glusterfs_cv_04/11121171/CVPR_Text/SemiText/PSENet/workdirs/eval"
data_shape = 640

# train config
gpu_id = '0,1'
workers = 10
start_epoch = 0
epochs = 100
train_batch_size = 20

lr = 1e-4
lr_gamma = 0.1
lr_decay_step = [60,80]

display_input_images = False
display_output_images = False
visualization = True
display_interval = 10
show_images_interval = 50
is_pseudo = False
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
