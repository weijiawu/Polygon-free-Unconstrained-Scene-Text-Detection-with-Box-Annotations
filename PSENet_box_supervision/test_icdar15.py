import cv2
import os
import shutil
import glob
import time
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torch.utils.data as Data
from torchvision import transforms
import torchvision.utils as vutils
from utils.utils import write_result_as_txt,debug,load_checkpoint, save_checkpoint, setup_logger
from models import PSENet
from evaluation.script import getresult
from pse import decode as pse_decode
from pse import decode_icdar17 as pse_decode_17
from mmcv import Config
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def scale_image(img, short_size=800):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    max_scale = 3200.0 / max(h, w)
    scale = min(scale, max_scale)

    # img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    h = (int)(h * scale + 0.5)
    w = (int)(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def eval(model, config, device, thre, long_size=1280):
    model.eval()
    img_path = os.path.join(config.testroot, 'test_img')
    save_path = os.path.join(config.workspace, 'output_eval')

    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    long_size = 2240
    # 预测所有测试图片
    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    for img_path in tqdm(img_paths, desc='test models'):
        img_name = os.path.basename(img_path).split('.')[0]

        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')

        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        scale = long_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            preds = model(tensor)
            preds, boxes_list = pse_decode(preds[0], config.scale)
            scale = (preds.shape[1] * 1.0 / w, preds.shape[0] * 1.0 / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    # 开始计算 recall precision f1

    methodHmean,methodPrecision,methodRecall = getresult(save_path, config.gt_name)
    print("precision: {} , recall: {},  f1: {}".format(methodPrecision,methodRecall,methodHmean))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('--config', type=str, default="./config/icdar15/icdar15_ST.py", help='')
    parser.add_argument('--resume_from', '-r', type=str,
                        default="/home/xjc/Desktop/CVPR_SemiText/SemiText/PSENet/workdirs/icdar15/psenet_ST/Best_model_0.863204.pth",
                        help='')
    parser.add_argument('--vis', action='store_true', help='')
    args = parser.parse_args()

    config = Config.fromfile(args.config)
    config.workspace = os.path.join(config.workspace_dir, config.exp_name)
    config.checkpoint = args.resume_from
    config.visualization = config.visualization

    if not os.path.exists(config.workspace):
        os.makedirs(config.workspace)
    logger = setup_logger(os.path.join(config.workspace, 'test_log'))
    logger.info(config.print())

    model = PSENet(backbone=config.backbone,
                   pretrained=config.pretrained,
                   result_num=config.kernel_num,
                   scale=config.scale)
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0")
    model = nn.DataParallel(model)
    model = model.to(device)
    state = torch.load(config.checkpoint)
    model.load_state_dict(state['state_dict'])
    logger.info('test epoch {}'.format(state['epoch']))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    start_epoch = load_checkpoint(config.checkpoint, model, logger, device, optimizer)
    for i in range(60,100,2):
        thre = i*0.01
        print(thre)
        eval(model, config, device,thre)
