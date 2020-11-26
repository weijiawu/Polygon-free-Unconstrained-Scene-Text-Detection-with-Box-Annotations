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
from utils.utils import write_result_as_txt,debug,load_checkpoint, save_checkpoint, setup_logger
from models import PSENet
from evaluation.script import getresult
from pse import decode_msra as pse_decode
from mmcv import Config
import argparse
from evaluation.msra.eval import get_msra_result
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def scale_image(img, short_size=704):
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


def eval(model, save_path, test_path, device,threshold):
    model.eval()

    save_path = os.path.join(save_path, "submit")
    os.makedirs(save_path, exist_ok=True)
    img_path_root = os.path.join(test_path, "MSRA-TD500", "test")

    # 预测所有测试图片
    img_paths = [os.path.join(img_path_root, x) for x in os.listdir(img_path_root)]
    for img_path in tqdm(img_paths, desc='test models'):
        if not img_path.endswith('.JPG') and not img_path.endswith('.jpg'):
            continue
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')

        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        org_img = img.copy()
        h, w = img.shape[:2]

        img = scale_image(img)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            preds = model(tensor)
            preds, boxes_list = pse_decode(preds[0], config.scale, org_img)

        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    # 开始计算 recall precision f1
    f_score_new = get_msra_result(save_path, img_path_root)
    return f_score_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('--config', type=str, default="./config/msra/msra_baseline.py", help='')
    parser.add_argument('--resume_from', '-r', type=str,
                        default="/home/xjc/Desktop/CVPR_SemiText/SemiText/PSENet_box_supervision/workspace/msra/msra_pseudo/Best_model_0.794250.pth",
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
    for i in range(70,80):
        thre = i*0.01
        print(thre)
        eval(model, config.workspace, config.msra_path, device, thre)
        # eval(model, config, device,thre)
        break