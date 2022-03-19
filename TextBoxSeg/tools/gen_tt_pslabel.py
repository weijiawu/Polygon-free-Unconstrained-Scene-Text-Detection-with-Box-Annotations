import os
import sys
import torch
from numpy import *
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg
from IPython import embed
import numpy as np
from tqdm import trange
import cv2

# debug-------------------------
from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image
import math
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
import importlib
import matplotlib.pyplot as plt
import scipy.io as sio
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

debug_flag = True
img_save_path = '/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/trash/imgs'
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

def demo():
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()

    #get img_patch from IC15
    # get img_patch from Total Text
    if os.path.exists('/home/xjc/Dataset/total-text/'):
        total_root_path = '/home/xjc/Dataset/total-text/'
    else:
        total_root_path = '/home/xjc/Dataset/total-text/'
    total_train_data = total_root_path + 'Images/Train/'
    total_train_gt = total_root_path + 'gt/Train/'

    assert os.path.exists(total_train_data) and os.path.exists(total_train_gt)

    '''bbox + pseudo box '''
    total_train_gt_bbox = total_root_path + 'pseudolabel_attention/Train_bbox'
    total_train_gt_pseudobox = total_root_path + 'pseudolabel_attention/Train_pseudo'
    if not os.path.exists(total_train_gt_bbox):
        os.mkdir(total_train_gt_bbox)
    if not os.path.exists(total_train_gt_pseudobox):
        os.mkdir(total_train_gt_pseudobox)
    os.system('rm -rf {}/*mat'.format(total_train_gt_bbox))
    os.system('rm -rf {}/*mat'.format(total_train_gt_pseudobox))

    hs = []
    ws = []
    max_images = 2000
    # 遍历图片
    for i in trange(1, max_images+1):
        img_path = 'img{}.jpg'.format(i)
        img_path = os.path.join(total_train_data, img_path)
        if not os.path.exists(img_path):
            img_path = img_path.replace('jpg', 'JPG') #有一张图是JPG后缀
        gt_path = 'poly_gt_img{}.mat'.format(i)
        gt_path = os.path.join(total_train_gt, gt_path)

        if os.path.exists(gt_path) and os.path.exists(img_path):
            img, boxes, ori_box = parse_img_gt(img_path, gt_path)
            img = np.array(img)
            ori_gt = sio.loadmat(gt_path)
            bbox_gt = sio.loadmat(gt_path)
            cbox_gt = sio.loadmat(gt_path)
            #遍历box
            f_bbox = os.path.join(total_train_gt_bbox, 'poly_gt_img{}.mat'.format(i))
            f_cbox = os.path.join(total_train_gt_pseudobox, 'poly_gt_img{}.mat'.format(i))

            for j, box in enumerate(boxes):
                mask = np.zeros_like(img)[:, :, 0]
                x1, y1, x2, y2, is_ignore = box
                x1 = max(0,x1); y1 = max(0,y1)
                x2 = min(x2, img.shape[1]); y2 = min(y2, img.shape[0])
                patch = img[y1:y2 + 1, x1:x2 + 1]
                patch = Image.fromarray(patch)

                origin_h = patch.height
                origin_w = patch.width
                ws.append(origin_w)
                hs.append(origin_h)

                pred_gt = inference(model, patch, transform)
                mask[y1:y2 + 1, x1:x2 + 1] = pred_gt
                # get bbox rbox
                _, cbox = get_pseudo_label(mask) # curve box
                bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='int32')
                if debug_flag:
                    thickness = 2
                    if is_ignore == 1:
                        cv2.drawContours(img, [cbox], 0, (0, 0, 255), thickness)
                        cv2.drawContours(img, [bbox], 0, (0, 255, 0), thickness)
                        cv2.drawContours(img, [ori_box[j]], 0, (255, 0, 0), thickness)
                    else:
                        cv2.drawContours(img, [cbox], 0, (0, 255, 255), thickness)
                        cv2.drawContours(img, [bbox], 0, (255, 255, 0), thickness)
                        cv2.drawContours(img, [ori_box[j]], 0, (255, 0, 255), thickness)

                if is_ignore == 1:
                    bbox_gt['polygt'][j][1] = bbox[None, :, 0]
                    bbox_gt['polygt'][j][3] = bbox[None, :, 1]
                    cbox_gt['polygt'][j][1] = cbox[None, :, 0]
                    cbox_gt['polygt'][j][3] = cbox[None, :, 1]
                else:
                    bbox_gt['polygt'][j][1] = bbox[None, :, 0]
                    bbox_gt['polygt'][j][3] = bbox[None, :, 1]
                    cbox_gt['polygt'][j][1] = cbox[None, :, 0]
                    cbox_gt['polygt'][j][3] = cbox[None, :, 1]
            assert len(ori_gt['polygt']) == len(bbox_gt['polygt']) == len(cbox_gt['polygt'])
            #save to mat
            sio.savemat(f_bbox, bbox_gt)
            sio.savemat(f_cbox, cbox_gt)

            if debug_flag:
                print('debug vis')
                cv2.imwrite('{}/img{}.png'.format(img_save_path, i), img[:,:,[2,1,0]])
        else:
            print('{} not exist!'.format(img_path))


    print(mean(hs))
    print(mean(ws))

def adjust_box_sort(box):
    start = -1
    _box = list(np.array(box).reshape(-1,2))
    min_x = min(box[0::2])
    min_y = min(box[1::2])
    _box.sort(key=lambda x:(x[0]-min_x)**2+(x[1]-min_y)**2)
    start_point = list(_box[0])
    for i in range(0,8,2):
        x,y = box[i],box[i+1]
        if [x,y] == start_point:
            start = i//2
            break

    new_box = []
    new_box.extend(box[start*2:])
    new_box.extend(box[:start*2])
    return new_box

def get_pseudo_label(mask):

    # smoothing
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # erosion = cv2.erode(img, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask_1 = mask.copy()

    try:
        contours, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        _,contours, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda x:cv2.contourArea(x), reverse=True)
    cnt = contours[0]
    #bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    bbox = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]],dtype='int32')
    c_box = cnt[:,0,:]
    # rotate box
    # rect = cv2.minAreaRect(cnt)
    # rbox = cv2.boxPoints(rect).astype('int32')
    return bbox, c_box

def inference(model, image, transform):
    origin_h, origin_w = image.height, image.width

    if origin_h > origin_w:
        image = image.transpose(Image.ROTATE_90)  # 将图片旋转90度


    resized_img = image.resize(cfg.TRAIN.BASE_SIZE)
    resized_img = transform(resized_img).unsqueeze(0).cuda()
    with torch.no_grad():
        output,_ = model(resized_img)
    pred = Image.fromarray(torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy().astype('int32')).resize(image.size)
    if origin_h > origin_w:
        pred = pred.transpose(Image.ROTATE_270)  # 将图片旋转90度
    pred = np.array(pred)
    return pred

def parse_img_gt(img_path, gt_path):
    img = Image.open(img_path)
    data = sio.loadmat(gt_path)['polygt']
    boxes = []
    ori_boxes = []
    for d in data:
        _, xs, _, ys, text, _ = d
        polygon_np = np.concatenate([xs, ys], 0).transpose(1, 0).astype('int32')
        x, y, w, h = cv2.boundingRect(polygon_np)
        if "#" in text:
            boxes.append([x, y, x + w, y + h, 0])  # 0=ignore
            ori_boxes.append(polygon_np)
        else:
            boxes.append([x, y, x + w, y + h, 1])  # 1=real
            ori_boxes.append(polygon_np)
    return img, boxes, ori_boxes

if __name__ == '__main__':
    demo()