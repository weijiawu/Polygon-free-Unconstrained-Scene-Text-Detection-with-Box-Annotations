import os
import sys
import torch

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



debug_flag = False
# img_save_path = '../demo/trash/imgs_icd15'
# if not os.path.exists(img_save_path):
#     os.mkdir(img_save_path)

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
    if os.path.exists('/mnt/lustre/share_data/xieenze/wuweijia/ICDAR2015'):
        ic15_root_path = '/mnt/lustre/share_data/xieenze/wuweijia/ICDAR2015'
    else:
        ic15_root_path = '/mnt/lustre/share_data/xieenze/wuweijia/ICDAR2015'
    ic15_train_data = ic15_root_path + '/ch4_training_images'
    ic15_train_gt = ic15_root_path + '/ch4_training_localization_transcription_gt'
    assert os.path.exists(ic15_train_data) and os.path.exists(ic15_train_gt)

    '''bbox + pseudo box '''
    ic15_train_gt_bbox = ic15_root_path + 'ch4_training_localization_transcription_gt_bbox'
    ic15_train_gt_pseudobox = ic15_root_path + 'ch4_training_localization_transcription_gt_pseudo'
    if not os.path.exists(ic15_train_gt_bbox):
        os.mkdir(ic15_train_gt_bbox)
    if not os.path.exists(ic15_train_gt_pseudobox):
        os.mkdir(ic15_train_gt_pseudobox)
    os.system('rm -rf {}/*txt'.format(ic15_train_gt_bbox))
    os.system('rm -rf {}/*txt'.format(ic15_train_gt_pseudobox))

    num_ic15_imgs = 1000
    #遍历图片
    for i in trange(1, num_ic15_imgs+1):
        img_path = 'img_{}.jpg'.format(i)
        img_path = os.path.join(ic15_train_data, img_path)
        gt_path = 'gt_img_{}.txt'.format(i)
        gt_path = os.path.join(ic15_train_gt, gt_path)

        if os.path.exists(gt_path) and os.path.exists(img_path):
            img, boxes, ori_box = parse_img_gt(img_path, gt_path)
            img = np.array(img)
            #遍历box
            f_bbox = open(os.path.join(ic15_train_gt_bbox,'gt_img_{}.txt'.format(i)),'w')
            f_rbox = open(os.path.join(ic15_train_gt_pseudobox, 'gt_img_{}.txt'.format(i)), 'w')
            seq_bbox, seq_rbox = [],[]

            for j, box in enumerate(boxes):
                mask = np.zeros_like(img)[:, :, 0]
                x1, y1, x2, y2, is_ignore = box
                patch = img[y1:y2 + 1, x1:x2 + 1]
                patch = Image.fromarray(patch)
                pred_gt = inference(model, patch, transform)
                mask[y1:y2 + 1, x1:x2 + 1] = pred_gt
                #get bbox rbox
                _, rbox = get_pseudo_pabel(mask)
                bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='int32')
                if debug_flag:
                    if is_ignore == 1:
                        cv2.drawContours(img, [rbox], 0, (0, 0, 255), 1)
                        cv2.drawContours(img, [bbox], 0, (0, 255, 0), 1)
                        cv2.drawContours(img, [ori_box[j]], 0, (255, 0, 0), 1)
                    else:
                        cv2.drawContours(img, [rbox], 0, (0, 255, 255), 1)
                        cv2.drawContours(img, [bbox], 0, (255, 255, 0), 1)
                        cv2.drawContours(img, [ori_box[j]], 0, (255, 0, 255), 1)
                bbox, rbox = list(bbox.reshape(-1)), list(rbox.reshape(-1))
                #起始点和IC15的需要对齐，平移一下
                # embed()
                bbox = adjust_box_sort(bbox)
                rbox = adjust_box_sort(rbox)

                if is_ignore == 1:
                    seq_bbox.append(",".join([str(int(i)) for i in bbox])+',aaa,\n')
                    seq_rbox.append(",".join([str(int(i)) for i in rbox])+',aaa,\n')
                else:
                    seq_bbox.append(",".join([str(int(i)) for i in bbox]) + ',###,\n')
                    seq_rbox.append(",".join([str(int(i)) for i in rbox]) + ',###,\n')
            f_bbox.writelines(seq_bbox)
            f_rbox.writelines(seq_rbox)
            f_bbox.close()
            f_rbox.close()

            if debug_flag:
                print('debug vis')
                cv2.imwrite('{}/img{}.png'.format(img_save_path, i), img[:,:,[2,1,0]])
        else:
            print(img_path)

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

def get_pseudo_pabel(mask):
    mask_ = mask.copy()
    try:
        contours, hierarchy = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        _,contours, hierarchy = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda x:cv2.contourArea(x), reverse=True)
    cnt = contours[0]
    #bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    bbox = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]],dtype='int32')
    #rotate box
    rect = cv2.minAreaRect(cnt)
    rbox = cv2.boxPoints(rect).astype('int32')
    return bbox, rbox

def inference(model, image, transform):
    resized_img = image.resize(cfg.TRAIN.BASE_SIZE)
    resized_img = transform(resized_img).unsqueeze(0).cuda()
    with torch.no_grad():
        output,_ = model(resized_img)
    pred = Image.fromarray(torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy().astype('int32')).resize(image.size)
    pred = np.array(pred)
    return pred

def parse_img_gt(img_path, gt_path):
    img = Image.open(img_path)
    with open(gt_path,encoding='utf-8', mode='r') as f:
        data=f.readlines()
    boxes = []
    ori_boxes = []
    for d in data:
        d = d.replace('\n','').split(',')
        polygon = d[:8]; text = d[8]
        polygon = [int(i.replace('\ufeff','')) for i in polygon]
        polygon_np = np.array(polygon).reshape([-1, 2])
        x, y, w, h = cv2.boundingRect(polygon_np)
        if "##" in text:
            boxes.append([x, y, x + w, y + h, 0]) #0=ignore
            ori_boxes.append(polygon_np)
        else:
            boxes.append([x, y, x + w, y + h, 1]) #1=real
            ori_boxes.append(polygon_np)
    return img, boxes, ori_boxes

if __name__ == '__main__':
    demo()

