import os
import sys
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import mmcv
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
img_save_path = '/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/trash/msra_debug/'
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

    dataset_name = "HUST"
    #get img_patch from msra
    msra_train = os.path.abspath('/home/xjc/Dataset/MSRA-TD500/MSRA-TD500/')
    hust_train = os.path.abspath("/home/xjc/Dataset/HUST-TR400/HUST-TR400/")
    if dataset_name == "msra":
        data_dirs = [os.path.join(msra_train, 'train')]
        gt_dirs = [os.path.join(msra_train, 'train')]
        root_pseudo_path = "/home/xjc/Dataset/MSRA-TD500/pseudo_label/"
    else:
        data_dirs = [hust_train]
        gt_dirs = [hust_train]
        root_pseudo_path = "/home/xjc/Dataset/HUST-TR400/pseudo_label/"
    img_paths = []
    gt_paths = []

    for data_dir, gt_dir in zip(data_dirs, gt_dirs):
        img_names = [img_name for img_name in mmcv.utils.scandir(data_dir) if img_name.endswith('.JPG')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir) if img_name.endswith('.jpg')])

        img_paths_ = []
        gt_paths_ = []
        for idx, img_name in enumerate(img_names):
            img_path = os.path.join(data_dir, img_name)
            img_paths_.append(img_path)

            gt_name = img_name.split('.')[0] + '.gt'
            gt_path = os.path.join(gt_dir, gt_name)
            gt_paths_.append(gt_path)

        img_paths.extend(img_paths_)
        gt_paths.extend(gt_paths_)

    '''bbox + pseudo box '''

    msra_train_gt_bbox = root_pseudo_path + 'gt_bbox'
    msra_train_gt_pseudobox = root_pseudo_path + 'gt_pseudo'
    if not os.path.exists(msra_train_gt_bbox):
        os.mkdir(msra_train_gt_bbox)
    if not os.path.exists(msra_train_gt_pseudobox):
        os.mkdir(msra_train_gt_pseudobox)

    #遍历图片
    for idx, img_path_one in enumerate(img_paths):
        img_path = img_path_one
        gt_path = gt_paths[idx]

        if os.path.exists(gt_path) and os.path.exists(img_path):
            boxes, ori_box = get_ann(gt_path)
            img = cv2.imread(img_path)
            img = np.array(img)


            #遍历box
            f_bbox = open(os.path.join(msra_train_gt_bbox,img_path.split("/")[-1].split(".")[0]+".txt"),'w')
            f_rbox = open(os.path.join(msra_train_gt_pseudobox, img_path.split("/")[-1].split(".")[0]+".txt"), 'w')
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
                        cv2.drawContours(img, [np.array(ori_box[j], dtype=np.int32)[:8].reshape(4, 2)], 0, (255, 0, 0), 1)

                    else:
                        cv2.drawContours(img, [rbox], 0, (0, 255, 255), 1)
                        cv2.drawContours(img, [bbox], 0, (255, 255, 0), 1)
                        cv2.drawContours(img, [np.array(ori_box[j], dtype=np.int32)[:8].reshape(4, 2)], 0, (255, 0, 255), 1)
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
                cv2.imwrite('{}/img{}.png'.format(img_save_path, idx), img[:,:,[2,1,0]])
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

def get_ann( gt_path):
    lines = mmcv.list_from_file(gt_path)
    polygon_list = []
    words = []
    boxes = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')

        gt = line.split(' ')

        w_ = np.float(gt[4])
        h_ = np.float(gt[5])
        x1 = np.float(gt[2]) + w_ / 2.0
        y1 = np.float(gt[3]) + h_ / 2.0
        theta = np.float(gt[6]) / math.pi * 180

        bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
        # print(np.asarray(bbox.reshape(-1)))
        x, y, w, h = cv2.boundingRect(bbox)
        boxes.append([max(x,0), max(y,0), max(x + w,0), max(y + h,0),1])

        bbox = np.array(np.asarray(bbox.reshape(-1)))
        # bbox = bbox.reshape(-1)

        polygon_list.append(bbox)
        words.append('???')
    return boxes,polygon_list

if __name__ == '__main__':
    demo()