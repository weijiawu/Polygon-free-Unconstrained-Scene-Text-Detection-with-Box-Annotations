import os
import sys
import torch
from numpy import *
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
from tqdm import tqdm
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
import json
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
import importlib
import matplotlib.pyplot as plt
import scipy.io as sio

debug_flag = True

def mask_image(image,mask_2d,rgb):
    h, w = mask_2d.shape

    # mask_3d = np.ones((h, w), dtype="uint8") * 255
    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    # mask_3d[mask_2d[:, :] == 1] = 0

    image.astype("uint8")
    # beijing = cv2.bitwise_and(image, image, mask=mask_3d)
    # R = random.randint(100,200)
    # G = random.randint(100,200)
    # B = random.randint(100,200)

    # mask_3d_color[mask_2d[:, :] == 1] = np.random.randint(50, 200, (1, 3), dtype=np.uint8)
    #
    # add_image = cv2.add(image, mask_3d_color)

    mask = (mask_2d!=0).astype(bool)
    # rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
    rgb = np.array(rgb, dtype=np.uint8)

    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5

    return image

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


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

    img_save_path = "./demo/trash/"
    ic19_train_data = '/data/data_weijiawu/ICDAR2019_Art/train_images/train_images/'
    ic19_train_gt = '/data/data_weijiawu/ICDAR2019_Art/train_labels.json'

    with open(ic19_train_gt, 'r', encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    image_list = os.listdir(ic19_train_data)
    # 遍历图片
    for i,img_path in enumerate(tqdm(image_list)):
        # print(img_path)
        img_path = "gt_2831.jpg"
        img_path = os.path.join(ic19_train_data, img_path)
        gt_ = gt[img_path.split("/")[-1].split(".")[0]]


        img, boxes, polygon_list = get_bboxes(img_path, gt_)
        if not os.path.exists(img_path):
            img_path = img_path.replace('jpg', 'JPG') #有一张图是JPG后缀
        img = np.array(img)
        vis_mask = np.zeros_like(img)
        vis_mask_box = np.zeros_like(img)

        vis_mask_hot = np.zeros_like(img[:,:,0]).astype('float')
        for j, box in enumerate(boxes):
            if j==0:
                rgb = [[187,23,207]]
            elif j ==1:
                rgb = [[255,255,10]]
            else:
                rgb = [[97,186,53]]
            if len(img.shape)==2:
                img = np.expand_dims(img,2)
                img = np.concatenate((img,img,img),-1)
            mask = np.zeros_like(img)[:, :, 0]
            mask_box = np.zeros_like(img)[:, :, 0]
            x1, y1, x2, y2, is_ignore = box
            x1 = max(0,x1); y1 = max(0,y1)
            x2 = min(x2, img.shape[1]); y2 = min(y2, img.shape[0])

            patch = img[y1:y2 + 1, x1:x2 + 1]
            try:
                patch = Image.fromarray(patch).convert("RGB")
            except:
                print(patch.shape)
                continue

            pred_gt,pred = inference(model, patch, transform)
            print(pred.shape)
            vis_mask_hot[y1:y2 + 1, x1:x2 + 1] = pred
            mask[y1:y2 + 1, x1:x2 + 1] = pred_gt
            mask_box[y1:y2 + 1, x1:x2 + 1] = 1
            vis_mask = mask_image(vis_mask,mask,rgb)
            vis_mask_box = mask_image(vis_mask_box, mask_box,rgb)
            # get bbox rbox
            _, cbox = get_pseudo_label(mask) # curve box
            bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='int32')
            if debug_flag:
                thickness = 15
                cv2.drawContours(img, [bbox], 0, (0, 255, 0), thickness)
                # if is_ignore == 1:
                    # cv2.drawContours(img, [cbox], 0, (0, 0, 255), thickness)
                    # cv2.drawContours(img, [bbox], 0, (0, 255, 0), thickness)
                    # cv2.drawContours(img, [polygon_list[j]], 0, (255, 0, 0), thickness)
                # else:
                #     cv2.drawContours(img, [cbox], 0, (0, 255, 255), thickness)
                #     cv2.drawContours(img, [bbox], 0, (255, 255, 0), thickness)
                #     cv2.drawContours(img, [polygon_list[j]], 0, (255, 0, 255), thickness)

            gt[img_path.split("/")[-1].split(".")[0]][j]['points'] = cbox


        if debug_flag:
            print('debug vis')
            vis_mask_hot = cvt2HeatmapImg(vis_mask_hot)
            cv2.imwrite('{}/hot{}.png'.format(img_save_path, i),vis_mask_hot)
            cv2.imwrite('{}/img{}.png'.format(img_save_path, i), img[:,:,[2,1,0]])
            cv2.imwrite('{}/img_mask{}.png'.format(img_save_path, i), vis_mask[:, :, [2, 1, 0]])
            cv2.imwrite('{}/img_mask_box{}.png'.format(img_save_path, i), vis_mask_box[:, :, [2, 1, 0]])

        break
    else:
        print('{} not exist!'.format(img_path))



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

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

# 顺时针旋转90度
def RotateClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 1)
    return new_img

def inference(model, image, transform):
    origin_h, origin_w = image.height, image.width

    if origin_h > origin_w:
        image = image.transpose(Image.ROTATE_90)  # 将图片旋转90度

    resize_h, resize_w = image.height, image.width
    resized_img_ = image.resize(cfg.TRAIN.BASE_SIZE)
    resized_img = transform(resized_img_).unsqueeze(0).cuda()
    with torch.no_grad():
        output,weight = model(resized_img)
    pred = Image.fromarray(torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy().astype('int32')).resize(image.size)

    hot = cv2.resize(output[0][0][1].sigmoid().squeeze(0).cpu().data.numpy().astype('float'),(resize_w,resize_h))
    if origin_h > origin_w:
        pred = pred.transpose(Image.ROTATE_270)  # 将图片旋转90度
        hot = RotateClockWise90(hot)
    pred = np.array(pred)
    return pred, hot

def get_bboxes(img_path, data_polygt):
    img = Image.open(img_path)
    boxes = []
    ptss = []
    for i, lines in enumerate(data_polygt):
        points = lines["points"]
        word = lines["illegibility"]

        # x, y, w_1, h_1 = cv2.boundingRect(arr.astype('int32'))
        # box=[x,y,x+w_1,y,x+w_1,y+h_1,x,y+h_1]
        # box = np.asarray(box) / ([w * 1.0, h * 1.0] * 4)

        bbox = np.array(points).flatten()
        bbox = bbox.reshape(int(bbox.shape[0] / 2), 2)
        ptss.append(bbox)
        x, y, w, h = cv2.boundingRect(bbox)
        if word:
            boxes.append([x, y, x + w, y + h, 0])
        else:
            boxes.append([x, y, x + w, y + h, 1])
    return img, boxes, ptss

if __name__ == '__main__':
    demo()