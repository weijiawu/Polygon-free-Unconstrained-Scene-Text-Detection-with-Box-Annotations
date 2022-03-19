import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import torch.nn as nn
import torch
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
from tools.demo_ctw1500 import parse_img_gt
from torchvision import transforms
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg
from IPython import embed
import numpy as np
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



debug_flag = True
img_save_path = './demo/trash/ctw1500_image'
os.system('rm -rf {}/*'.format(img_save_path))
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
    if os.path.exists('/data/data_weijiawu/ctw1500/train/'):
        ic15_root_path = '/data/data_weijiawu/ctw1500/train/'
    else:
        ic15_root_path = '/data/data_weijiawu/ctw1500/train/'
    ctw_train_data = ic15_root_path + 'text_image'
    ctw_train_gt = ic15_root_path + 'text_label_curve'
    assert os.path.exists(ctw_train_data) and os.path.exists(ctw_train_gt)

    '''bbox + pseudo box '''
    CTW_train_gt_bbox = os.path.join(ic15_root_path , 'SBP_gt_bbox')
    CTW_train_gt_pseudobox = os.path.join(ic15_root_path , 'SBP_gt_pseudo')



    if not os.path.exists(CTW_train_gt_bbox):
        os.mkdir(CTW_train_gt_bbox)
    if not os.path.exists(CTW_train_gt_pseudobox):
        os.mkdir(CTW_train_gt_pseudobox)
    os.system('rm -rf {}/*txt'.format(CTW_train_gt_bbox))
    os.system('rm -rf {}/*txt'.format(CTW_train_gt_pseudobox))

    image_list = os.listdir(ctw_train_data)
    annotation_list = ['{}.txt'.format(img_name.split('.')[0]) for img_name in image_list]

    #遍历图片
    for idx,(img_path,gt_path) in enumerate(zip(image_list, annotation_list)):
        img_path_ = os.path.join(ctw_train_data, img_path)
        gt_path_ = os.path.join(ctw_train_gt, gt_path)

        if os.path.exists(gt_path_) and os.path.exists(img_path_):
            img, boxes, ori_box = parse_img_gt(img_path_, gt_path_)
            img = np.array(img)
            #遍历box

            f_bbox = open(os.path.join(CTW_train_gt_bbox, gt_path),'w')
            f_rbox = open(os.path.join(CTW_train_gt_pseudobox, gt_path), 'w')

            seq_bbox, seq_rbox = [],[]

            for j, box in enumerate(boxes):
                mask = np.zeros_like(img)[:, :, 0]
                x1, y1, x2, y2 = box
                patch = img[y1:y2 + 1, x1:x2 + 1]
                patch = Image.fromarray(patch)
                pred_gt,confident = inference(model, patch, transform)
                # if not confident:
                #     continue

                mask[y1:y2 + 1, x1:x2 + 1] = pred_gt

                #get bbox rbox
                try:
                    _, rbox = get_pseudo_label(mask)
                except:
                    continue

                bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='int32')
                if debug_flag:
                    if confident:
                        cv2.drawContours(img, [rbox], 0, (0, 0, 255), 2)
                        cv2.drawContours(img, [bbox], 0, (0, 255, 0), 2)
                        cv2.drawContours(img, [ori_box[j]], 0, (255, 0, 0), 2)
                    else:
                        cv2.drawContours(img, [rbox], 0, (0, 255, 255), 2)
                        cv2.drawContours(img, [bbox], 0, (0, 255, 0), 2)
                        cv2.drawContours(img, [ori_box[j]], 0, (255, 0, 0), 2)

                bbox, rbox = list(bbox.reshape(-1)), list(rbox.reshape(-1))

                if confident:
                    seq_bbox.append(",".join([str(int(i)) for i in bbox]) + ',aaa\n')
                    seq_rbox.append(",".join([str(int(i)) for i in rbox]) + ',aaa\n')
                else:
                    seq_bbox.append(",".join([str(int(i)) for i in bbox]) + ',###\n')
                    seq_rbox.append(",".join([str(int(i)) for i in rbox]) + ',###\n')

            f_bbox.writelines(seq_bbox)
            f_rbox.writelines(seq_rbox)
            f_bbox.close()
            f_rbox.close()

            if debug_flag:
                print('debug vis')
                cv2.imwrite('{}/img{}.png'.format(img_save_path,idx), img[:,:,[2,1,0]])
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

def scale_image(img,short_line = 96, scale=32):
    resize_h, resize_w = img.height, img.width
    # short_line = min(resize_h,resize_w)
    scale_1 = short_line / resize_h

    resize_w =  int(resize_w*scale_1)
    resize_h =  short_line
    resize_w =  int(resize_w / scale) * scale

    img = img.resize((resize_w,resize_h))
    return img

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

    resized_img = image.resize(cfg.TRAIN.BASE_SIZE).convert("RGB")
    # resized_img = scale_image(image, cfg.TRAIN.BASE_SIZE[1]).convert("RGB")
    resized_img = transform(resized_img).unsqueeze(0).cuda()
    with torch.no_grad():
        output,_ = model(resized_img)
    pred = Image.fromarray(torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy().astype('int32')).resize(image.size)

    # fileter
    # print(output[0][:,1,:,:].shape)
    # m = nn.Sigmoid()
    # confidence = m(output[0][:,1,:,:])
    # confidence = ((confidence > 0.5) * confidence).sum()/((confidence > 0.5)*1.0).sum()
    # print(confidence)


    if origin_h > origin_w:
        pred = pred.transpose(Image.ROTATE_270)  # 将图片旋转90度
    pred = np.array(pred)

    # if confidence<0.7:
    #     return pred, False
    # else:
    return pred, True



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    demo()

