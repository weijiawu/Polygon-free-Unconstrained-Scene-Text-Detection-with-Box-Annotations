# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : weijia
import json
import os
import random
import pathlib
import pyclipper
from torch.utils import data
import glob
from PIL import Image
import numpy as np
import cv2
from dataset.augment import DataAugment
from utils.utils import draw_bbox
from torchvision import transforms
import torch
import mmcv
import math

data_aug = DataAugment()


def check_and_validate_polys(polys, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

    validated_polys = []
    for poly in polys:
        p_area = cv2.contourArea(poly)
        if abs(p_area) < 1:
            continue
        validated_polys.append(poly)
    return np.array(validated_polys)


def generate_rbox(im_size, text_polys, text_tags, training_mask, i, n, m):
    """
    生成mask图，白色部分是文本，黑色是背景
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param text_tags: 标注文本框是否参与训练
    :return: 生成的mask图
    """
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    for poly, tag in zip(text_polys, text_tags):
        poly = poly.astype(np.int)
        r_i = 1 - (1 - m) * (n - i) / (n - 1)
        d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        # pco.AddPath(pyclipper.scale_to_clipper(poly), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # shrinked_poly = np.floor(np.array(pyclipper.scale_from_clipper(pco.Execute(-d_i)))).astype(np.int)
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))
        cv2.fillPoly(score_map, shrinked_poly, 1)
        # 制作mask
        # rect = cv2.minAreaRect(shrinked_poly)
        # poly_h, poly_w = rect[1]

        # if min(poly_h, poly_w) < 10:
        #     cv2.fillPoly(training_mask, shrinked_poly, 0)
        if not tag:
            cv2.fillPoly(training_mask, shrinked_poly, 0)
        # 闭运算填充内部小框
        # kernel = np.ones((3, 3), np.uint8)
        # score_map = cv2.morphologyEx(score_map, cv2.MORPH_CLOSE, kernel)
    return score_map, training_mask


def augmentation(im: np.ndarray, text_polys: np.ndarray, scales: np.ndarray, degrees: int, input_size: int) -> tuple:
    # the images are rescaled with ratio {0.5, 1.0, 2.0, 3.0} randomly
    im, text_polys = data_aug.random_scale(im, text_polys, scales)
    # the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    if random.random() < 0.5:
        im, text_polys = data_aug.horizontal_flip(im, text_polys)
    if random.random() < 0.5:
        im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)

    return im, text_polys

def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = (int)(h * scale + 0.5)
    w = (int)(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def random_scale(img):
    h, w = img.shape[0:2]

    # base_scale = 1
    min_scale = 640.0 / min(h, w)
    # max_scale = 2000.0 / max(h, w)
    base_scale = min_scale

    random_scale = np.array([0.6,0.7,0.8,0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.4, 1.3])
    scale = np.random.choice(random_scale) * base_scale

    img = scale_aligned(img, scale)
    return img

def image_label(im, text_polys: np.ndarray, text_tags: list, n: int, m: float, input_size: int) -> tuple:
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''


    h, w, _ = im.shape
    # 检查越界
    im = random_scale(im)

    h, w, _ = im.shape
    if text_polys.shape[0] > 0:
        text_polys = np.reshape(text_polys * ([im.shape[1], im.shape[0]] * 4),
                                (text_polys.shape[0], int(text_polys.shape[1] / 2), 2)).astype('int32')

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    score_maps = []
    for i in range(1, n + 1):
        # s1->sn,由小到大
        score_map, training_mask = generate_rbox((h, w), text_polys, text_tags, training_mask, i, n, m)
        score_maps.append(score_map)
    score_maps = np.array(score_maps, dtype=np.float32)

    imgs = [im, training_mask]
    imgs.extend(score_maps)

    # if random.random()<0.8:
    imgs = data_aug.random_horizontal_flip(imgs)
    imgs = data_aug.random_rotate(imgs)
    imgs = data_aug.random_crop_padding(imgs, (input_size, input_size))
    # else:
    # imgs = data_aug.resize_author(imgs, (input_size, input_size))
    im, training_mask, score_maps = imgs[0], imgs[1], imgs[2:]

    im = Image.fromarray(im)
    im = im.convert('RGB')
    im = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(im)

    im = transforms.ToTensor()(im)
    im = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(im)
    score_maps = torch.from_numpy(np.array(score_maps)).float()
    training_mask = torch.from_numpy(training_mask).float()

    return im, score_maps, training_mask
    # img: image
    #

def extract_vertices(img,lines):
    '''extract vertices info from txt lines
    Input:
        lines   : list of string info
    Output:
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
    '''
    # h, w = img.height, img.width
    h, w = img.shape[:2]
    labels = []
    vertices = []
    for line in lines:
        box = list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8]))
        box = np.asarray(box) / ([w * 1.0, h * 1.0] * 4)
        vertices.append(box)
        label = 0 if '###' in line else 1
        labels.append(label)
    return np.array(vertices), np.array(labels)

def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    # h, w = img.height, img.width
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
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

        x, y, w_1, h_1 = cv2.boundingRect(bbox.astype('int32'))
        bbox=[x,y,x+w_1,y,x+w_1,y+h_1,x,y+h_1]
        bbox = np.array(np.asarray(np.array(bbox).reshape(-1))) / ([w * 1.0, h * 1.0] * 4)

        # bbox = np.array(np.asarray(bbox.reshape(-1))) / ([w * 1.0, h * 1.0] * 4)

        bboxes.append(bbox)
        words.append('???')
    return np.array(bboxes), words


class MSRA_TD500(data.Dataset):
    def __init__(self, msra_train, hust_train,is_box_pseudo=False, data_shape: int = 640, n=6, m=0.5):
        self.is_box_pseudo = is_box_pseudo
        if self.is_box_pseudo:
            data_dirs = [os.path.join(msra_train, 'MSRA-TD500', 'train'), os.path.join(hust_train, "HUST-TR400")]
            gt_dirs = [os.path.join(msra_train, "pseudo_label", "gt_pseudo"),
                       os.path.join(hust_train, "pseudo_label", "gt_pseudo")]
        else:
            data_dirs = [os.path.join(msra_train, 'MSRA-TD500', 'train'), os.path.join(hust_train, "HUST-TR400")]
            gt_dirs = [os.path.join(msra_train, 'MSRA-TD500', 'train'), os.path.join(hust_train, "HUST-TR400")]

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = [img_name for img_name in mmcv.utils.scandir(data_dir) if img_name.endswith('.JPG')]
            img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir) if img_name.endswith('.jpg')])

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = os.path.join(data_dir, img_name)
                img_paths.append(img_path)

                if self.is_box_pseudo:
                    gt_name = img_name.split('.')[0] + '.txt'
                else:
                    gt_name = img_name.split('.')[0] + '.gt'
                gt_path = os.path.join(gt_dir, gt_name)
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)
        self.data_shape = data_shape
        self.n = n
        self.m = m

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.is_box_pseudo:
            with open(self.gt_paths[index], "r") as f:
                lines = f.readlines()
            vertices, labels = extract_vertices(img, lines)
        else:
            vertices, labels = get_ann(img, self.gt_paths[index])

        img, score_maps, training_mask = image_label(img, vertices, labels, input_size=self.data_shape,
                                                     n=self.n,
                                                     m=self.m)
        return img, score_maps, training_mask

    def __len__(self):
        return len(self.img_paths)

    def save_label(self, img_path, label):
        save_path = img_path.replace('img', 'save')
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        img = draw_bbox(img_path, label)
        cv2.imwrite(save_path, img)
        return img


if __name__ == '__main__':
    import torch
    import config
    from utils.utils import show_img
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms

    msra_data = '/home/xjc/Dataset/MSRA-TD500/'
    hust_path = '/home/xjc/Dataset/HUST-TR400/'
    train_data = MSRA_TD500(msra_data,hust_path, data_shape=640, n=6, m=0.5)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    for i in range(len(train_data)):
        img, kernel, mask = train_data[random.randint(0, 700)]
        img = img.permute(1, 2, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = img.data.cpu().numpy()
        img = img.astype('uint8')
        i = 0
        cv2.imwrite('/home/xjc/Desktop/CVPR_SemiText/SemiText/PSENet_box_supervision/workspace/img{}.png'.format(i),
                    img)
        for idx, k in enumerate(kernel):
            cv2.imwrite(
                '/home/xjc/Desktop/CVPR_SemiText/SemiText/PSENet_box_supervision/workspace/kernel{}_{}.png'.format(i,
                                                                                                                    idx),
                k.data.cpu().numpy() * 255)
        cv2.imwrite('/home/xjc/Desktop/CVPR_SemiText/SemiText/PSENet_box_supervision/workspace/mask{}.png'.format(i),
                    mask.data.cpu().numpy() * 255)
        break
