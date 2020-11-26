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
import torch
import numpy as np
import cv2
from dataset.augment import DataAugment
from utils.utils import draw_bbox
import scipy.io as scio
from PIL import Image
import torchvision.transforms as transforms
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


def generate_rbox(im_size, text_polys, training_mask, i, n, m):
    """
    生成mask图，白色部分是文本，黑色是背景
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param text_tags: 标注文本框是否参与训练
    :return: 生成的mask图
    """
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    for poly in zip(text_polys):
        poly = np.array(poly[0],dtype=np.int)

        r_i = 1 - (1 - m) * (n - i) / (n - 1)
        d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        # pco.AddPath(pyclipper.scale_to_clipper(poly), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # shrinked_poly = np.floor(np.array(pyclipper.scale_from_clipper(pco.Execute(-d_i)))).astype(np.int)
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))
        cv2.fillPoly(score_map, shrinked_poly, 1)

    return score_map, training_mask


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


def random_scale(img, min_size):
    h, w = img.shape[0:2]

    random_scale = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    scale = (np.random.choice(random_scale) * 640.0) / min(h, w)

    # img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    img = scale_aligned(img, scale)
    return img

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def random_crop_padding(imgs, target_size):
    """ using padding and the final crop size is (800, 800) """
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs

def image_label(img, text_polys: np.ndarray, n: int, m: float, input_size: int):
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''

    h, w, _ = img.shape
    # 检查越界
    # text_polys = check_and_validate_polys(text_polys, (h, w))
    img = random_scale(img, input_size)

    # h, w, _ = img.shape
    # short_edge = min(h, w)
    # if short_edge < input_size:
    #     保证短边 >= inputsize
        # scale = input_size / short_edge
        # im = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        # text_polys *= scale
    if text_polys.shape[0] > 0:
        text_polys = np.reshape(text_polys * ([img.shape[1], img.shape[0]] * 4),
                            (text_polys.shape[0], int(text_polys.shape[1] / 2), 2)).astype('int32')

    h, w, _ = img.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    score_maps = []
    for i in range(1, n + 1):
        # s1->sn,由小到大
        score_map, training_mask = generate_rbox((h, w), text_polys, training_mask, i, n, m)
        score_maps.append(score_map)
    score_maps = np.array(score_maps, dtype=np.float32)


    imgs = [img, training_mask]
    imgs.extend(score_maps)

    imgs = random_horizontal_flip(imgs)
    imgs = random_rotate(imgs)
    imgs = random_crop_padding(imgs, (input_size,input_size))

    img, training_mask, score_maps = imgs[0], imgs[1], imgs[2:]

    return img, score_maps, training_mask
    # img: image
    #

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img
class Synthtext(data.Dataset):
    def __init__(self, data_dir, data_shape: int = 640, n=6, m=0.5):
        self.data_dir = data_dir
        self.data_shape = data_shape
        self.n = n
        self.m = m

        synth_train_gt_path = data_dir + 'gt.mat'
        data = scio.loadmat(synth_train_gt_path)

        self.img_paths = data['imnames'][0]
        self.gts = data['wordBB'][0]

    def __getitem__(self, index):

        img_path = self.data_dir + self.img_paths[index][0]
        img = get_img(img_path)

        bboxes = np.array(self.gts[index])
        bboxes = np.reshape(bboxes, (bboxes.shape[0], bboxes.shape[1], -1))
        bboxes = bboxes.transpose(2, 1, 0)
        bboxes = np.reshape(bboxes, (bboxes.shape[0], -1)) / ([img.shape[1], img.shape[0]] * 4)


        img, score_maps, training_mask = image_label(img, bboxes, input_size=self.data_shape,
                                                     n=self.n,
                                                     m=self.m)
        img = Image.fromarray(img)
        img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        score_maps = torch.from_numpy(np.array(score_maps)).float()
        training_mask = torch.from_numpy(training_mask).float()

        return img, score_maps, training_mask
        # img: image
        # score map:  各种size 的kernal
        # training mask ： 一个map包含ignore

    def __len__(self):
        return self.img_paths.shape[0]

    def save_label(self, img_path, label):
        save_path = img_path.replace('img', 'save')
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        img = draw_bbox(img_path, label)
        cv2.imwrite(save_path, img)
        return img


if __name__ == '__main__':
    import torch
    import utils.config_synthtext as config
    from utils.utils import show_img
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms

    train_data = Synthtext(config.trainroot, data_shape=config.data_shape, n=config.n, m=config.m)
    train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=False, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    for i, (img, label, mask) in enumerate(train_loader):
        print(label.shape)
        print(img.shape)
        print(label[0][-1].sum())
        print(mask[0].shape)
        # pbar.update(1)
        show_img((img[0] * mask[0].to(torch.float)).numpy().transpose(1, 2, 0)*[0.229, 0.224, 0.225]+[0.485, 0.456, 0.406], color=True)
        show_img(label[0])
        show_img(mask[0])
        plt.show()
        break
    pbar.close()
