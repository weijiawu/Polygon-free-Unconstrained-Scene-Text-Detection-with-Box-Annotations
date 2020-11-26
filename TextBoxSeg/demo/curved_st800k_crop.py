import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random
# from IPython import embed
import re
import itertools
import json
def readimg(path):
    return cv2.imread(path)


def show(img):
    if len(img.shape) == 3:
        return plt.imshow(img[:, :, ::-1])
    else:
        return plt.imshow(img)


def makedirs(path):
    """Create directory recursively if not exists.
    Similar to `makedir -p`, you can skip checking existence before this function.
    Parameters
    ----------
    path : str
        Path of the desired dir
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


def get_patchs(img, char_boxs):
    img_h, img_w, _ = img.shape

    char_boxs = np.array(char_boxs).transpose(2, 1, 0)

    contours = []
    for c in char_boxs:
        contours.append(c[0])
        contours.append(c[1])
    for c in char_boxs[::-1, :, :]:
        contours.append(c[2])
        contours.append(c[3])
    contours = np.array(contours)

    mask = np.zeros((img_h, img_w))
    cv2.fillPoly(mask, [contours], 1)

    return img, mask


if __name__ == "__main__":
    st800k_path = '/mnt/lustre/share_data/xieenze/wuweijia/CurvedSynthText'
    save_path = '/mnt/lustre/share_data/xieenze/wuweijia/CurvedSynthText/SegData/'
    # print(len(os.listdir(os.path.join(save_path,"image"))))
    # raise NameError
    max_num = 50000
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json_filename = '{}/Label.json'.format(st800k_path)

    print('load json label..')
    with open(json_filename) as f:
        pop_data = json.load(f)
    print('json label loaded')

    shuffle_ids = [i for i in range(len(pop_data))]
    random.shuffle(shuffle_ids)
    save_number = 0
    total_p_imgs, total_p_masks = [], []
    print('generate patch...')
    for img_idx in tqdm(shuffle_ids):
        assert len(total_p_imgs) == len(total_p_masks)

        data = pop_data[img_idx]
        char_box = data["chars"]

        file_number,_,image_name =  data["img"].split("/")
        file_number = file_number.zfill(4)
        image_path = os.path.join(st800k_path,"images",file_number,image_name)

        if os.path.exists(image_path):
            image = readimg(image_path)
            try:
                patch_imgs, patch_masks = get_patchs(image,char_box)
            except:
                continue


            img_path = os.path.join(save_path, 'image', '{}.png'.format(save_number))
            mask_path = os.path.join(save_path, 'mask', '{}.png'.format(save_number))
            # print(img_path)
            # cv2.imwrite(img_path, patch_imgs)
            # cv2.imwrite(mask_path, patch_masks)
            save_number += 1
            if save_number>= max_num:
                break

        # if min(img.shape[:2]) < 20:
        #     continue

    # print('save images...')
    # debug = False
    # for i in tqdm(range(len(total_p_imgs))):
    #     img = total_p_imgs[i]
    #     mask = total_p_masks[i]
    #     if min(img.shape[:2]) < 20:
    #         continue
    #     if debug:
    #         print('debug vis')
    #         mask *= 255
    #     img_path = os.path.join(save_path, 'image', '{}.png'.format(i))
    #     mask_path = os.path.join(save_path, 'mask', '{}.png'.format(i))
    #
    #     cv2.imwrite(img_path, img)
    #     cv2.imwrite(mask_path, mask)