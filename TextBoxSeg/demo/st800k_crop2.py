import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random
from IPython import embed
import re
import itertools

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


def get_patchs(img, r_box, char_boxs):
    patch_imgs, patch_masks = [], []
    img_h, img_w, _ = img.shape
    contours = []
    for c in char_boxs:
        contours.append(c[0])
        contours.append(c[1])
    for c in char_boxs[::-1, :, :]:
        contours.append(c[2])
        contours.append(c[3])
    contours = np.array(contours)

    rect = cv2.boundingRect(r_box)
    x, y, w, h = rect
    mask = np.zeros((img_h, img_w))
    cv2.fillPoly(mask, [contours], 1)

    patch_img = img[y:y + h + 1, x:x + w + 1]
    patch_imgs.append(patch_img)
    patch_mask = mask[y:y + h + 1, x:x + w + 1]
    patch_masks.append(patch_mask)

    return patch_imgs, patch_masks


if __name__ == "__main__":
    st800k_path = '/data/glusterfs_cv_04/11121171/data/SynthText/'
    save_path = '/data/glusterfs_cv_04/11121171/data/SynthText/'
    max_num = 15000

    gt_mat = '{}/gt.mat'.format(st800k_path)

    print('load mat...')
    mat = sio.loadmat(gt_mat)
    print('mat loaded')

    charBB, wordBB = mat['charBB'][0], mat['wordBB'][0]
    wordText = mat['txt'][0]

    img_names = mat['imnames'][0]
    img_paths = [os.path.join(st800k_path, i[0]) for i in img_names]

    shuffle_ids = [i for i in range(len(img_paths))]
    random.shuffle(shuffle_ids)
    save_number = 0
    total_p_imgs, total_p_masks = [], []
    print('generate patch...')
    for img_idx in tqdm(shuffle_ids[:max_num]):
        assert len(total_p_imgs) == len(total_p_masks)
        img = readimg(img_paths[img_idx])

        if len(wordBB[img_idx].shape) == 2:
            continue

        r_boxes = wordBB[img_idx].transpose(2, 1, 0)
        r_boxes = np.array(r_boxes, dtype='int32')

        char_boxes = np.array(charBB[img_idx].transpose(2, 1, 0),dtype='int32')
        text = wordText[img_idx]
        words = [re.split(' \n|\n |\n| ', t.strip()) for t in text]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        total = 0
        for i in range(len(words)):
            r_box = r_boxes[i]
            char_box = char_boxes[total:total+len(words[i])]
            assert (len(char_box) == len(words[i]))
            total += len(words[i])
            patch_img, patch_mask = get_patchs(img, r_box, char_box)


            # total_p_imgs.extend(patch_img)
            # total_p_masks.extend(patch_mask)
            for y in range(len(patch_img)):
                if min(img.shape[:2]) < 20:
                    continue
                img_path = os.path.join(save_path, 'image', '{}.png'.format(save_number))
                mask_path = os.path.join(save_path, 'mask', '{}.png'.format(save_number))
                print(img_path)
                cv2.imwrite(img_path, patch_img[y])
                cv2.imwrite(mask_path, patch_mask[y])
                save_number+=1

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