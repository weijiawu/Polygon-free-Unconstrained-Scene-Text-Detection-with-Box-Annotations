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

def readimg(path):
    return cv2.imread(path)

def show(img):
    if len(img.shape) == 3:
        return plt.imshow(img[:,:,::-1])
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


def get_patchs(img, r_boxes):
    patch_imgs, patch_masks = [], []
    img_h, img_w, _ = img.shape
    for i, r_box in enumerate(r_boxes):
        rect = cv2.boundingRect(r_box)
        x,y,w,h = rect
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        mask = np.zeros((img_h,img_w))
        cv2.fillPoly(mask, [r_box], 1)

        patch_img = img[y:y+h+1,x:x+w+1]; patch_imgs.append(patch_img)
        patch_mask = mask[y:y+h+1,x:x+w+1]; patch_masks.append(patch_mask)
    
    return patch_imgs, patch_masks




if __name__ == "__main__":
    st800k_path = '/mnt/lustre/share_data/xieenze/xez_space/Text/SynthText'
    save_path = '/mnt/lustre/share_data/xieenze/xez_space/Text/st800k_crop'
    max_num = 10000

    gt_mat = '{}/gt.mat'.format(st800k_path)

    print('load mat...')
    mat = sio.loadmat(gt_mat)
    print('mat loaded')

    charBB, wordBB = mat['charBB'][0], mat['wordBB'][0]
    img_names = mat['imnames'][0]
    img_paths = [os.path.join(st800k_path, i[0]) for i in img_names]

    shuffle_ids = [i for i in range(len(img_paths))]
    random.shuffle(shuffle_ids)

    total_p_imgs, total_p_masks = [], []
    print('generate patch...')
    for img_idx in tqdm(shuffle_ids[:max_num]):
        assert len(total_p_imgs) == len(total_p_masks)
        img = readimg(img_paths[img_idx])
        if len(wordBB[img_idx].shape) == 2:
            continue
        r_boxes = wordBB[img_idx].transpose(2, 1, 0)
        r_boxes = np.array(r_boxes, dtype='int32')
        p_imgs, p_masks = get_patchs(img, r_boxes)
        total_p_imgs.extend(p_imgs)
        total_p_masks.extend(p_masks)



    print('save images...')
    debug = False
    for i in tqdm(range(len(total_p_imgs))):
        img = total_p_imgs[i]
        mask = total_p_masks[i]
        if min(img.shape[:2]) < 20:
            continue
        if debug:
            print('debug vis')
            mask *= 255
        img_path = os.path.join(save_path, 'image', '{}.png'.format(i))
        mask_path = os.path.join(save_path, 'mask', '{}.png'.format(i))
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)