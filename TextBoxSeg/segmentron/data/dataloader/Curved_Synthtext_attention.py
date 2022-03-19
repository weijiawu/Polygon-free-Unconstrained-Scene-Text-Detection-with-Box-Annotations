"""Prepare Trans10K dataset"""
import os
import torch
import numpy as np
import logging
import random
from PIL import Image
from segmentron.data.dataloader.seg_data_base import SegmentationDataset_total
# from IPython import embed
import cv2

class TextSegmentation_attention(SegmentationDataset_total):
    """Text Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Sagement folder. Default is './datasets/Trans10K'
    split: string
        'train', 'validation', 'test'
    transform : callable, optional
        A function that transforms the image
    """
    BASE_DIR = 'st800k'
    NUM_CLASS = 2

    def __init__(self, root='data/st800k_crop', split='train', mode=None, transform=None, debug=True, **kwargs):
        super(TextSegmentation_attention, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root), "Please put dataset in {}".format(root)
        self.images, self.mask_paths = _get_st800kcrop_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [0,1]
        self._key = np.array([0,1])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32') + 1
        self.debug = debug
        self.mode =mode

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        try:
            # index = random.randint(10000)
            img = Image.open(self.images[index]).convert('RGB')
            mask = Image.open(self.mask_paths[index])
            origin_h = img.height
            origin_w = img.width
            if origin_h > origin_w:
                img = img.transpose(Image.ROTATE_90)  # 将图片旋转90度
                mask = mask.transpose(Image.ROTATE_90)  # 将图片旋转90度
                origin_h = img.height
                origin_w = img.width

        except:
            print("invalid image:",self.images[index] )
            return self.__getitem__(index+1)
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])



        dist_img = cv2.distanceTransform(np.array(mask), cv2.DIST_L1, cv2.DIST_MASK_3)
        # dist_back = cv2.distanceTransform((1-np.array(mask)), cv2.DIST_L1, cv2.DIST_MASK_3)
        dist_img_ = (dist_img / dist_img.max() * 0.7 + 0.3)


        if self.mode == 'train':
            img, mask, dist_img_ = self._sync_transform(img, mask, dist_img_)
        elif self.mode == 'val':
            img, mask, dist_img_ = self._val_sync_transform(img, mask, dist_img_)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        if self.debug == True:
            # print('debug vis')
            # _img = Image.fromarray(img)
            img = np.array(img)
            mask_SHOW = np.array(mask).astype("int16")*255
            mask_SHOW = mask_SHOW[:, :, np.newaxis]
            mask_SHOW = mask_SHOW.repeat([3], axis=2)

            dist_img_SHOW = np.array(dist_img_*255).astype("int16")
            dist_img_SHOW = dist_img_SHOW[:, :, np.newaxis]
            dist_img_SHOW = dist_img_SHOW.repeat([3], axis=2)

            show = np.concatenate([img, mask_SHOW,dist_img_SHOW ], axis=1)
            print(img.shape)
            cv2.imwrite('./show/img{}.jpg'.format(index),np.array(show))

        # synchrosized transform


        # general resize, normalize and toTensor
        # print(img.shape, mask.shape, dist_img_.shape)
        # if img.shape[0]!=128:
        #     print(img.shape, mask.shape, dist_img_.shape)
        #     print(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, dist_img_, (origin_h,origin_w),  self.images[index]

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('background', 'text')


def _get_st800kcrop_pairs(folder, split='train'):

    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        imgs = os.listdir(img_folder)

        for imgname in imgs:
            imgpath = os.path.join(img_folder, imgname)
            maskname = imgname
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                logging.info('cannot find the mask or image: {} {}'.format(imgpath, maskpath))

        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split == 'train':
        img_folder  = os.path.join(folder, 'image')
        mask_folder = os.path.join(folder, 'mask')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    else:
        pass
    return img_paths, mask_paths

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from numpy import *

    train_data = TextSegmentation_attention("/home/xjc/Dataset/Curved_SynthText/SegData/",
                                mode='train')

    hs = []
    ws = []
    for i in range(len(train_data)):
        img, mask, dist_img_, wh, index = train_data[i]
        origin_h, origin_w = wh
        hs.append(origin_h)
        ws.append(origin_w)
        # print(i)
        # print(np.array(img).shape)
        # cv2.imwrite('/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/trash/img.jpg', np.array(img))
        # cv2.imwrite('/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/trash/mask.png',
        #             np.array(mask).astype("int16") * 255)
        # cv2.imwrite('/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/trash/skeleton.png',
        #             np.array(dist_img_ * 255).astype("int16"))

        break

    print(mean(hs))
    print(mean(ws))

