"""Prepare Trans10K dataset"""
import os
import torch
import numpy as np
import logging

from PIL import Image
from segmentron.data.dataloader.seg_data_base import SegmentationDataset
from IPython import embed
import cv2
import json

class TextSegmentation_CurvedST800(SegmentationDataset):
    """Trans10K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Trans10K folder. Default is './datasets/Trans10K'
    split: string
        'train', 'validation', 'test'
    transform : callable, optional
        A function that transforms the image
    """
    BASE_DIR = 'st800k'
    NUM_CLASS = 2

    def __init__(self, root='/home/xjc/Dataset/Curved_SynthText/', split='train', mode=None, transform=None, debug=False, **kwargs):
        super(TextSegmentation_CurvedST800, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root), "Please put dataset in {}".format(root)
        self.gt_name = os.path.join(self.root,"Label.json")

        self.images, self.mask_paths = _get_curved_st800_pairs(self.gt_name, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [0,1]
        self._key = np.array([0,1])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32') + 1
        self.debug = debug

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        try:
            img = Image.open(self.images[index]).convert('RGB')
        except:
            print("invalid image:",self.images[index] )
            return self.__getitem__(index+1)
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        if self.debug == True:
            print('debug vis')
            _img = Image.fromarray(img)
            _img.save('trash/img.png')
            _mask = Image.fromarray(mask.float().data.cpu().numpy()*255).convert('L')
            _mask.save('trash/mask.png')

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, self.images[index]

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


def _get_curved_st800_pairs(json_filename, split='train'):

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
        with open(json_filename) as f:
            pop_data = json.load(f)
            print(pop_data[0])
            orig = pop_data[0].transpose(2, 1, 0)
            print(orig)
            # for pop_dict in pop_data:
            #     print(pop_dict)
            #     raise NameError
                # label_id = pop_dict['label_id']
                # image_id = pop_dict['image_id']
                # temp = str(image_id) + ' ' + str(label_id)


        # img_folder  = os.path.join(folder, 'image')
        # mask_folder = os.path.join(folder, 'mask')
        # img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    # else:
    #     pass
    # return img_paths, mask_paths

if __name__ == "__main__":
    # trainset = TextSegmentation_CurvedST800()
    # for i in range(len(trainset)):
    #     image,mask,idx = trainset[i]
    #     print(image.shape)
    print(len(os.listdir("/home/xjc/Dataset/Curved_SynthText/SegData/image")))
    print(len(os.listdir("/home/xjc/Dataset/Curved_SynthText/SegData/mask")))