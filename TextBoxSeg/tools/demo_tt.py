import os
import sys
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg
from tools.gen_tt_pslabel import get_pseudo_label
from IPython import embed
import numpy as np
from tqdm import trange
import cv2
import scipy.io as sio
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
def scale_image(img, scale=32):
    resize_h, resize_w = img.height, img.width

    resize_h = resize_h if resize_h % scale == 0 else int(resize_h / scale) * scale
    resize_w = resize_w if resize_w % scale == 0 else int(resize_w / scale) * scale

    resize_w = max(32,resize_w)
    resize_h = max(32,resize_h)
    img = img.resize((resize_w,resize_h))

    return img
# 顺时针旋转90度
def RotateClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 0)
    return new_img


def demo():
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    # output folder
    output_dir = '/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/TT_origin'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()

    #get img_patch from Total Text
    if os.path.exists('/home/xjc/Dataset/total-text/'):
        total_root_path = '/home/xjc/Dataset/total-text/'
    else:
        total_root_path = '/home/xjc/Dataset/total-text/'
    total_train_data = total_root_path + 'Images/Train/'
    total_train_gt = total_root_path + 'gt/Train/'
    assert os.path.exists(total_train_data) and os.path.exists(total_train_gt)


    patch_imgs = []
    patch_gt = []
    for i in trange(1, 501):
        img_path = 'img{}.jpg'.format(i)
        img_path = os.path.join(total_train_data, img_path)
        gt_path = 'poly_gt_img{}.mat'.format(i)
        gt_path = os.path.join(total_train_gt, gt_path)

        if os.path.exists(gt_path) and os.path.exists(img_path):
            img, boxes,polygon_list = parse_img_gt(img_path, gt_path)
            img = np.array(img)
            if boxes == []:
                continue
            for bo_idx,box in enumerate(boxes):
                x1, y1, x2, y2 = box
                patch = img[y1:y2 + 1, x1:x2 + 1]
                patch_imgs.append(Image.fromarray(patch))

                gt_image = np.zeros(img.shape[:2],dtype=np.uint8)
                # print(polygon_list[bo_idx])
                cv2.fillPoly(gt_image,[np.array(polygon_list[bo_idx])],1)
                gt_path = gt_image[y1:y2 + 1, x1:x2 + 1]
                patch_gt.append(gt_path)
                # 先只测500张
                if len(patch_imgs) > 500:
                    break
        else:
            print(img_path)
    print('total patch images:{}'.format(len(patch_imgs)))

    pool_imgs, pool_masks, pool_gts = [], [], []
    count = 0
    for idx_image,image in enumerate(patch_imgs):
        # image = Image.open(img_path).convert('RGB')
        gt_path_one = patch_gt[idx_image]
        origin_h, origin_w = image.height, image.width
        if origin_h>origin_w:
            image = image.transpose(Image.ROTATE_90)  # 将图片旋转90度
            gt_path_one = RotateClockWise90(gt_path_one)


        # (128,32)  cfg.TRAIN.BASE_SIZE
        resized_img = image.resize((160,96))
        # resized_img = scale_image(image)
        resized_img = transform(resized_img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = model(resized_img)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()


        img = np.array(image.resize((128,96)))
        gt_path_one = np.array(get_color_pallete(gt_path_one, cfg.DATASET.NAME))[:, :, None].repeat(3, -1) * 255
        gt_path_one = cv2.resize(gt_path_one, (128, 96))

        mask = np.array(get_color_pallete(pred, cfg.DATASET.NAME))[:,:,None].repeat(3,-1) * 255
        kernel = np.ones((6, 6), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # erosion = cv2.erode(img, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        mask = cv2.resize(mask,(128,96))

        # # mask = np.expand_dims(np.array(mask), -1)
        # # print(mask.shape)
        # mask_ = mask.copy().astype('float32')
        # # kernel = np.ones((3, 3), np.uint8)
        # # mask_ = cv2.dilate(mask_, kernel, iterations=1)
        # mask_ = np.array(mask_, np.uint8)
        # mask_1 = mask_[:,:,0].copy()
        # try:
        #     contours, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # except:
        #     _, contours, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
        # cnt = contours[0]
        #
        # mask_ = np.zeros_like(mask)
        # cv2.fillPoly(mask_, np.array([cnt]), 1)
        # # mask_ = np.expand_dims(np.array(mask_), -1)
        # # mask_ = np.concatenate((mask_,mask_,mask_),-1)
        # # print(mask_.shape)
        # mask = mask_* 255

        if len(pool_imgs)<20:
            pool_imgs.append(img)
            pool_masks.append(mask)
            pool_gts.append(gt_path_one)

        else:
            big_img = np.concatenate(pool_imgs, axis=0)
            big_mask = np.concatenate(pool_masks, axis=0)
            pool_gt = np.concatenate(pool_gts, axis=0)
            big_img_mask = Image.fromarray(np.concatenate([big_img, big_mask,pool_gt], axis=1))
            big_img_mask.save('{}/{}.png'.format(output_dir, count))
            print('{}/{}.png'.format(output_dir, count))
            count += 1
            pool_imgs, pool_masks, pool_gts = [], [], []


def parse_img_gt(img_path, gt_path):
    img = Image.open(img_path)
    # img = cv2.imread(img_path)
    data = sio.loadmat(gt_path)['polygt']
    boxes = []
    polygon_list = []
    for d in data:
        _, xs, _, ys, text, _ = d
        if "#" in text:
            continue  # 过滤掉ignore的
        polygon_np = np.concatenate([xs, ys], 0).transpose(1, 0).astype('int32')
        polygon_list.append(polygon_np)
        x, y, w, h = cv2.boundingRect(polygon_np)
        boxes.append([x, y, x + w, y + h])
    return img, boxes,polygon_list


if __name__ == '__main__':
    demo()