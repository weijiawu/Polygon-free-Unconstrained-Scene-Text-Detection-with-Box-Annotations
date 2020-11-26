import os
import sys
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import mmcv
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
import math
import cv2
import scipy.io as sio
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
def scale_image(img, scale=32):
    resize_h, resize_w = img.height, img.width

    base_size = 96
    resize_w = (resize_w / resize_h * base_size)
    resize_h = base_size

    # resize_h = (int(resize_h / scale) + 1) * scale
    resize_w = min((int(resize_w / scale) + 1) * scale,160)

    # resize_w = max(32,resize_w)
    # resize_h = max(32,resize_h)

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
    output_dir = '/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/msra_attention/'
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
    msra_train = os.path.abspath('/home/xjc/Dataset/MSRA-TD500/MSRA-TD500/')
    hust_train = os.path.abspath("/home/xjc/Dataset/HUST-TR400/HUST-TR400/")
    data_dirs = [os.path.join(msra_train, 'train'), hust_train]
    gt_dirs = [os.path.join(msra_train, 'train'), hust_train]

    img_paths = []
    gt_paths = []

    for data_dir, gt_dir in zip(data_dirs, gt_dirs):
        img_names = [img_name for img_name in mmcv.utils.scandir(data_dir) if img_name.endswith('.JPG')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir) if img_name.endswith('.jpg')])

        img_paths_ = []
        gt_paths_ = []
        for idx, img_name in enumerate(img_names):
            img_path = os.path.join(data_dir, img_name)
            img_paths_.append(img_path)

            gt_name = img_name.split('.')[0] + '.gt'
            gt_path = os.path.join(gt_dir, gt_name)
            gt_paths_.append(gt_path)

        img_paths.extend(img_paths_)
        gt_paths.extend(gt_paths_)

    patch_imgs = []
    patch_gt = []
    for idx,img_path_one in enumerate(img_paths):
        img_path = img_path_one
        gt_path = gt_paths[idx]

        if os.path.exists(gt_path) and os.path.exists(img_path):
            boxes,polygon_list = get_ann( gt_path)
            img = cv2.imread(img_path)
            img = np.array(img)
            if boxes == []:
                continue
            for bo_idx,box in enumerate(boxes):
                x1, y1, x2, y2 = box
                # print(y1,y2 + 1, x1,x2 + 1)
                patch = img[y1:y2 + 1, x1:x2 + 1]
                patch_imgs.append(Image.fromarray(patch))

                gt_image = np.zeros(img.shape[:2],dtype=np.uint8).copy()
                # print(np.array(polygon_list[bo_idx])[:8].reshape(4, 2).astype("int16"))
                cv2.fillPoly(gt_image,[np.array(polygon_list[bo_idx],dtype=np.int32)[:8].reshape(4, 2)],1)
                gt_path = gt_image[y1:y2 + 1, x1:x2 + 1]
                patch_gt.append(gt_path)
                # 先只测500张
                if len(patch_imgs) > 500:
                    break
        else:
            print(img_path)
    print('total patch images:{}'.format(len(patch_imgs)))

    pool_imgs, pool_masks, pool_gts, dist_imgs, dist_img_pres = [], [], [], [], []
    count = 0
    for idx_image,image in enumerate(patch_imgs):
        # image = Image.open(img_path).convert('RGB')
        gt_path_one = patch_gt[idx_image]
        origin_h, origin_w = image.height, image.width
        if origin_h>origin_w:
            image = image.transpose(Image.ROTATE_90)  # 将图片旋转90度
            gt_path_one = RotateClockWise90(gt_path_one)

        # cfg.TRAIN.BASE_SIZE
        resized_img = image.resize(cfg.TRAIN.BASE_SIZE)
        # resized_img = scale_image(image)
        resized_img = transform(resized_img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output, skeleton = model(resized_img)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()

        skeleton_1 = (skeleton[0][0]*255).squeeze(0).cpu().data.numpy().astype("int64")

        dist_img = np.array(get_color_pallete(skeleton_1, cfg.DATASET.NAME))[:, :, None].repeat(3, -1)
        dist_img_pre = cv2.resize(dist_img, (128, 96))


        img = np.array(image.resize((128,96)))

        dis_ = gt_path_one.copy()
        gt_path_one = np.array(get_color_pallete(gt_path_one, cfg.DATASET.NAME))[:, :, None].repeat(3, -1) * 255
        gt_path_one = cv2.resize(gt_path_one, (128, 96))

        dis_ = cv2.resize(dis_, (128, 96))
        dist_img = cv2.distanceTransform(np.array(dis_), cv2.DIST_L1, cv2.DIST_MASK_3)
        # dist_back = cv2.distanceTransform((1 - np.array(dis_)), cv2.DIST_L1, cv2.DIST_MASK_3)
        dist_img = (dist_img / dist_img.max() * 0.5 + 0.5)
        dist_img = np.array(get_color_pallete(dist_img* 255, cfg.DATASET.NAME))[:, :, None].repeat(3, -1)


        mask = np.array(get_color_pallete(pred, cfg.DATASET.NAME))[:,:,None].repeat(3,-1) * 255
        kernel = np.ones((6, 6), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # erosion = cv2.erode(img, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        mask = cv2.resize(mask,(128,96))

        if len(pool_imgs)<20:
            pool_imgs.append(img)
            pool_masks.append(mask)
            pool_gts.append(gt_path_one)
            dist_imgs.append(dist_img)
            dist_img_pres.append(dist_img_pre)
        else:
            big_img = np.concatenate(pool_imgs, axis=0)
            big_mask = np.concatenate(pool_masks, axis=0)
            pool_gt = np.concatenate(pool_gts, axis=0)
            pool_dist_img_pres = np.concatenate(dist_img_pres, axis=0)
            pool_dist_img = np.concatenate(dist_imgs, axis=0)

            big_img_mask = Image.fromarray(np.concatenate([big_img, big_mask,pool_gt,pool_dist_img_pres,pool_dist_img], axis=1))
            big_img_mask.save('{}/{}.png'.format(output_dir, count))
            print('{}/{}.png'.format(output_dir, count))
            count += 1
            pool_imgs, pool_masks, pool_gts, dist_imgs, dist_img_pres = [], [], [], [], []

def get_ann( gt_path):
    lines = mmcv.list_from_file(gt_path)
    polygon_list = []
    words = []
    boxes = []
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
        # print(np.asarray(bbox.reshape(-1)))
        x, y, w, h = cv2.boundingRect(bbox)
        boxes.append([max(x,0), max(y,0), max(x + w,0), max(y + h,0)])

        bbox = np.array(np.asarray(bbox.reshape(-1)))
        # bbox = bbox.reshape(-1)

        polygon_list.append(bbox)
        words.append('???')
    return boxes,polygon_list

if __name__ == '__main__':
    demo()