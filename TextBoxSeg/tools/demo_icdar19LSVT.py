import os
import sys
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
from segmentron.data.dataloader.utils import read_lines,remove_all
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
import json
import cv2

# 顺时针旋转90度
def RotateClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 0)
    return new_img


def scale_image(img,short_line = 96, scale=32):
    resize_h, resize_w = img.height, img.width
    # short_line = min(resize_h,resize_w)
    scale_1 = short_line / resize_h

    resize_w =  int(resize_w*scale_1)
    resize_h =  short_line
    resize_w =  int(resize_w / scale) * scale

    img = img.resize((resize_w,resize_h))
    return img

def demo():
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    # output folder
    output_dir = 'demo/trash/ICDAR19LSVT'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()

    ic15_train_data = '/data/data_weijiawu/ICDAR2019LSVT/train/'
    ic15_train_gt =  '/data/data_weijiawu/ICDAR2019LSVT/train_full_labels.json'
    assert os.path.exists(ic15_train_data) and os.path.exists(ic15_train_gt)

    image_list = os.listdir(ic15_train_data)

    with open(ic15_train_gt, 'r', encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    patch_imgs = []
    patch_gt = []
    for img_path in image_list:
        img_path = os.path.join(ic15_train_data, img_path)
        gt_ = gt[img_path.split("/")[-1].split(".")[0]]

        img, boxes, polygon_list = get_bboxes(img_path, gt_)
        img = np.array(img)
        if boxes == []:
            continue
        for bo_idx,box in enumerate(boxes):
            x1, y1, x2, y2 = box
            patch = img[y1:y2 + 1, x1:x2 + 1]
            patch_imgs.append(Image.fromarray(patch))

            gt_image = np.zeros(img.shape[:2], dtype=np.uint8).copy()
            # print(np.array(polygon_list[bo_idx])[:8].reshape(4, 2).astype("int16"))
            cv2.fillPoly(gt_image, [np.array(polygon_list[bo_idx], dtype=np.int32)], 1)
            gt_path = gt_image[y1:y2 + 1, x1:x2 + 1]
            patch_gt.append(gt_path)

        # 先只测500张
        if len(patch_imgs) > 50:
            break
        else:
            print(img_path)
    print('total patch images:{}'.format(len(patch_imgs)))

    pool_imgs, pool_masks, pool_gts, dist_imgs, dist_img_pres = [], [], [], [], []
    count = 0
    for idx_image, image in enumerate(patch_imgs):
        # image = Image.open(img_path).convert('RGB')
        gt_path_one = patch_gt[idx_image]
        origin_h, origin_w = image.height, image.width
        if origin_h > origin_w:
            image = image.transpose(Image.ROTATE_90)  # 将图片旋转90度
            gt_path_one = RotateClockWise90(gt_path_one)

        # cfg.TRAIN.BASE_SIZE
        resized_img = image.resize(cfg.TRAIN.BASE_SIZE)
        # resized_img = scale_image(image,cfg.TRAIN.BASE_SIZE[1])

        resized_img = resized_img.convert("RGB")
        resized_img = transform(resized_img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output, skeleton = model(resized_img)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()

        skeleton_1 = (skeleton[0][0] * 255).squeeze(0).cpu().data.numpy().astype("int64")

        dist_img = np.array(get_color_pallete(skeleton_1, cfg.DATASET.NAME))[:, :, None].repeat(3, -1)
        dist_img_pre = cv2.resize(dist_img, (128, 96))

        image = image.convert("RGB")
        img = np.array(image.resize((128, 96)))


        dis_ = gt_path_one.copy()
        gt_path_one = np.array(get_color_pallete(gt_path_one, cfg.DATASET.NAME))[:, :, None].repeat(3, -1) * 255
        gt_path_one = cv2.resize(gt_path_one, (128, 96))

        dis_ = cv2.resize(dis_, (128, 96))
        dist_img = cv2.distanceTransform(np.array(dis_), cv2.DIST_L1, cv2.DIST_MASK_3)
        # dist_back = cv2.distanceTransform((1 - np.array(dis_)), cv2.DIST_L1, cv2.DIST_MASK_3)
        dist_img = (dist_img / dist_img.max() * 0.5 + 0.5)
        dist_img = np.array(get_color_pallete(dist_img * 255, cfg.DATASET.NAME))[:, :, None].repeat(3, -1)

        mask = np.array(get_color_pallete(pred, cfg.DATASET.NAME))[:, :, None].repeat(3, -1) * 255
        kernel = np.ones((6, 6), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # erosion = cv2.erode(img, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        mask = cv2.resize(mask, (128, 96))

        if len(pool_imgs) < 20:
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

            big_img_mask = Image.fromarray(
                np.concatenate([big_img, big_mask, pool_gt, pool_dist_img_pres, pool_dist_img], axis=1))
            big_img_mask.save('{}/{}.png'.format(output_dir, count))
            print('{}/{}.png'.format(output_dir, count))
            count += 1
            pool_imgs, pool_masks, pool_gts, dist_imgs, dist_img_pres = [], [], [], [], []


def get_bboxes(img_path, data_polygt):
    img = Image.open(img_path)
    boxes = []
    ptss = []
    for i, lines in enumerate(data_polygt):
        points = lines["points"]
        word = lines["illegibility"]

        # x, y, w_1, h_1 = cv2.boundingRect(arr.astype('int32'))
        # box=[x,y,x+w_1,y,x+w_1,y+h_1,x,y+h_1]
        # box = np.asarray(box) / ([w * 1.0, h * 1.0] * 4)

        bbox = np.array(points).flatten()
        bbox = bbox.reshape(int(bbox.shape[0] / 2), 2)
        ptss.append(bbox)
        x, y, w, h = cv2.boundingRect(bbox)
        boxes.append([x, y, x + w, y + h])
    return img, boxes, ptss



if __name__ == '__main__':
    demo()