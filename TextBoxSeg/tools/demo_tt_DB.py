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
    output_dir = 'demo/trash/Total_DB'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()

    #get img_patch from Total
    if os.path.exists('/data/data_weijiawu/Total_Text/DB/'):
        ic15_root_path = '/data/data_weijiawu/Total_Text/DB/'
    else:
        ic15_root_path = '/data/data_weijiawu/Total_Text/DB/'
    ic15_train_data = ic15_root_path + 'train_images'
    ic15_train_gt = ic15_root_path + 'train_gts'
    assert os.path.exists(ic15_train_data) and os.path.exists(ic15_train_gt)

    image_list = os.listdir(ic15_train_data)
    annotation_list = ['{}.txt'.format(img_name) for img_name in image_list]

    patch_imgs = []
    patch_gt = []
    for img_path,gt_path in zip(image_list, annotation_list):
        img_path = os.path.join(ic15_train_data, img_path)
        gt_path = os.path.join(ic15_train_gt, gt_path)

        if os.path.exists(gt_path) and os.path.exists(img_path):
            img, boxes, polygon_list = parse_img_gt(img_path, gt_path)
            img = np.array(img)
            if boxes == []:
                continue
            for bo_idx,box in enumerate(boxes):
                x1, y1, x2, y2 = box
                patch = img[y1:y2 + 1, x1:x2 + 1]
                if patch.shape[1] == 0 or patch.shape[0] == 0 :
                    continue

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


def parse_img_gt(img_path, txt_path):
    """
    .mat file parser
    :param mat_path: (str), mat file path
    :return: (list), TextInstance
    """
    img = Image.open(img_path)
    lines = read_lines(txt_path)
    boxes = []
    ptss = []
    for line in lines:
        line = remove_all(line, '\xef\xbb\xbf')
        line = remove_all(line, '\n')
        gt = line.split(',')
        oriented_box = [int(gt[i]) for i in range(len(gt)-1)]
        oriented_box = np.asarray(oriented_box)

        xs = oriented_box.reshape(int(len(oriented_box) / 2), 2)[:, 0]
        ys = oriented_box.reshape(int(len(oriented_box)/2), 2)[:, 1]

        if len(xs) < 4: # too few points
            continue

        pts = np.stack([xs, ys]).T.astype(np.int32)
        ptss.append(pts)
        x, y, w, h = cv2.boundingRect(pts)
        boxes.append([x, y, x + w, y + h])
    return img, boxes, ptss


if __name__ == '__main__':
    demo()