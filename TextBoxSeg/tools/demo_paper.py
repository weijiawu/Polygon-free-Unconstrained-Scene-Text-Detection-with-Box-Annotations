import cv2
import numpy as np
# def cvt2HeatmapImg(img):
#     img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
#     img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
#     return img
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
from IPython import embed
import numpy as np
from tqdm import trange
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
    output_dir = '/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/TT_attention'
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
    img_path = "/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/box_1.jpg"
    image = Image.open(img_path).convert('RGB')
    image_1 = image.copy
    gt_path_one = image
    origin_h, origin_w = image.height, image.width
    if origin_h>origin_w:
        image = image.transpose(Image.ROTATE_90)  # 将图片旋转90度
        # gt_path_one = RotateClockWise90(gt_path_one)

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

    # dis_ = gt_path_one.copy()
    # gt_path_one = np.array(get_color_pallete(gt_path_one, cfg.DATASET.NAME))[:, :, None].repeat(3, -1) * 255
    # gt_path_one = cv2.resize(gt_path_one, (128, 96))

    # dis_ = cv2.resize(dis_, (128, 96))
    # dist_img = cv2.distanceTransform(np.array(dis_), cv2.DIST_L1, cv2.DIST_MASK_3)
    # dist_back = cv2.distanceTransform((1 - np.array(dis_)), cv2.DIST_L1, cv2.DIST_MASK_3)
    # dist_img = (dist_img / dist_img.max() * 0.5 + 0.5)
    # dist_img = np.array(get_color_pallete(dist_img* 255, cfg.DATASET.NAME))[:, :, None].repeat(3, -1)
    #

    mask = np.array(get_color_pallete(pred, cfg.DATASET.NAME))[:,:,None].repeat(3,-1) * 255
    kernel = np.ones((6, 6), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # erosion = cv2.erode(img, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    if origin_h > origin_w:
        mask = Image.fromarray(mask).transpose(Image.ROTATE_270)  # 将图片旋转90度
    mask = mask.resize((origin_w, origin_h))


    mask.save(format("/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/result.png"))



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


