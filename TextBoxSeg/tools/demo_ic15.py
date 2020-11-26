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

def demo():
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    # output folder
    output_dir = 'demo/trash/IC15'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()

    #get img_patch from IC15
    if os.path.exists('/mnt/lustre/share_data/xieenze/xez_space/Text/ICDAR2015/'):
        ic15_root_path = '/mnt/lustre/share_data/xieenze/xez_space/Text/ICDAR2015/'
    else:
        ic15_root_path = '/mnt/lustre/share/xieenze/Text/ICDAR2015/'
    ic15_train_data = ic15_root_path + 'ch4_training_images'
    ic15_train_gt = ic15_root_path + 'ch4_training_localization_transcription_gt'
    assert os.path.exists(ic15_train_data) and os.path.exists(ic15_train_gt)


    patch_imgs = []
    for i in trange(1, 501):
        img_path = 'img_{}.jpg'.format(i)
        img_path = os.path.join(ic15_train_data, img_path)
        gt_path = 'gt_img_{}.txt'.format(i)
        gt_path = os.path.join(ic15_train_gt, gt_path)

        if os.path.exists(gt_path) and os.path.exists(img_path):
            img, boxes = parse_img_gt(img_path, gt_path)
            img = np.array(img)
            if boxes == []:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box
                patch = img[y1:y2 + 1, x1:x2 + 1]
                patch_imgs.append(Image.fromarray(patch))
                # 先只测500张
                if len(patch_imgs) > 500:
                    break
        else:
            print(img_path)
    print('total patch images:{}'.format(len(patch_imgs)))

    pool_imgs, pool_masks = [], []
    count = 0
    for image in patch_imgs:
        # image = Image.open(img_path).convert('RGB')
        resized_img = image.resize(cfg.TRAIN.BASE_SIZE)
        resized_img = transform(resized_img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = model(resized_img)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()

        img = np.array(image.resize(cfg.TRAIN.BASE_SIZE))
        mask = np.array(get_color_pallete(pred, cfg.DATASET.NAME))[:,:,None].repeat(3,-1) * 255
        if len(pool_imgs)<20:
            pool_imgs.append(img)
            pool_masks.append(mask)
        else:
            big_img = np.concatenate(pool_imgs, axis=0)
            big_mask = np.concatenate(pool_masks, axis=0)
            big_img_mask = Image.fromarray(np.concatenate([big_img, big_mask], axis=1))
            big_img_mask.save('{}/{}.png'.format(output_dir, count))
            print('{}/{}.png'.format(output_dir, count))
            count += 1
            pool_imgs, pool_masks = [], []


def parse_img_gt(img_path, gt_path):
    img = Image.open(img_path)
    with open(gt_path,'r') as f:
        data=f.readlines()
    boxes = []
    for d in data:
        d = d.replace('\n','').split(',')
        polygon = d[:8]; text = d[8]
        if "#" in text:
            continue #过滤掉ignore的
        polygon = [int(i.replace('\ufeff','')) for i in polygon]
        polygon_np = np.array(polygon).reshape([-1, 2])
        x, y, w, h = cv2.boundingRect(polygon_np)
        boxes.append([x,y,x+w,y+h])
    return img,boxes

if __name__ == '__main__':
    demo()