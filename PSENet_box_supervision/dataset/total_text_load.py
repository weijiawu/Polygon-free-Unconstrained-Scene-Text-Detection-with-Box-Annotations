import numpy as np
from PIL import Image
from torch.utils import data
from utils.utils import get_absolute_path,ls
import cv2
import os
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import scipy.io
from dataset.total_aug import DataAugment
from utils.utils import draw_bbox
from IPython import embed

data_aug = DataAugment()

random.seed(123456)


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img



def read_mat_lindes(p):
    p = get_absolute_path(p)
    f = scipy.io.loadmat(p)
    return f


def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    bboxes = []
    tags = []
    point_nums = []
    data = read_mat_lindes(gt_path)
    data_polygt = data['polygt']
    # for lines in data_polygt:
    for i, lines in enumerate(data_polygt):
        X = np.array(lines[1])
        Y = np.array(lines[3])

        point_num = len(X[0])
        point_nums.append(point_num)
        word = np.array(lines[4])
        if len(word) == 0:
            word = '?'
        else:
            word = str(word[0].encode("utf-8"))

        if "#" in word:
            tags.append(True)
        else:
            tags.append(False)
        arr = np.concatenate([X, Y]).T

        # x, y, w_1, h_1 = cv2.boundingRect(arr.astype('int32'))
        # box=[x,y,x+w_1,y,x+w_1,y+h_1,x,y+h_1]
        # box = np.asarray(box) / ([w * 1.0, h * 1.0] * 4)

        box = []

        for i in range(point_num):
            box.append(arr[i][0])
            box.append(arr[i][1])

        box = np.asarray(box) / ([w * 1.0, h * 1.0] * point_num)
        bboxes.append(box)

    return bboxes, tags


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=640):
    h, w = img.shape[0:2]

    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    # img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    img = scale_aligned(img, scale)
    return img


def random_crop_padding(imgs, target_size):
    """ using padding and the final crop size is (800, 800) """
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

        shrinked_bbox = np.array(pco.Execute(-offset))

        if len(shrinked_bbox.shape) != 3 or shrinked_bbox.shape[1] <= 2:
            shrinked_bboxes.append(bbox)
        else:
            shrinked_bboxes.append(shrinked_bbox[0])

    return np.array(shrinked_bboxes)

def check_and_validate_polys(polys, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    # polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
    # polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

    validated_polys = []
    for poly in polys:
        poly = np.array(poly, dtype=np.int)
        p_area = cv2.contourArea(poly)
        if abs(p_area) < 1:
            continue
        validated_polys.append(poly)
    return np.array(validated_polys)


def generate_rbox(im_size, text_polys, text_tags, training_mask, i, n, m):
    """
    生成mask图，白色部分是文本，黑色是背景
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param text_tags: 标注文本框是否参与训练
    :return: 生成的mask图
    """

    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    for poly, tag in zip(text_polys, text_tags):

        poly = np.array(poly,dtype=np.int)
        # shrink rate
        r_i = 1 - (1 - m) * (n - i) / (n - 1)
        d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        # pco.AddPath(pyclipper.scale_to_clipper(poly), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # shrinked_poly = np.floor(np.array(pyclipper.scale_from_clipper(pco.Execute(-d_i)))).astype(np.int)
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))

        for shrinked_mark in shrinked_poly:
            cv2.fillPoly(score_map, np.array([shrinked_mark]), 1)
            if tag:
                cv2.fillPoly(training_mask, np.array([shrinked_mark]), 0)

    return score_map, training_mask



def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)

        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs

def resize(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs

    # return i, j, th, tw
    for idx in range(len(imgs)):
        imgs[idx] = cv2.resize(imgs[idx],img_size)
    return imgs

def image_label(im, text_polys: np.ndarray, text_tags: list, n: int, m: float, input_size: int) :
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''
    im = random_scale(im)
    h, w, _ = im.shape
    for i in range(len(text_polys)):
        text_polys[i] = np.reshape(text_polys[i] * ([im.shape[1], im.shape[0]] * int((text_polys[i].shape[0] / 2))),
                               (int(text_polys[i].shape[0] / 2), 2)).astype('int32')
    # 检查越界
    text_polys = check_and_validate_polys(np.array(text_polys), (h, w))
    short_edge = min(h, w)

    if short_edge < input_size:
        # 保证短边 >= inputsize
        scale = input_size / short_edge
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys = [np.array(one_text,dtype=np.float) * scale for one_text in text_polys]

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    score_maps = []
    for i in range(1, n + 1):
        # s1->sn,由小到大
        score_map, training_mask = generate_rbox((h, w), text_polys, text_tags, training_mask, i, n, m)
        score_maps.append(score_map)
    score_maps = np.array(score_maps, dtype=np.float32)


    imgs = [im, training_mask]
    imgs.extend(score_maps)
    imgs = random_horizontal_flip(imgs)
    imgs = random_rotate(imgs)
    imgs = random_crop(imgs, (input_size, input_size))
    img, training_mask, score_maps = imgs[0], imgs[1],imgs[2:]

    #     imgs = [im, training_mask]
    #     imgs.extend(score_maps)
    #     imgs = resize(imgs, (input_size, input_size))
    #     img, training_mask, score_maps = imgs[0], imgs[1], imgs[2:]

    return img, np.array(score_maps), training_mask

class TotalTextoader(data.Dataset):
    def __init__(self,
                 train_data_dir,
                 train_gt_dir,
                 test_data_dir,
                 test_gt_dir,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 kernel_num=7,
                 min_scale=0.4):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size
        self.kernel_num = kernel_num
        self.min_scale = min_scale

        total_text_train_data_dir = train_data_dir
        total_text_train_gt_dir = train_gt_dir
        total_text_test_data_dir = test_data_dir
        total_text_test_gt_dir = test_gt_dir

        if split == 'train':
            data_dirs = [total_text_train_data_dir]
            gt_dirs = [total_text_train_gt_dir]
        else:
            data_dirs = [total_text_test_data_dir]
            gt_dirs = [total_text_test_gt_dir]

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = ls(data_dir, '.jpg')
            img_names.extend(ls(data_dir, '.png'))

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = 'poly_gt_' + img_name.split('.')[0] + '.mat'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        bboxes, tags = get_bboxes(img, gt_path)

        img, gt_kernals, training_mask = image_label(img, bboxes, tags, input_size=self.img_size,
                                                     n=self.kernel_num,
                                                     m=self.min_scale)
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_kernals = torch.from_numpy(gt_kernals).float()
        training_mask = torch.from_numpy(training_mask).float()
        return img, gt_kernals, training_mask



if __name__ == '__main__':
    import torch
    import config.totaltext.psenet_pseudo as config
    from utils.utils import show_img
    from tqdm import tqdm, trange
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms

    train_data = TotalTextoader(config.train_data_dir,
                                config.train_gt_dir,
                                config.test_data_dir,
                                config.test_gt_dir,
                                split='train', is_transform=True, img_size=config.data_shape,
                                kernel_num=config.kernel_num, min_scale=config.min_scale)
    for i in range(len(train_data)):
        img, kernel, mask = train_data[9]
    # for i, (img, label, mask) in enumerate(train_data):

        print(i)
        print(kernel.shape)
        print(img.shape)
        print(kernel[0][-1].sum())
        print(mask[0].sum())

        show_img(((img.to(torch.float)).numpy().transpose(1, 2, 0)*[0.229, 0.224, 0.225]+[0.485, 0.456, 0.406]), color=True)
        show_img(kernel)
        show_img(mask)
        plt.show()
        break

    # for i in trange(len(train_data)):
    #     img, kernel, mask = train_data[i]
    #     img = img.permute(1,2,0)
    #     img = img-img.min()
    #     img = img/img.max()*255
    #     img = img.data.cpu().numpy()
    #     img = img.astype('uint8')
    #     i=0
    #     cv2.imwrite('trash/img{}.png'.format(i), img)
    #     for idx, k in enumerate(kernel):
    #         cv2.imwrite('trash/kernel{}_{}.png'.format(i, idx), k.data.cpu().numpy()*255)
    #     cv2.imwrite('trash/mask{}.png'.format(i), mask.data.cpu().numpy()*255)
    #
    #     embed()

