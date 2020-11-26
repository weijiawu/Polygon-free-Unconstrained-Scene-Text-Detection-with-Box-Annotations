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

# debug-------------------------
from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image
import math
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
import importlib
import matplotlib.pyplot as plt



debug_flag = False
if debug_flag:
    def cal_distance(x1, y1, x2, y2):
        '''calculate the Euclidean distance'''
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    def move_points(vertices, index1, index2, r, coef):
        '''move the two points to shrink edge
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
            index1  : offset of point1
            index2  : offset of point2
            r       : [r1, r2, r3, r4] in paper
            coef    : shrink ratio in paper
        Output:
            vertices: vertices where one edge has been shinked
        '''
        index1 = index1 % 4
        index2 = index2 % 4
        x1_index = index1 * 2 + 0
        y1_index = index1 * 2 + 1
        x2_index = index2 * 2 + 0
        y2_index = index2 * 2 + 1

        r1 = r[index1]
        r2 = r[index2]
        length_x = vertices[x1_index] - vertices[x2_index]
        length_y = vertices[y1_index] - vertices[y2_index]
        length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
        if length > 1:
            ratio = (r1 * coef) / length
            vertices[x1_index] += ratio * (-length_x)
            vertices[y1_index] += ratio * (-length_y)
            ratio = (r2 * coef) / length
            vertices[x2_index] += ratio * length_x
            vertices[y2_index] += ratio * length_y
        return vertices
    def shrink_poly(vertices, coef=0.3):
        '''shrink the text region
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
            coef    : shrink ratio in paper
        Output:
            v       : vertices of shrinked text region <numpy.ndarray, (8,)>
        '''
        x1, y1, x2, y2, x3, y3, x4, y4 = vertices
        r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
        r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
        r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
        r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
        r = [r1, r2, r3, r4]

        # obtain offset to perform move_points() automatically
        if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > \
                cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4):
            offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
        else:
            offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

        v = vertices.copy()
        v = move_points(v, 0 + offset, 1 + offset, r, coef)
        v = move_points(v, 2 + offset, 3 + offset, r, coef)
        v = move_points(v, 1 + offset, 2 + offset, r, coef)
        v = move_points(v, 3 + offset, 4 + offset, r, coef)
        return v
    def get_rotate_mat(theta):
        '''positive theta value means rotate clockwise'''
        return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    def rotate_vertices(vertices, theta, anchor=None):
        '''rotate vertices around anchor
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
            theta   : angle in radian measure
            anchor  : fixed position during rotation
        Output:
            rotated vertices <numpy.ndarray, (8,)>
        '''
        v = vertices.reshape((4, 2)).T
        if anchor is None:
            anchor = v[:, :1]
        rotate_mat = get_rotate_mat(theta)
        res = np.dot(rotate_mat, v - anchor)
        return (res + anchor).T.reshape(-1)
    def get_boundary(vertices):
        '''get the tight boundary around given vertices
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
        Output:
            the boundary
        '''
        x1, y1, x2, y2, x3, y3, x4, y4 = vertices
        x_min = min(x1, x2, x3, x4)
        x_max = max(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        y_max = max(y1, y2, y3, y4)
        return x_min, x_max, y_min, y_max
    def cal_error(vertices):
        '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
        calculate the difference between the vertices orientation and default orientation
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
        Output:
            err     : difference measure
        '''
        x_min, x_max, y_min, y_max = get_boundary(vertices)
        x1, y1, x2, y2, x3, y3, x4, y4 = vertices
        err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
              cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
        return err
    def find_min_rect_angle(vertices):
        '''find the best angle to rotate poly and obtain min rectangle
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
        Output:
            the best angle <radian measure>
        '''
        angle_interval = 1
        angle_list = list(range(-90, 90, angle_interval))
        area_list = []
        for theta in angle_list:
            rotated = rotate_vertices(vertices, theta / 180 * math.pi)
            x1, y1, x2, y2, x3, y3, x4, y4 = rotated
            temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                        (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
            area_list.append(temp_area)

        sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
        min_error = float('inf')
        best_index = -1
        rank_num = 10
        # find the best angle with correct orientation
        for index in sorted_area_index[:rank_num]:
            rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
            temp_error = cal_error(rotated)
            if temp_error < min_error:
                min_error = temp_error
                best_index = index
        return angle_list[best_index] / 180 * math.pi
    def is_cross_text(start_loc, length, vertices):
        '''check if the crop image crosses text regions
        Input:
            start_loc: left-top position
            length   : length of crop image
            vertices : vertices of text regions <numpy.ndarray, (n,8)>
        Output:
            True if crop image crosses text region
        '''
        if vertices.size == 0:
            return False
        start_w, start_h = start_loc
        a = np.array([start_w, start_h, start_w + length, start_h, \
                      start_w + length, start_h + length, start_w, start_h + length]).reshape((4, 2))
        p1 = Polygon(a).convex_hull
        for vertice in vertices:
            p2 = Polygon(vertice.reshape((4, 2))).convex_hull
            inter = p1.intersection(p2).area
            if 0.01 <= inter / p2.area <= 0.99:
                return True
        return False
    def crop_img(img, vertices, labels, length):
        '''crop img patches to obtain batch and augment
        Input:
            img         : PIL Image
            vertices    : vertices of text regions <numpy.ndarray, (n,8)>
            labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
            length      : length of cropped image region
        Output:
            region      : cropped image region
            new_vertices: new vertices in cropped region
        '''
        h, w = img.height, img.width
        # confirm the shortest side of image >= length
        if h >= w and w < length:
            img = img.resize((length, int(h * length / w)), Image.BILINEAR)
        elif h < w and h < length:
            img = img.resize((int(w * length / h), length), Image.BILINEAR)
        ratio_w = img.width / w
        ratio_h = img.height / h
        assert (ratio_w >= 1 and ratio_h >= 1)

        new_vertices = np.zeros(vertices.shape)
        if vertices.size > 0:
            new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
            new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

        # find random position
        remain_h = img.height - length
        remain_w = img.width - length
        flag = True
        cnt = 0
        while flag and cnt < 1000:
            cnt += 1
            start_w = int(np.random.rand() * remain_w)
            start_h = int(np.random.rand() * remain_h)
            flag = is_cross_text([start_w, start_h], length, new_vertices[labels == 1, :])
        box = (start_w, start_h, start_w + length, start_h + length)
        region = img.crop(box)
        if new_vertices.size == 0:
            return region, new_vertices

        new_vertices[:, [0, 2, 4, 6]] -= start_w
        new_vertices[:, [1, 3, 5, 7]] -= start_h
        return region, new_vertices
    def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
        '''get rotated locations of all pixels for next stages
        Input:
            rotate_mat: rotatation matrix
            anchor_x  : fixed x position
            anchor_y  : fixed y position
            length    : length of image
        Output:
            rotated_x : rotated x positions <numpy.ndarray, (length,length)>
            rotated_y : rotated y positions <numpy.ndarray, (length,length)>
        '''
        x = np.arange(length)
        y = np.arange(length)
        x, y = np.meshgrid(x, y)
        x_lin = x.reshape((1, x.size))
        y_lin = y.reshape((1, x.size))
        coord_mat = np.concatenate((x_lin, y_lin), 0)
        rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                        np.array([[anchor_x], [anchor_y]])
        rotated_x = rotated_coord[0, :].reshape(x.shape)
        rotated_y = rotated_coord[1, :].reshape(y.shape)
        return rotated_x, rotated_y
    def adjust_height(img, vertices, ratio=0.2):
        '''adjust height of image to aug data
        Input:
            img         : PIL Image
            vertices    : vertices of text regions <numpy.ndarray, (n,8)>
            ratio       : height changes in [0.8, 1.2]
        Output:
            img         : adjusted PIL Image
            new_vertices: adjusted vertices
        '''
        ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
        old_h = img.height
        new_h = int(np.around(old_h * ratio_h))
        img = img.resize((img.width, new_h), Image.BILINEAR)

        new_vertices = vertices.copy()
        if vertices.size > 0:
            new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
        return img, new_vertices
    def rotate_img(img, vertices, angle_range=10):
        '''rotate image [-10, 10] degree to aug data
        Input:
            img         : PIL Image
            vertices    : vertices of text regions <numpy.ndarray, (n,8)>
            angle_range : rotate range
        Output:
            img         : rotated PIL Image
            new_vertices: rotated vertices
        '''
        center_x = (img.width - 1) / 2
        center_y = (img.height - 1) / 2
        angle = angle_range * (np.random.rand() * 2 - 1)
        img = img.rotate(angle, Image.BILINEAR)
        new_vertices = np.zeros(vertices.shape)
        for i, vertice in enumerate(vertices):
            new_vertices[i, :] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
        return img, new_vertices
    def get_score_geo(img, vertices, labels, scale, length):
        '''generate score gt and geometry gt
        Input:
            img     : PIL Image
            vertices: vertices of text regions <numpy.ndarray, (n,8)>
            labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
            scale   : feature map / image
            length  : image length
        Output:
            score gt, geo gt, ignored
        '''
        score_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
        geo_map = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
        ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)

        index = np.arange(0, length, int(1 / scale))
        index_x, index_y = np.meshgrid(index, index)
        ignored_polys = []
        polys = []

        for i, vertice in enumerate(vertices):
            if labels[i] == 0:
                ignored_polys.append(np.around(scale * vertice.reshape((4, 2))).astype(np.int32))
                continue

            poly = np.around(scale * shrink_poly(vertice).reshape((4, 2))).astype(np.int32)  # scaled & shrinked
            polys.append(poly)
            temp_mask = np.zeros(score_map.shape[:-1], np.float32)
            cv2.fillPoly(temp_mask, [poly], 1)

            theta = find_min_rect_angle(vertice)
            rotate_mat = get_rotate_mat(theta)

            rotated_vertices = rotate_vertices(vertice, theta)
            x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
            rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)

            d1 = rotated_y - y_min
            d1[d1 < 0] = 0
            d2 = y_max - rotated_y
            d2[d2 < 0] = 0
            d3 = rotated_x - x_min
            d3[d3 < 0] = 0
            d4 = x_max - rotated_x
            d4[d4 < 0] = 0
            geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
            geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
            geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
            geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
            geo_map[:, :, 4] += theta * temp_mask

        cv2.fillPoly(ignored_map, ignored_polys, 1)
        cv2.fillPoly(score_map, polys, 1)
        return torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(geo_map).permute(2, 0, 1), torch.Tensor(
            ignored_map).permute(2, 0, 1)
    def extract_vertices(lines):
        '''extract vertices info from txt lines
        Input:
            lines   : list of string info
        Output:
            vertices: vertices of text regions <numpy.ndarray, (n,8)>
            labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        '''
        labels = []
        vertices = []
        for line in lines:
            vertices.append(list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
            label = 0 if '###' in line else 1
            labels.append(label)
        return np.array(vertices), np.array(labels)


    theta1s = []
    theta2s = []
    delta_theat = []
    # debug-------------------------

def demo():
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()

    #get img_patch from IC15
    ic15_root_path = '/mnt/lustre/share_data/xieenze/xez_space/Text/ICDAR2015/'
    ic15_train_data = ic15_root_path + 'ch4_training_images'
    ic15_train_gt = ic15_root_path + 'ch4_training_localization_transcription_gt'
    '''bbox + pseudo box '''
    ic15_train_gt_bbox = ic15_root_path + 'ch4_training_localization_transcription_gt_bbox'
    ic15_train_gt_pseudobox = ic15_root_path + 'ch4_training_localization_transcription_gt_pseudo'
    os.system('rm -rf {}/*txt'.format(ic15_train_gt_bbox))#; os.mkdir(ic15_train_gt_bbox)
    os.system('rm -rf {}/*txt'.format(ic15_train_gt_pseudobox))#; os.mkdir(ic15_train_gt_pseudobox)

    num_ic15_imgs = 1000
    #遍历图片
    for i in trange(1, num_ic15_imgs+1):
        if debug_flag:
            #debug-----------
            if len(theta2s) > 500:
                break
            # debug-----------
        img_path = 'img_{}.jpg'.format(i)
        img_path = os.path.join(ic15_train_data, img_path)
        gt_path = 'gt_img_{}.txt'.format(i)
        gt_path = os.path.join(ic15_train_gt, gt_path)

        if os.path.exists(gt_path) and os.path.exists(img_path):
            img, boxes, ori_box = parse_img_gt(img_path, gt_path)
            img = np.array(img)
            #遍历box
            f_bbox = open(os.path.join(ic15_train_gt_bbox,'gt_img_{}.txt'.format(i)),'w')
            f_rbox = open(os.path.join(ic15_train_gt_pseudobox, 'gt_img_{}.txt'.format(i)), 'w')
            seq_bbox, seq_rbox = [],[]

            for j, box in enumerate(boxes):
                mask = np.zeros_like(img)[:, :, 0]
                x1, y1, x2, y2, is_ignore = box
                patch = img[y1:y2 + 1, x1:x2 + 1]
                patch = Image.fromarray(patch)
                pred_gt = inference(model, patch, transform)
                mask[y1:y2 + 1, x1:x2 + 1] = pred_gt
                #get bbox rbox
                _, rbox = get_pseudo_pabel(mask)
                bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype='int32')
                if debug_flag:
                    if is_ignore == 1:
                        cv2.drawContours(img, [rbox], 0, (0, 0, 255), 1)
                        cv2.drawContours(img, [bbox], 0, (0, 255, 0), 1)
                        cv2.drawContours(img, [ori_box[j]], 0, (255, 0, 0), 1)
                    else:
                        cv2.drawContours(img, [rbox], 0, (0, 255, 255), 1)
                        cv2.drawContours(img, [bbox], 0, (255, 255, 0), 1)
                        cv2.drawContours(img, [ori_box[j]], 0, (255, 0, 255), 1)
                bbox, rbox = list(bbox.reshape(-1)), list(rbox.reshape(-1))
                #起始点和IC15的需要对齐，平移一下
                # embed()
                bbox = adjust_box_sort(bbox)
                rbox = adjust_box_sort(rbox)

                if debug_flag:
                    #debug-------------------------
                    if is_ignore == 1:
                        pes_r_box = np.array(rbox)[None,...]
                        ori_r_box = ori_box[j].reshape(-1)
                        theta1 = find_min_rect_angle(pes_r_box) / math.pi * 180
                        theta2 = find_min_rect_angle(ori_r_box) / math.pi * 180
                        theta1s.append(theta1)
                        theta2s.append(theta2)
                        if abs(theta2-theta1)>25:
                            print(i)
                        delta_theat.append(theta1-theta2)
                    # debug-------------------------

                if is_ignore == 1:
                    seq_bbox.append(",".join([str(int(i)) for i in bbox])+',aaa,\n')
                    seq_rbox.append(",".join([str(int(i)) for i in rbox])+',aaa,\n')
                else:
                    seq_bbox.append(",".join([str(int(i)) for i in bbox]) + ',###,\n')
                    seq_rbox.append(",".join([str(int(i)) for i in rbox]) + ',###,\n')
            f_bbox.writelines(seq_bbox)
            f_rbox.writelines(seq_rbox)
            f_bbox.close()
            f_rbox.close()

            if debug_flag:
                print('debug vis')
                cv2.imwrite('trash/img{}.png'.format(i), img[:,:,[2,1,0]])
        else:
            print(img_path)

        if debug_flag:
            # debug-------------------------
            x = [i for i in range(len(theta1s))]
            plt.rcParams['figure.figsize'] = (10.0, 4.0)
            plt.rcParams['savefig.dpi'] = 300  # 图片像素
            plt.rcParams['figure.dpi'] = 300  # 分辨
            # plt.plot(x, theta1s)
            # plt.plot(x, theta2s)
            plt.plot(x, delta_theat)
            plt.title('line chart')
            plt.xlabel('x')
            plt.ylabel('theta')
            plt.savefig('trash/theta.png')
            # debug-------------------------

def adjust_box_sort(box):
    start = -1
    _box = list(np.array(box).reshape(-1,2))
    min_x = min(box[0::2])
    min_y = min(box[1::2])
    _box.sort(key=lambda x:(x[0]-min_x)**2+(x[1]-min_y)**2)
    start_point = list(_box[0])
    for i in range(0,8,2):
        x,y = box[i],box[i+1]
        if [x,y] == start_point:
            start = i//2
            break

    new_box = []
    new_box.extend(box[start*2:])
    new_box.extend(box[:start*2])
    return new_box

def get_pseudo_pabel(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda x:cv2.contourArea(x), reverse=True)
    cnt = contours[0]
    #bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    bbox = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]],dtype='int32')
    #rotate box
    rect = cv2.minAreaRect(cnt)
    rbox = cv2.boxPoints(rect).astype('int32')
    return bbox, rbox

def inference(model, image, transform):
    resized_img = image.resize(cfg.TRAIN.BASE_SIZE)
    resized_img = transform(resized_img).unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(resized_img)
    pred = Image.fromarray(torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy().astype('int32')).resize(image.size)
    pred = np.array(pred)
    return pred

def parse_img_gt(img_path, gt_path):
    img = Image.open(img_path)
    with open(gt_path,'r') as f:
        data=f.readlines()
    boxes = []
    ori_boxes = []
    for d in data:
        d = d.replace('\n','').split(',')
        polygon = d[:8]; text = d[8]
        polygon = [int(i.replace('\ufeff','')) for i in polygon]
        polygon_np = np.array(polygon).reshape([-1, 2])
        x, y, w, h = cv2.boundingRect(polygon_np)
        if "##" in text:
            boxes.append([x, y, x + w, y + h, 0]) #0=ignore
            ori_boxes.append(polygon_np)
        else:
            boxes.append([x, y, x + w, y + h, 1]) #1=real
            ori_boxes.append(polygon_np)
    return img, boxes, ori_boxes

if __name__ == '__main__':
    demo()