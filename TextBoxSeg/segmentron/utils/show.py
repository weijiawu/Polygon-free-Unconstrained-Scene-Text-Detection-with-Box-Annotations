import cv2
import numpy as np
import scipy.io
import random


def mask_image(image,mask_2d):
    h, w = mask_2d.shape

    # mask_3d = np.ones((h, w), dtype="uint8") * 255
    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    # mask_3d[mask_2d[:, :] == 1] = 0

    image.astype("uint8")
    # beijing = cv2.bitwise_and(image, image, mask=mask_3d)
    # R = random.randint(100,200)
    # G = random.randint(100,200)
    # B = random.randint(100,200)

    # mask_3d_color[mask_2d[:, :] == 1] = np.random.randint(50, 200, (1, 3), dtype=np.uint8)
    #
    # add_image = cv2.add(image, mask_3d_color)

    mask = (mask_2d!=0).astype(bool)

    mask_3d_color[mask_2d[:, :] == 1] = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
    image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5

    return image

def read_mat_lindes(p):
    f = scipy.io.loadmat(p)
    return f

def get_bboxes(gt_path):

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

        # box = []

        # for i in range(point_num):
        #     box.append(arr[i][0])
        #     box.append(arr[i][1])

        box = np.asarray(arr)
        bboxes.append(box)

    return bboxes
# gt_path = "/mnt/lustre/share_data/xieenze/wuweijia/Total_text/pseudolabel_attention/Train_bbox/poly_gt_img89.mat"
# image = cv2.imread("/mnt/lustre/share_data/xieenze/wuweijia/Total_text/Images/Train/img89.jpg")
# gt = get_bboxes(gt_path)
# for i,gt_one in enumerate(gt):
#     if i==1:
#         x_,y_ = 586, 713
#         # continue
#     cv2.drawContours(image, [gt_one], 0, (0, 0, 255), 15)
#
# gt_bpath = "/mnt/lustre/share_data/xieenze/wuweijia/Total_text/pseudolabel_attention/Train_pseudo/poly_gt_img89.mat"
#
#
# mask = cv2.imread("/mnt/lustre/xieenze/wuweijia/CVPR_SemiText/SemiText/TextBoxSeg/workdirs/show/resultd.png")
# patch = image[713:1700, 586:1089]
# mask = cv2.resize(mask,(patch.shape[1],patch.shape[0]))[:,:,0]
# try:
#     _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# except:
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# gt = get_bboxes(gt_bpath)
# for i,gt_one in enumerate(gt):
#     if i==1:
#         # continue
#         gt_one = contours[0]
#         print(gt_one.shape)
#         gt_one = gt_one[:, 0, :]
#
#         gt_one[:,0] += 586
#         gt_one[:, 1] += 713
#
#     mask_1 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#     cv2.fillPoly(mask_1, [gt_one], 1)
#     image = mask_image(image, mask_1)
#     cv2.drawContours(image, [gt_one], 0, (0, 255, 0), 10)
#
#
# cv2.imwrite("/mnt/lustre/xieenze/wuweijia/CVPR_SemiText/SemiText/TextBoxSeg/workdirs/show/show.jpg",image)
# mask_2d = cv2.imread("/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/resultd.png",cv2.IMREAD_GRAYSCALE)

# print(mask_2d.shape)

# cv2.imwrite("/home/xjc/Desktop/CVPR_SemiText/SemiText/TextBoxSeg/demo/show.jpg",add_image)
