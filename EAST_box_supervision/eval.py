import torch
from network.model import EAST
from network.loss import Loss
import numpy as np
from PIL import Image
from tqdm import tqdm
from lib.detect import detect
from evaluate.script import getresult
import argparse
import os
import cv2
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='EAST reimplementation')

# Model path
parser.add_argument('--resume', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/worksapce/ICDAR17/best_model_640aug.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--eval_path', default="/data/data_weijiawu/Sence_Text_detection/Paper-ACCV/DomainAdaptive/ICDAR2015/EAST_v2/ICDAR15/Test/image/", type=str,
                    help='the test image of target domain ')

parser.add_argument('--output_path', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/evaluate/15_submit/", type=str,
                    help='the predicted output of target domain')

parser.add_argument('--vis_path', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/worksapce/ICDAR17/show/", type=str,
                    help='the predicted output of target domain')
args = parser.parse_args()

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def test(model, input_path,output_path):
    model.eval()
    # model_box.eval()
    image_list = os.listdir(input_path)
    print("         ----------------------------------------------------------------")
    print("                    Starting Eval...")
    print("         ----------------------------------------------------------------")


	# getresult(output_path)
    # for size in range(640,2048,32):
    #     print("short_line:",size)
    for one_image in tqdm(image_list):
        image_path = os.path.join(input_path, one_image)
        img = Image.open(image_path).convert('RGB')
        orign_img = cv2.imread(image_path)
        filename, file_ext = os.path.splitext(os.path.basename(one_image))
        filename = filename.split("ts_")[-1]
        res_file = output_path + "res_" + filename + '.txt'
        vis_file = args.vis_path + filename + '.jpg'
        print(res_file)
        boxes = detect(img, model, device)


        with open(res_file, 'w') as f:
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32)
                points = np.reshape(poly, -1)
                # print(points[8])
                strResult = ','.join(
                    [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                     str(points[6]), str(points[7])]) + '\r\n'
                # strResult = ','.join(
                #     [str(points[1]), str(points[0]), str(points[3]), str(points[2]), str(points[5]), str(points[4]),
                #      str(points[7]), str(points[6]), str("1.0")]) + '\r\n'
                f.write(strResult)

            for bbox in boxes:
                # bbox = bbox / scale.repeat(int(len(bbox) / 2))
                bbox = np.array(bbox,np.int)
                cv2.drawContours(orign_img, [bbox[:8].reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 0, 255), 2)
        # cv2.imwrite(vis_file, orign_img)
    # f_score_new = getresult(output_path)


if __name__ == '__main__':

    device = torch.device("cuda")
    model = EAST()
    data_parallel = False
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     data_parallel = True
    model.to(device)

    print("loading pretrained model from ",args.resume)
    model.load_state_dict(torch.load(args.resume))
    # print(getresult(args.output_path))
    test(model,args.eval_path,args.output_path)







