import torch
from network.model import EAST
from network.loss import Loss
import numpy as np
from PIL import Image
from tqdm import tqdm
from lib.detect import detect_msra
from evaluate.msra.eval import get_msra_result
import argparse
import os
import cv2
import shutil
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='EAST reimplementation')

# Model path
parser.add_argument('--exp_name',default= "MSRA", help='Where to store logs and models')
parser.add_argument('--resume', default="best_model_640aug.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--eval_path', default='/data/data_weijiawu/TD500/', type=str,
                    help='the test image of target domain ')
parser.add_argument('--workspace', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/worksapce/", type=str,
                    help='save model')
parser.add_argument('--vis', default=True, type=bool, help='visualization')
parser.add_argument('--is_test', default=False, type=bool, help='is test')
parser.add_argument('--vis_path', default="/home/wwj/workspace/Sence_Text_detection/AAAI_EAST/Baseline/EAST_v1/worksapce/MSRA/show/", type=str, help='visu')
parser.add_argument('--gt_name', default="msra_gt.zip", type=str, help='gt name')

args = parser.parse_args()

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

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
        box = list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8]))
        vertices.append(box)
        label = 0 if '###' in line else 1
        labels.append(label)
    return np.array(vertices)

def test(model,args,ther_):
    model.eval()
    output_path = os.path.join(args.workspace, "submit_test")
    os.makedirs(output_path, exist_ok=True)
    input_path = os.path.join(args.eval_path, "Test_image")
    image_list = os.listdir(input_path)

    gt_paths = os.path.join(args.eval_path,"to_icdar15","test_label")

    print("     ----------------------------------------------------------------")
    print("                           Starting Eval...")
    print("     ----------------------------------------------------------------")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for one_image in tqdm(image_list):
        image_path = os.path.join(input_path, one_image)

        img = Image.open(image_path).convert('RGB')
        orign_img = cv2.imread(image_path)
        filename, file_ext = os.path.splitext(os.path.basename(one_image))
        res_file = output_path + "/res_" + filename + '.txt'

        gt_path = os.path.join(gt_paths, "gt_" + filename + '.txt')
        with open(gt_path,"r") as f:
            lines = f.readlines()
        gt_line = extract_vertices(lines)


        vis_file = args.vis_path + filename + '.jpg'
        boxes = detect_msra(img, model, device,short_line = ther_)

        with open(res_file, 'w') as f:
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32)
                points = np.reshape(poly, -1)

                strResult = ','.join(
                        [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                         str(points[6]), str(points[7])]) + '\r\n'

                f.write(strResult)
            if args.vis:
                for bbox in boxes:
                    # bbox = bbox / scale.repeat(int(len(bbox) / 2))
                    bbox = np.array(bbox,np.int)
                    cv2.drawContours(orign_img, [bbox[:8].reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 0, 255), 2)

                for bbox in gt_line:
                    # bbox = bbox / scale.repeat(int(len(bbox) / 2))
                    bbox = np.array(bbox,np.int)
                    cv2.drawContours(orign_img, [bbox[:8].reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 255, 0), 2)
                cv2.imwrite(vis_file, orign_img)
    if not args.is_test:
        f_score_new = get_msra_result(output_path,"/data/data_weijiawu/TD500/Test_gt/")


if __name__ == '__main__':

    device = torch.device("cuda")
    model = EAST()
    data_parallel = False
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     data_parallel = True
    model.to(device)
    args.workspace = os.path.join(args.workspace, args.exp_name)
    args.resume = os.path.join(args.workspace,args.resume)
    print("loading pretrained model from ",args.resume)

    model.load_state_dict(torch.load(args.resume))
    for ther in range(416,2000,32):
        ther_ = 448
        print("threshold:",ther_)
        test(model,args,ther_)







