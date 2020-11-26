import cv2
import os
# import utils.config_totaltext as config
import shutil
import glob
import time
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torch.utils.data as Data
from torchvision import transforms
import torchvision.utils as vutils
from utils.utils import write_result_as_txt,debug,load_checkpoint, save_checkpoint, setup_logger
from dataset.total_text_load import TotalTextoader
from models import PSENet
from models.loss import PSELoss
from evaluation.script import getresult
from torch.autograd import Variable
from utils.utils import AverageMeter
from pse import decode_total as pse_decode
from evaluation.total_text.eval_total import evl_totaltext
import argparse
from test_totaltext import eval
from mmcv import Config
from IPython import embed


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(torch.sum(gt_text > 0.5)) - (int)(torch.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_num = (int)(torch.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted, _ = torch.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = torch.cat(selected_masks, 0).float()
    return selected_masks

def train_epoch(net, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,  logger):
    net.train()
    train_loss = 0.
    start = time.time()
    scheduler.step()
    lr = scheduler.get_lr()[0]
    for idx, (images, labels, training_mask) in enumerate(train_loader):
        cur_batch = images.size()[0]
        images, labels, training_mask = images.to(device), labels.to(device), training_mask.to(device)

        outputs = net(images)

        texts = outputs[:, -1, :, :]
        kernels = outputs[:, :-1, :, :]
        gt_texts = labels[:, -1, :, :]
        gt_kernels = labels[:, :-1, :, :]

        selected_masks = ohem_batch(texts, gt_texts, training_mask)
        loss_text = criterion(texts, gt_texts, selected_masks)
        loss_kernels = []
        mask0 = torch.sigmoid(texts)
        mask1 = training_mask
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).float()

        for i in range(6):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = sum(loss_kernels) / len(loss_kernels)

        loss = 0.7 * loss_text + 0.3 * loss_kernel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loss = loss.item()
        cur_step = epoch * all_step + idx

        if idx % config.display_interval == 0:
            batch_time = time.time() - start
            logger.info(
                '[{}/{}], [{}/{}], step: {}, {:.3f} samples/s, batch_loss: {:.4f}, time:{:.4f}, lr:{}'.format(
                    epoch, config.epochs, idx, all_step, cur_step, config.display_interval * cur_batch / batch_time,
                    loss,  batch_time, lr))
            start = time.time()

    return train_loss / all_step, lr

def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def dice_loss(input, target, mask):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    mask = mask.contiguous().view(mask.size()[0], -1)

    input = input * mask
    target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    _dice_loss = torch.mean(d)
    return 1 - _dice_loss

def main():
    config.workspace = os.path.join(config.workspace_dir, config.exp_name)
    if not os.path.exists(config.workspace):
        os.makedirs(config.workspace)

    logger = setup_logger(os.path.join(config.workspace, 'train_log'))
    logger.info(config.pprint())

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    torch.backends.cudnn.benchmark = True
    logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子

    train_data = TotalTextoader(config.train_data_dir,
                                config.train_gt_dir,
                                config.test_data_dir,
                                config.test_gt_dir,
                                split='train',
                                is_transform=True,
                                img_size=config.data_shape,
                                kernel_num=config.kernel_num,
                                min_scale=config.min_scale)
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=config.train_batch_size,
                                   shuffle=True,
                                   num_workers=int(config.workers))

    model = PSENet(backbone=config.backbone,
                   pretrained=config.pretrained,
                   result_num=config.kernel_num,
                   scale=config.scale)
    if not config.pretrained and not config.restart_training:
        model.apply(weights_init)

    num_gpus = torch.cuda.device_count()
    # if num_gpus > 1:
    model = nn.DataParallel(model)
    model = model.to(device)

    criterion = dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.checkpoint != '' and config.restart_training == True:
        start_epoch = load_checkpoint(config.checkpoint, model, logger, device, optimizer)
        start_epoch += 1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         config.lr_decay_step,
                                                         gamma=config.lr_gamma,
                                                         last_epoch=start_epoch)
        logger.info('resume from {}, epoch={}'.format(config.checkpoint,start_epoch))
    else:
        start_epoch = 1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         config.lr_decay_step,
                                                         gamma=config.lr_gamma)

    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} iters in dataloader'.format(train_data.__len__(), all_step))
    for epoch in range(start_epoch, config.epochs+1):
        start = time.time()
        train_loss, lr = train_epoch(model, optimizer, scheduler, train_loader,
                                     device, criterion, epoch, all_step, logger)
        logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            epoch, config.epochs, train_loss, time.time() - start, lr))

        if epoch % config.save_interval == 0:
            save_path = '{}/epoch_{}.pth'.format(config.workspace, epoch)
            latest_path = '{}/latest.pth'.format(config.workspace)
            save_checkpoint(save_path, model, optimizer, epoch, logger)
            save_checkpoint(latest_path, model, optimizer, epoch, logger)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default="./config/totaltext/psenet_pseudo.py", help='')
    parser.add_argument('--resume_from', '-r', type=str, default="/home/xjc/Desktop/CVPR_SemiText/SemiText/PSENet/workdirs/Synthtext/7kernal_epoch_1.0_model.pth", help='')

    args = parser.parse_args()
    config = Config.fromfile(args.config)

    if args.resume_from != "":
        config.checkpoint = args.resume_from
        config.restart_training = True


    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    main()
