import cv2
import os
from mmcv import Config
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
# from torch.utils.tensorboard import SummaryWriter
import argparse
from dataset.icdar15_load import ICDAR15
from models import PSENet
from models.loss import PSELoss
from utils.utils import load_checkpoint, save_checkpoint, setup_logger
from pse import decode as pse_decode
from evaluation.script import getresult


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < config.warm_up_epoch:
        lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
    else:
        lr = config.lr * (config.lr_gamma ** (epoch / config.lr_decay_step[0]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_epoch(net, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,  logger):
    net.train()
    train_loss = 0.
    start = time.time()
    scheduler.step()
    # lr = adjust_learning_rate(optimizer, epoch)
    lr = scheduler.get_lr()[0]
    for i, (images, labels, training_mask) in enumerate(train_loader):
        cur_batch = images.size()[0]
        images, labels, training_mask = images.to(device), labels.to(device), training_mask.to(device)
        # Forward
        y1 = net(images)
        loss_c, loss_s, loss = criterion(y1, labels, training_mask)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loss_c = loss_c.item()
        loss_s = loss_s.item()
        loss = loss.item()
        cur_step = epoch * all_step + i

        if i % config.display_interval == 0:
            batch_time = time.time() - start
            logger.info(
                '[{}/{}], [{}/{}], step: {}, {:.3f} samples/sec, batch_loss: {:.4f}, batch_loss_c: {:.4f}, batch_loss_s: {:.4f}, time:{:.4f}, lr:{}'.format(
                    epoch, config.epochs, i, all_step, cur_step, config.display_interval * cur_batch / batch_time,
                    loss, loss_c, loss_s, batch_time, lr))
            start = time.time()

        if i % config.show_images_interval == 0:
            if config.display_input_images:
                # show images on tensorboard
                x = vutils.make_grid(images.detach().cpu(), nrow=4, normalize=True, scale_each=True, padding=20)
                # writer.add_image(tag='input/image', img_tensor=x, global_step=cur_step)

                show_label = labels.detach().cpu()
                b, c, h, w = show_label.size()
                show_label = show_label.reshape(b * c, h, w)
                show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=config.n, normalize=False, padding=20,
                                              pad_value=1)
                # writer.add_image(tag='input/label', img_tensor=show_label, global_step=cur_step)

            if config.display_output_images:
                y1 = torch.sigmoid(y1)
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=False, padding=20, pad_value=1)

    return train_loss / all_step, lr


def main():

    config.workspace = os.path.join(config.workspace_dir,config.exp_name)
    # if config.restart_training:
    #     shutil.rmtree(config.workspace, ignore_errors=True)
    if not os.path.exists(config.workspace):
        os.makedirs(config.workspace)

    logger = setup_logger(os.path.join(config.workspace, 'train_log'))
    logger.info(config.print())

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    if config.gpu_id is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")

    train_data = ICDAR15(config.trainroot,config.is_pseudo , data_shape=config.data_shape, n=config.kernel_num, m=config.m)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True,
                                   num_workers=int(config.workers))

    # writer = SummaryWriter(config.output_dir)
    model = PSENet(backbone=config.backbone, pretrained=config.pretrained, result_num=config.kernel_num, scale=config.scale)
    if not config.pretrained and not config.restart_training:
        model.apply(weights_init)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = PSELoss(Lambda=config.Lambda, ratio=config.OHEM_ratio, reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.checkpoint != '' and not config.restart_training:
        start_epoch = load_checkpoint(config.checkpoint, model, logger, device, optimizer)
        start_epoch += 1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma,
                                                         last_epoch=start_epoch)
        logger.info('resume from {}, epoch={}'.format(config.checkpoint, start_epoch))
    else:
        start_epoch = config.start_epoch
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)

    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_data.__len__(), all_step))
    epoch = 0
    f1 = 0

    for epoch in range(start_epoch, config.epochs):
        start = time.time()
        train_loss, lr = train_epoch(model, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,
                                      logger)
        logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            epoch, config.epochs, train_loss, time.time() - start, lr))

        if epoch % config.save_interval == 0:
            save_path = '{}/epoch_{}.pth'.format(config.workspace, epoch)
            latest_path = '{}/latest.pth'.format(config.workspace)
            save_checkpoint(save_path, model, optimizer, epoch, logger)
            save_checkpoint(latest_path, model, optimizer, epoch, logger)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default="./config/icdar15/icdar15_baseline.py", help='')
    parser.add_argument('--resume_from', '-r', type=str, default="", help='')

    args = parser.parse_args()
    config = Config.fromfile(args.config)

    if args.resume_from != "":
        config.checkpoint = args.resume_from
        config.restart_training = False

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    main()
