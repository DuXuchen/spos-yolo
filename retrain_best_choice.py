import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from thop import profile
from torchvision import datasets

import util
from models.model import SinglePath_Network
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_img_size, colorstr
from utils.loss import ComputeLoss
import val as validate
from utils.general import non_max_suppression, scale_boxes, xywh2xyxy
from utils.metrics import ap_per_class

parser = argparse.ArgumentParser("Single_Path_One_Shot")
parser.add_argument('--exp_name', type=str, default='spos_c10_train_choice_model', help='experiment name')
# Supernet Settings
parser.add_argument('--layers', type=int, default=20, help='batch size')
parser.add_argument('--num_choices', type=int, default=4, help='number choices per layer')
# Training Settings
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--epochs', type=int, default=600, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--print_freq', type=int, default=100, help='print frequency of training')
parser.add_argument('--val_interval', type=int, default=5, help='validate and save frequency')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/', help='checkpoints direction')
parser.add_argument('--seed', type=int, default=0, help='training seed')
# Dataset Settings
parser.add_argument('--data_root', type=str, default='./dataset/', help='dataset dir')
parser.add_argument('--classes', type=int, default=10, help='dataset classes')
parser.add_argument('--dataset', type=str, default='cifar10', help='path to the dataset')
parser.add_argument('--cutout', action='store_true', help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--resize', action='store_true', default=False, help='use resize')
parser.add_argument('--data', type=str, default='data/VisDrone.yaml', help='dataset yaml path')
parser.add_argument('--imgsz', type=int, default=640, help='inference image size')
parser.add_argument('--cache', type=str, default='', help='cache images: "ram", "disk", or ""')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--workers', type=int, default=8, help='dataloader workers')
parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)
util.set_seed(args.seed)


def train(args, epoch, train_loader, model, compute_loss, optimizer):
    """Detection training loop for one epoch using ComputeLoss."""
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    total_loss_meter = util.AverageMeter()
    box_loss_meter = util.AverageMeter()
    obj_loss_meter = util.AverageMeter()
    cls_loss_meter = util.AverageMeter()

    steps_per_epoch = len(train_loader)
    for step, (imgs, targets, paths, shapes) in enumerate(train_loader):
        imgs = imgs.to(args.device, non_blocking=True)
        targets = targets.to(args.device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss, loss_items = compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss_meter.update(loss.item() * bs, bs)
        box_loss_meter.update(loss_items[0].item() * bs, bs)
        obj_loss_meter.update(loss_items[1].item() * bs, bs)
        cls_loss_meter.update(loss_items[2].item() * bs, bs)

        if step % args.print_freq == 0 or step == steps_per_epoch - 1:
            logging.info(
                '[Model Training] lr: %.5f epoch: %03d/%03d, step: %03d/%03d, total_loss: %.4f(%.4f), box: %.4f, obj: %.4f, cls: %.4f'
                % (lr, epoch+1, args.epochs, step+1, steps_per_epoch, loss.item(), total_loss_meter.avg,
                   box_loss_meter.avg, obj_loss_meter.avg, cls_loss_meter.avg)
            )

    return {
        'total_loss': total_loss_meter.avg,
        'lbox': box_loss_meter.avg,
        'lobj': obj_loss_meter.avg,
        'lcls': cls_loss_meter.avg,
    }


def validate(args, val_loader, model, compute_loss):
    """Run full detection validation delegating to val.run to compute mAP and losses."""
    logging.info("Running detection validation (mAP) using val.run()...")
    r, maps, times = validate.run(
        data=args.data,
        batch_size=max(1, args.batch_size),
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        task="val",
        device="",
        workers=args.workers,
        single_cls=args.single_cls,
        augment=False,
        verbose=False,
        save_txt=False,
        save_hybrid=False,
        save_conf=False,
        project="runs/val",
        name="exp",
        exist_ok=True,
        half=False,
        dnn=False,
        model=model,
        dataloader=val_loader,
        save_dir="",
        compute_loss=compute_loss,
    )

    # r contains (mp, mr, map50, map, lbox, lobj, lcls)
    mp, mr, map50, map095, lbox, lobj, lcls = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
    results = {
        "precision": float(mp),
        "recall": float(mr),
        "mAP@.5": float(map50),
        "mAP@.5:.95": float(map095),
        "box_loss": float(lbox),
        "obj_loss": float(lobj),
        "cls_loss": float(lcls),
    }
    return results


def main():
    # Prepare dataset and dataloaders (YOLO-style)
    data_dict = check_dataset(args.data)
    train_path, val_path = data_dict.get('train'), data_dict.get('val')
    nc = 1 if args.single_cls else int(data_dict.get('nc', 1))

    gs = 32
    imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)

    train_loader, train_dataset = create_dataloader(
        train_path,
        imgsz,
        args.batch_size,
        gs,
        args.single_cls,
        hyp=None,
        augment=True,
        cache=None,
        rect=args.rect,
        rank=-1,
        workers=args.workers,
        image_weights=False,
        quad=False,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=args.seed,
    )

    val_loader, val_dataset = create_dataloader(
        val_path,
        imgsz,
        args.batch_size,
        gs,
        args.single_cls,
        hyp=None,
        augment=False,
        cache=None,
        rect=args.rect,
        rank=-1,
        workers=args.workers,
        image_weights=False,
        quad=False,
        prefix=colorstr("val: "),
        shuffle=False,
        seed=args.seed,
    )

    # Define Choice Model (fixed architecture)
    choice = [1, 0, 3, 1, 3, 0, 3, 0, 0, 3, 3, 0, 1, 0, 1, 2, 2, 1, 1, 3]
    model = SinglePath_Network(args.dataset, args.resize, args.layers, choice, nc, anchors=None)
    compute_loss = ComputeLoss(model)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - (epoch / args.epochs))

    # Print Model Information
    # Optional: compute FLOPs/params on a representative input
    try:
        sample_size = (1, 3, imgsz, imgsz)
        flops, params = profile(model, inputs=(torch.randn(*sample_size),), verbose=False)
        logging.info('Choice Model Information: params: %.2fM, flops:%.2fM' % ((params / 1e6), (flops / 1e6)))
    except Exception:
        logging.info('FLOPs/profile failed - continuing')
    model = model.to(args.device)
    logging.info(model)
    print('\n')

    # Running
    start = time.time()
    best_metric = -1e9
    for epoch in range(args.epochs):
        # Choice Model Training
        train_stats = train(args, epoch, train_loader, model, compute_loss, optimizer)
        scheduler.step()
        logging.info('[Model Training] epoch: %03d, train_total_loss: %.3f' % (epoch + 1, train_stats['total_loss']))

        # Choice Model Validation (mAP)
        val_stats = validate(args, val_loader, model, compute_loss)

        # prefer mAP@.5 for checkpointing when available
        if 'mAP@.5' in val_stats:
            metric = val_stats['mAP@.5']
        elif 'mAP@.5:.95' in val_stats:
            metric = val_stats['mAP@.5:.95']
        else:
            metric = -val_stats.get('box_loss', 1e9)

        if metric > best_metric:
            best_metric = metric
            best_ckpt = os.path.join(args.ckpt_dir, '%s_%s' % (args.exp_name, 'best.pth'))
            torch.save(model.state_dict(), best_ckpt)
            logging.info('Saved best checkpoint to %s' % best_ckpt)

        logging.info('[Model Validation] epoch: %03d, val_mAP@.5: %.3f, best_metric: %.3f' % (epoch + 1, val_stats.get('mAP@.5', -1.0), best_metric))
        print('\n')

    # Record Time
    util.time_record(start)


if __name__ == '__main__':
    main()
