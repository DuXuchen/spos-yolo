import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

import util
from models.model import SinglePath_OneShot
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_img_size, colorstr, non_max_suppression, xywh2xyxy, scale_boxes
from utils.loss import ComputeLoss
from utils.metrics import ap_per_class
from val import process_batch

parser = argparse.ArgumentParser("Single_Path_One_Shot")
parser.add_argument('--exp_name', type=str, default='spos_c10_train_supernet', help='experiment name')
# Supernet Settings
parser.add_argument('--layers', type=int, default=20, help='batch size')
parser.add_argument('--num_choices', type=int, default=4, help='number choices per layer')
# Search Settings
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--search_num', type=int, default=1000, help='search number')
parser.add_argument('--seed', type=int, default=0, help='search seed')
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
parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold for NMS')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--workers', type=int, default=8, help='dataloader workers')
parser.add_argument('--single-cls', action='store_true', help='treat dataset as single-class')
args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)
util.set_seed(args.seed)


def evaluate_single_path_detection(args, val_loader, model, compute_loss, choice):
    """Evaluate one sampled path on the detection val_loader and return mAP metrics.

    Returns a dict with keys: 'mAP@.5', 'mAP@.5:.95', 'precision', 'recall', 'loss'.
    """
    model.eval()
    device = args.device
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    stats = []  # (correct, conf, pred_cls, target_cls)
    loss_meter = util.AverageMeter()

    with torch.no_grad():
        for batch_i, (imgs, targets, paths, shapes) in enumerate(val_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            imgs = imgs.float() / 255.0

            # forward with fixed choice
            outputs = model(imgs, choice)

            # loss
            if compute_loss is not None:
                loss, loss_items = compute_loss(outputs, targets)
                loss_meter.update(loss.item() * imgs.size(0), imgs.size(0))

            # NMS
            preds = non_max_suppression(outputs, conf_thres=args.conf_thres, iou_thres=args.iou_thres)

            # per-image stats
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl = labels.shape[0]
                if pred is None or pred.shape[0] == 0:
                    if nl:
                        stats.append((np.zeros((0, niou), dtype=bool), np.zeros(0), np.zeros(0), labels[:, 0].cpu().numpy()))
                    continue

                predn = pred.clone()
                # native-space predictions and labels
                shape = shapes[si][0]
                scale_boxes(imgs[si].shape[1:], predn[:, :4], shape, shapes[si][1])

                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_boxes(imgs[si].shape[1:], tbox, shape, shapes[si][1])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    correct = process_batch(predn, labelsn, iouv)
                    stats.append((correct.cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), labels[:, 0].cpu().numpy()))
                else:
                    stats.append((np.zeros((0, niou), dtype=bool), pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), np.zeros(0)))

    # compute mAP
    if len(stats):
        # concatenate
        correct = np.concatenate([s[0] for s in stats], 0)
        conf = np.concatenate([s[1] for s in stats], 0)
        pcls = np.concatenate([s[2] for s in stats], 0)
        tcls = np.concatenate([s[3] for s in stats], 0)
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(correct, conf, pcls, tcls)
        ap50 = ap[:, 0] if ap.size else np.array([0.0])
        map50 = float(ap50.mean())
        map = float(ap.mean())
        precision = float(p.mean())
        recall = float(r.mean())
    else:
        map50 = 0.0
        map = 0.0
        precision = 0.0
        recall = 0.0

    return {
        'mAP@.5': map50,
        'mAP@.5:.95': map,
        'precision': precision,
        'recall': recall,
        'loss': loss_meter.avg,
    }


if __name__ == '__main__':
    # Dataset yaml and valloader (YOLO-style)
    data_dict = check_dataset(args.data)
    train_path, val_path = data_dict.get('train'), data_dict.get('val')
    nc = 1 if args.single_cls else int(data_dict.get('nc', 1))

    gs = 32
    imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)

    val_loader, val_dataset = create_dataloader(
        val_path,
        imgsz,
        args.batch_size,
        gs,
        args.single_cls,
        hyp=None,
        augment=False,
        cache=None,
        rect=False,
        rank=-1,
        workers=args.workers,
        image_weights=False,
        quad=False,
        prefix=colorstr("val: "),
        shuffle=False,
        seed=args.seed,
    )

    # Load Pretrained Supernet
    model = SinglePath_OneShot(args.dataset, args.resize, args.layers, nc, anchors=None).to(args.device)
    best_supernet_weights = './checkpoints/spos_c10_train_supernet_best.pth'
    checkpoint = torch.load(best_supernet_weights, map_location=args.device)
    model.load_state_dict(checkpoint, strict=True)
    logging.info('Finish loading checkpoint from %s', best_supernet_weights)

    # Detection loss wrapper
    compute_loss = ComputeLoss(model)

    # Random Search over sampled architectures
    start = time.time()
    best_map50 = -1.0
    best_choice = None
    for num in range(args.search_num):
        choice = util.random_choice(args.num_choices, args.layers)
        stats = evaluate_single_path_detection(args, val_loader, model, compute_loss, choice)
        map50 = stats['mAP@.5']
        map095 = stats['mAP@.5:.95']
        logging.info('Num: %04d/%04d, choice: %s, mAP@.5: %.4f, mAP@.5:.95: %.4f, loss: %.4f'
                     % (num, args.search_num, choice, map50, map095, stats['loss']))
        if map50 > best_map50:
            best_map50 = map50
            best_choice = choice

    logging.info('Best mAP@.5: %.4f Best_choice: %s' % (best_map50, best_choice))
    util.time_record(start)
