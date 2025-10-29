import argparse
import logging
import os
import sys
import time
import yaml

import numpy as np
import torch

import util
from models.model import SinglePath_OneShot
import val as validate  # for end-of-epoch mAP
from utils.dataloaders import create_dataloader
from utils.general import LOGGER, check_dataset, check_img_size, colorstr
from utils.loss import ComputeLoss

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1)) 
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

parser = argparse.ArgumentParser("Single_Path_One_Shot")
parser.add_argument('--exp_name', type=str, default='spos_c10_train_supernet', help='experiment name')
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
parser.add_argument('--cutout', action='store_true', help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--resize', action='store_true', default=False, help='use resize')
parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold for NMS')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--save-dir', type=str, default='runs/val', help='directory to save validation outputs')
# Dataset / dataloader args (YOLO-style)
parser.add_argument('--data', type=str, default='data/VisDrone.yaml', help='dataset yaml path')
parser.add_argument('--imgsz', type=int, default=640, help='inference image size')
parser.add_argument('--cache', type=str, default='', help='cache images: "ram", "disk", or ""')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--workers', type=int, default=8, help='dataloader workers')
parser.add_argument('--image-weights', action='store_true', help='use image weights for sampling')
parser.add_argument('--quad', action='store_true', help='quad dataloader')
parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)
util.set_seed(args.seed)


def train(args, epoch, train_loader, model, compute_loss, optimizer):
    """Detection training loop for one epoch.

    Args:
        args: parsed args (contains print_freq, num_choices, layers, device)
        epoch: current epoch index
        train_loader: dataloader yielding (images, targets, paths, shapes)
        model: supernet model
        compute_loss: ComputeLoss instance (returns loss and loss items)
        optimizer: optimizer

    Returns:
        avg_loss: average total loss over the epoch
    """
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    total_loss_meter = util.AverageMeter()
    box_loss_meter = util.AverageMeter()
    obj_loss_meter = util.AverageMeter()
    cls_loss_meter = util.AverageMeter()

    steps_per_epoch = len(train_loader)
    for step, (inputs, targets, paths, shapes) in enumerate(train_loader):
        inputs = inputs.to(args.device, non_blocking=True)
        targets = targets.to(args.device)

        optimizer.zero_grad()
        # sample a random architecture path for this batch
        choice = util.random_choice(args.num_choices, args.layers)
        outputs = model(inputs, choice)

        # compute detection losses (loss_items: [box, obj, cls])
        loss, loss_items = compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        total_loss_meter.update(loss.item() * bs, bs)
        box_loss_meter.update(loss_items[0].item() * bs, bs)
        obj_loss_meter.update(loss_items[1].item() * bs, bs)
        cls_loss_meter.update(loss_items[2].item() * bs, bs)

        if step % args.print_freq == 0 or step == steps_per_epoch - 1:
            logging.info(
                '[Supernet Training] lr: %.5f epoch: %03d/%03d, step: %03d/%03d, '
                'total_loss: %.4f(%.4f), box: %.4f, obj: %.4f, cls: %.4f'
                % (lr, epoch+1, args.epochs, step+1, steps_per_epoch,
                   loss.item(), total_loss_meter.avg, box_loss_meter.avg, obj_loss_meter.avg, cls_loss_meter.avg)
            )

    return total_loss_meter.avg


def validate(args, val_loader, model, compute_loss):
    """Run full detection validation with mAP computation using the repo's `val.run`.

    We delegate to val.run(...) which already implements the full YOLOv5-style
    validation (NMS, scaling, mAP@.5 and mAP@.5:.95). Pass the active model
    and the prepared dataloader so `val.run` runs in "training" mode and uses
    the provided PyTorch model/device.
    """
    # val.run returns: (mp, mr, map50, map, lbox, lobj, lcls), maps, times
    # maps: per-class APs; times: timing profile
    logging.info("Running detection validation (mAP) using val.run()...")
    r, maps, times = validate.run(
        data=args.data,
        batch_size=max(1, args.batch_size // WORLD_SIZE),
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
        project=args.save_dir,
        name="exp",
        exist_ok=True,
        half=False,
        dnn=False,
        model=model,
        dataloader=val_loader,
        save_dir=args.save_dir,
        compute_loss=compute_loss,
    )

    # r contains (mp, mr, map50, map, lbox, lobj, lcls)
    mp, mr, map50, map, lbox, lobj, lcls = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
    results = {
        "precision": float(mp),
        "recall": float(mr),
        "mAP@.5": float(map50),
        "mAP@.5:.95": float(map),
        "box_loss": float(lbox),
        "obj_loss": float(lobj),
        "cls_loss": float(lcls),
    }
    return results


def main():
    # Check Checkpoints Direction
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)

    # Unpack basic args
    data = args.data
    single_cls = args.single_cls
    hyp = args.hyp
    batch_size = args.batch_size
    workers = args.workers

    # Hyperparameters
    if isinstance(hyp, str):
        try:
            with open(hyp, errors="ignore") as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        except Exception:
            hyp = {}
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()) if hyp else "(empty)" )
    args.hyp = hyp.copy() if isinstance(hyp, dict) else {}

    # Define Data (parse dataset yaml)
    data_dict = check_dataset(data)
    train_path, val_path = data_dict.get("train"), data_dict.get("val")

    # Number of classes
    nc = 1 if single_cls else int(data_dict.get("nc", 1))  # number of classes

    # Grid size and image size
    gs = 32  # default grid size (this repo's model does not export .stride like yolov5). Use 32 as safe default.
    imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Trainloader
    train_loader, trainset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if args.cache == "val" else args.cache,
        rect=args.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=args.image_weights,
        quad=args.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=args.seed,
    )

    # Sanity check label classes
    try:
        labels = np.concatenate(trainset.labels, 0)
        mlc = int(labels[:, 0].max())  # max label class
        assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"
    except Exception:
        LOGGER.info("Warning: unable to validate trainset labels against nc; continuing.")

    # Valloader
    val_loader, valset = create_dataloader(
        val_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=False,
        cache=None if args.cache == "val" else args.cache,
        rect=args.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=args.image_weights,
        quad=args.quad,
        prefix=colorstr("val: "),
        shuffle=False,
        seed=args.seed,
    )

    # Define Supernet
    # SinglePath_OneShot(signature: dataset, resize, layers, nc, anchors)
    model = SinglePath_OneShot(args.dataset, args.resize, args.layers, nc, anchors=hyp.get("anchors") if isinstance(hyp, dict) else None)
    logging.info(model)
    model = model.to(args.device)

    # Detection loss wrapper (expects model to provide required outputs format)
    compute_loss = ComputeLoss(model)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print('\n')

    # Running
    start = time.time()
    best_val_metric = -1e9
    for epoch in range(args.epochs):
        # Supernet Training
        train_loss = train(args, epoch, train_loader, model, compute_loss, optimizer)
        scheduler.step()
        logging.info(
            '[Supernet Training] epoch: %03d, train_loss: %.3f' %
            (epoch + 1, train_loss)
        )
        # Supernet Validation
        val_stats = validate(args, val_loader, model, compute_loss)
        # Prefer mAP@0.5 for checkpointing when available; otherwise use negative total_loss
        if 'mAP@.5' in val_stats:
            val_metric = val_stats['mAP@.5']
        elif 'mAP@.5:.95' in val_stats:
            val_metric = val_stats['mAP@.5:.95']
        else:
            # lower total_loss is better -> invert sign to make larger-is-better metric
            val_metric = -val_stats.get('total_loss', 1e9)

        # Save Best Supernet Weights
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_ckpt = os.path.join(args.ckpt_dir, '%s_%s' % (args.exp_name, 'best.pth'))
            torch.save(model.state_dict(), best_ckpt)
            logging.info('Save best checkpoints to %s' % best_ckpt)

        logging.info(
            '[Supernet Validation] epoch: %03d, val_total_loss: %.3f, val_metric: %.4f, best_metric: %.4f'
            % (epoch + 1, val_stats.get('total_loss', -1.0), val_metric, best_val_metric)
        )
        print('\n')

    # Record Time
    util.time_record(start)


if __name__ == '__main__':
    main()
