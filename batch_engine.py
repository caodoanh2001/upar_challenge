import math
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.distributed import reduce_tensor
from tools.utils import AverageMeter, to_scalar, time_str
from tqdm import tqdm

def logits4pred(criterion, logits_list):
    if criterion.__class__.__name__.lower() in ['bceloss']:
        logits = logits_list[0]
        probs = logits.sigmoid()
    else:
        assert False, f"{criterion.__class__.__name__.lower()} not exits"

    return probs, logits

#text_feats = np.load("/home/compu/doanhbc/upar_challenge/SOLIDER-PersonAttributeRecognition/feat_list_attributes.npy")
#text_feats = torch.from_numpy(text_feats).to('cuda')

def batch_trainer(cfg, args, epoch, model, model_ema, train_loader, criterion, optimizer, loss_w=[1, ], scheduler=None, tb_writer=None):
    model.train()
    epoch_time = time.time()

    loss_meter = AverageMeter()
    subloss_meters = [AverageMeter() for i in range(len(loss_w))]

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []
    preds_logits = []
    imgname_list = []
    loss_mtr_list = []

    with tqdm(desc='Epoch %d - Training' % epoch, unit='it', total=len(train_loader)) as pbar:
        for step, (imgs, gt_label, imgname) in enumerate(train_loader):
            iter_num = epoch * len(train_loader) + step

            batch_time = time.time()
            imgs, gt_label = imgs.cuda(), gt_label.cuda()
            train_logits, _ = model(imgs, gt_label)

            loss_list, loss_mtr = criterion(train_logits, gt_label)

            train_loss = 0

            for i, l in enumerate(loss_w):
                train_loss += loss_list[i] * l

            optimizer.zero_grad()
            train_loss.backward()

            if cfg.TRAIN.CLIP_GRAD:
                clip_grad_norm_(model.parameters(), max_norm=10.0)  # make larger learning rate works

            optimizer.step()
            ft_lr = optimizer.param_groups[0]['lr']
            fresh_lr = optimizer.param_groups[1]['lr']

            if cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmup_cosine' and scheduler is not None:
                scheduler.step(iter_num+1)

                if tb_writer is not None:
                    tb_writer.add_scalar('lr/step', fresh_lr, epoch)

            if model_ema is not None:
                model_ema.update(model)

            torch.cuda.synchronize()

            if len(loss_list) > 1:
                for i, meter in enumerate(subloss_meters):
                    meter.update(
                        to_scalar(reduce_tensor(loss_list[i], args.world_size)
                                if args.distributed else loss_list[i]))
            loss_meter.update(to_scalar(reduce_tensor(train_loss, args.world_size) if args.distributed else train_loss))

            train_probs, train_logits = logits4pred(criterion, train_logits)

            gt_list.append(gt_label.cpu().numpy())
            preds_probs.append(train_probs.detach().cpu().numpy())
            preds_logits.append(train_logits.detach().cpu().numpy())

            imgname_list.append(imgname)

            pbar.set_postfix(loss=loss_meter.avg)
            pbar.update()


    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    if args.local_rank == 0:
        print(f'Epoch {epoch}, LR {fresh_lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs, imgname_list, preds_logits, loss_mtr_list


def valid_trainer(cfg, args, epoch, model, valid_loader, criterion, loss_w=[1, ]):
    model.eval()
    loss_meter = AverageMeter()
    subloss_meters = [AverageMeter() for i in range(len(loss_w))]

    preds_probs = []
    preds_logits = []
    gt_list = []
    imgname_list = []
    loss_mtr_list = []

    with tqdm(desc='Epoch %d - Validation' % epoch, unit='it', total=len(valid_loader)) as pbar:
        with torch.no_grad():
            for step, (imgs, gt_label, imgname) in enumerate(valid_loader):
                imgs = imgs.cuda()
                gt_label = gt_label.cuda()
                gt_list.append(gt_label.cpu().numpy())
                gt_label[gt_label == -1] = 0
                valid_logits, _ = model(imgs, gt_label)

                loss_list, loss_mtr = criterion(valid_logits, gt_label)
                valid_loss = 0
                for i, l in enumerate(loss_list):
                    valid_loss += loss_w[i] * l

                valid_probs, valid_logits = logits4pred(criterion, valid_logits)
                preds_probs.append(valid_probs.cpu().numpy())
                preds_logits.append(valid_logits.cpu().numpy())

                if len(loss_list) > 1:
                    for i, meter in enumerate(subloss_meters):
                        meter.update(
                            to_scalar(reduce_tensor(loss_list[i], args.world_size) if args.distributed else loss_list[i]))
                loss_meter.update(to_scalar(reduce_tensor(valid_loss, args.world_size) if args.distributed else valid_loss))

                torch.cuda.synchronize()

                imgname_list.append(imgname)
                pbar.set_postfix(loss=loss_meter.avg)
                pbar.update()

    valid_loss = loss_meter.avg

    if args.local_rank == 0:
        print([f'{meter.avg:.4f}' for meter in subloss_meters])

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    preds_logits = np.concatenate(preds_logits, axis=0)

    return valid_loss, gt_label, preds_probs, imgname_list, preds_logits, loss_mtr_list
