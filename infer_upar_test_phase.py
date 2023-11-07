import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

from configs import cfg, update_config
#from dataset.multi_label.coco import COCO14
from dataset.augmentation import get_transform
from metrics.ml_metrics import get_multilabel_metrics
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_ema import ModelEmaV2
from optim.adamw import AdamW
# from scheduler.cosine_lr import CosineLRScheduler
from tools.distributed import distribute_bn
from tools.vis import tb_visualizer_pedes
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from dataset.pedes_attr.pedes import PedesAttr, PedesAttrUPAR, PedesAttrUPARInfer, PedesAttrUPARInferTestPhase
from models.base_block import FeatClassifier
from models.model_factory import build_loss, build_classifier, build_backbone

from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool, gen_code_archive
from models.backbone import swin_transformer2
from losses import bceloss, scaledbceloss
from models import base_block
from tqdm import tqdm
torch.set_printoptions(precision=10)

set_seed(605)


def main(cfg, args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, _ = get_model_log_path(exp_dir, cfg.NAME)

    _, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)

    template_submission_csv = os.path.join(cfg.DATASET.PHASE2_ROOT_PATH, 'submission_templates_test/task1', 'predictions.csv')
    valid_set = PedesAttrUPARInferTestPhase(cfg=cfg, csv_file=template_submission_csv, transform=valid_tsfm,
                              root_path=cfg.DATASET.PHASE2_ROOT_PATH, target_transform=cfg.DATASET.TARGETTRANSFORM)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Build model
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)
    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=40,
        c_in=2048,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model = get_reload_weight(model_dir, model, pth='./checkpoints/best_model.pth')
    # Write results
    model.eval()
    f = open('./predictions.csv', 'w')
    f.write('# Image,Age-Young,Age-Adult,Age-Old,Gender-Female,Hair-Length-Short,Hair-Length-Long,Hair-Length-Bald,UpperBody-Length-Short,UpperBody-Color-Black,UpperBody-Color-Blue,UpperBody-Color-Brown,UpperBody-Color-Green,UpperBody-Color-Grey,UpperBody-Color-Orange,UpperBody-Color-Pink,UpperBody-Color-Purple,UpperBody-Color-Red,UpperBody-Color-White,UpperBody-Color-Yellow,UpperBody-Color-Other,LowerBody-Length-Short,LowerBody-Color-Black,LowerBody-Color-Blue,LowerBody-Color-Brown,LowerBody-Color-Green,LowerBody-Color-Grey,LowerBody-Color-Orange,LowerBody-Color-Pink,LowerBody-Color-Purple,LowerBody-Color-Red,LowerBody-Color-White,LowerBody-Color-Yellow,LowerBody-Color-Other,LowerBody-Type-Trousers&Shorts,LowerBody-Type-Skirt&Dress,Accessory-Backpack,Accessory-Bag,Accessory-Glasses-Normal,Accessory-Glasses-Sun,Accessory-Hat\n')
    with torch.no_grad():
        for step, (imgs, imgnames) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda()
            valid_logits, attns = model(imgs)
            valid_probs = torch.sigmoid(valid_logits[0])
            for imgname, valid_prob in zip(imgnames, valid_probs):
                write_score = ','.join(list(map(str, valid_prob.tolist())))
                f.write(imgname + ',' + write_score + '\n')
    
    f.close()


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", default='./configs/upar.yaml', help="decide which cfg to use", type=str,
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args)
