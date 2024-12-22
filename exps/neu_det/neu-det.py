# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp
import torch.nn as nn

ROOT = "/home/onepiece/programs/DFeS-DNet"
dateset_path=os.path.join(ROOT, "data/NEU-DET")


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 6
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 20
        self.warmup_lr = 1e-6
        self.weight_decay=0.05

        self.dateset_path=dateset_path
        self.iouweigth = 5.0

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.eval_interval=1

        self.basic_lr_per_img= 1.6e-3 / 48 
        self.min_lr_ratio=1e-5 / 1.6e-3
        self.max_epoch = 150
        self.no_aug_epochs=30 
        self.save_history_ckpt=False

        self.data_dir=dateset_path
        self.output_dir=os.path.join(ROOT, "work_dir/baseline_outputs_neucoco")
        self.train_ann = "train2017.json"
        self.val_ann = "val2017.json"

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_optimizer(self, batch_size):
        from tools.set_optim import build_optimizer_lightvit
        dir_path=os.path.join(self.output_dir,"optim_parm.txt")
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size


            def get_parms(model):
                has_decay = []
                no_decay = []

                for name, param in model.named_parameters():

                    if not param.requires_grad:
                        continue  # frozen weights
                    if (len(param.shape) == 1 or name.endswith(".bias")):
                        no_decay.append(param)
                        # logger.info(f"{name} has no weight decay")
                    else:
                        has_decay.append(param)
                return [{'params': has_decay},
                        {'params': no_decay, 'weight_decay': 0.}]

            parameters=get_parms(self.model)

            optimizer = torch.optim.AdamW(parameters,
                                    betas=(0.9, 0.999),
                                    lr=lr,
                                    weight_decay=self.weight_decay)

            self.optimizer = optimizer

        return self.optimizer

    def get_model(self):
        from yolox.models import YOLOX
        from yolox.models.modified_LVT import M_LVT
        from yolox.models.mcnet import MCNet
        from yolox.models.edhead import EDHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            out_channels= [64,160,256]
            backbone_noneck=M_LVT()
            backbone = MCNet(self.depth, self.width, out_channels=out_channels, act=self.act,backbone=backbone_noneck)
            head = EDHead(self.num_classes, self.width, in_channels=in_channels, act=self.act,reg_weight=self.iouweigth)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)

        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model