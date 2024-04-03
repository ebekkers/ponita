import pytorch_lightning as pl
import torch
import torch.nn as nn
from ponita.models.ponita import Ponita
import torchmetrics
import numpy as np
from .scheduler import CosineWarmupScheduler
from ponita.transforms.random_rotate import RandomRotate
#from ponita.transforms.pose_transforms import ShearTransform
#from ponita.transforms.pose_transforms import CenterAndScaleNormalize
import os
from ponita.models.temporal_ponita import TemporalPonita


class PONITA_ISR(pl.LightningModule):
    """
    """

    def __init__(self, args):
        super().__init__()

        # Store some of the relevant args
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup = args.warmup
        if args.layer_scale == 0.:
            args.layer_scale = None

        # For rotation augmentations during training and testing
        self.train_augm = args.train_augm
        self.rotation_transform = RandomRotate(['pos'], n=2)
        #self.shear_transform = ShearTransform(shear_std = 0.1)
        
        
        # The metrics to log
        self.train_metric = torchmetrics.Accuracy(task='multiclass', num_classes=args.n_classes)
        self.valid_metric = torchmetrics.Accuracy(task='multiclass', num_classes=args.n_classes)
        self.test_metric = torchmetrics.Accuracy(task='multiclass', num_classes=args.n_classes)

        # Input/output specifications:
        in_channels_scalar = 0  
        in_channels_vec = args.n_nodes  # Node landmark one-hot encoding 
        out_channels_scalar =  args.n_classes # The target
        out_channels_vec = 0  # 

        # Make the model
        self.model = TemporalPonita(in_channels_scalar + in_channels_vec,
                        args.hidden_dim,
                        out_channels_scalar,
                        args.layers,
                        output_dim_vec=out_channels_vec,
                        radius=args.radius,
                        num_ori=args.num_ori,
                        basis_dim=args.basis_dim,
                        degree=args.degree,
                        widening_factor=args.widening_factor,
                        layer_scale=args.layer_scale,
                        task_level='graph',
                        multiple_readouts=args.multiple_readouts,
                        lift_graph=True,
                        args=args).to('cuda:0')
    
    def forward(self, graph):
        # Only utilize the scalar (energy) prediction
        pred, _ = self.model(graph)

        return pred

    def training_step(self, graph):
        if self.train_augm:
            graph = self.rotation_transform(graph)
            # graph = self.shear_transform(graph)
            #graph = CenterAndScaleNormalize(graph)
        pred = self(graph)
        pred = torch.nn.functional.log_softmax(pred, dim=-1)
        loss = torch.nn.functional.nll_loss(pred, graph.y)
        self.train_metric(pred, graph.y)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pred = self(graph)
        self.valid_metric(pred, graph.y)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.valid_metric, prog_bar=True)
    
    def test_step(self, graph, batch_idx):
        pred = self(graph)
        self.test_metric(pred, graph.y)

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_metric, prog_bar=True)
    
    def configure_optimizers(self):
        """
        Adapted from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        decay_conv = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        whitelist_weight_conv_modules = (torch.nn.Conv1d,)
        blacklist_weight_modules = (torch.nn.LazyBatchNorm1d, torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_conv_modules):
                    # weights of whitelist modules will be weight decayed
                    decay_conv.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('layer_scale'):
                    no_decay.add(fpn)
           

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay & decay_conv
        union_params = decay | no_decay | decay_conv
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(decay_conv))], "weight_decay": 1e-3},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=self.lr)
        scheduler = CosineWarmupScheduler(optimizer, self.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    