import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.ponita import Ponita
import torchmetrics
import numpy as np
from .scheduler import CosineWarmupScheduler
from torch_geometric.data import Batch
from ponita.transforms.random_rotate import RandomRotate3D


class PONITA_QM9(pl.LightningModule):
    """
    """

    def __init__(self, args):
        super().__init__()

        # Store some of the relevant args
        self.repeats = args.repeats
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup = args.warmup
        if args.layer_scale == 0.:
            args.layer_scale = None
        
        # For rotation augmentations during training and testing
        self.train_augm = args.train_augm
        self.rotation_transform = RandomRotate3D(['pos'])
        
        # Shift and scale before callibration
        self.shift = 0.
        self.scale = 1.

        # The metrics to log
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()
        self.test_metrics = nn.ModuleList([torchmetrics.MeanAbsoluteError() for r in range(self.repeats)])

        # Input/output specifications:
        in_channels_scalar = 11  # One-hot encoding
        in_channels_vec = 0  # 
        out_channels_scalar = 1  # The target
        out_channels_vec = 0  # 

        # Make the model
        self.model = Ponita(in_channels_scalar + in_channels_vec,
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
                        lift_graph=True)
    
    def set_dataset_statistics(self, dataloader):
        print('Computing dataset statistics...')
        ys = []
        for data in dataloader:
            ys.append(data.y)
        ys = np.concatenate(ys)
        self.shift = np.mean(ys)
        self.scale = np.std(ys)
        print('Mean and std of target are:', self.shift, '-', self.scale)

    def forward(self, graph):
        # Only utilize the scalar (energy) prediction
        pred, _ = self.model(graph)
        return pred.squeeze(-1)

    def training_step(self, graph):
        pred = self(graph)
        # loss = torch.mean((pred - (graph.y - self.shift) / self.scale)**2)
        loss = torch.mean((pred - (graph.y - self.shift) / self.scale).abs())
        self.train_metric(pred * self.scale + self.shift, graph.y)
        return loss

    def on_training_epoch_end(self):
        self.log("train MAE", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pred = self(graph)
        self.valid_metric(pred * self.scale + self.shift, graph.y)

    def on_validation_epoch_end(self):
        self.log("valid MAE", self.valid_metric, prog_bar=True)
    
    def test_step(self, graph, batch_idx):
        pred = self(graph)
        self.test_metric(pred * self.scale + self.shift, graph.y)

    def on_test_epoch_end(self):
        self.log("test MAE", self.test_metric, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}