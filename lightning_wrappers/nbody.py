import pytorch_lightning as pl
import torch
import torchmetrics
from models.ponita import PONITA
from .scheduler import CosineWarmupScheduler
from ponita.transforms.random_rotate import RandomRotate3D
from torch_geometric.data import Batch
import numpy as np


class PONITA_NBODY(pl.LightningModule):
    """Graph Neural Network module"""

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
        self.rotation_transform = RandomRotate3D(['pos','vec','y'])

        # The metrics to log
        self.train_metric = torchmetrics.MeanSquaredError()
        self.valid_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()

        # Input/output specifications:
        in_channels_scalar = 1  # Charge
        in_channels_vec = 1  # Velocity
        out_channels_scalar = 0  # None
        out_channels_vec = 1  # Output velocity

        # Make the model
        self.model = PONITA(in_channels_scalar + in_channels_vec,
                        args.hidden_dim,
                        out_channels_scalar,
                        args.layers,
                        output_dim_vec = out_channels_vec,
                        radius=args.radius,
                        n=args.n,
                        basis_dim=args.basis_dim,
                        degree=args.degree,
                        widening_factor=args.widening_factor,
                        layer_scale=args.layer_scale,
                        task_level='node',
                        multiple_readouts=args.multiple_readouts)
        
    def forward(self, graph):
        _, pred = self.model(graph)
        return graph.pos + pred[..., 0, :]

    def training_step(self, graph):
        if self.train_augm:
            graph = self.rotation_transform(graph)
        pos_pred = self(graph)
        loss = torch.mean((pos_pred - graph.y)**2)
        self.train_metric(pos_pred, graph.y)
        return loss

    def on_train_epoch_end(self):
        self.log("train MSE", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pos_pred = self(graph)
        self.valid_metric(pos_pred, graph.y)  

    def on_validation_epoch_end(self):
        self.log("valid MSE", self.valid_metric, prog_bar=True)
    
    def test_step(self, graph, batch_idx):
        pos_pred = self(graph)
        self.test_metric(pos_pred, graph.y)  

    def on_test_epoch_end(self):
        self.log("test MSE", self.test_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}