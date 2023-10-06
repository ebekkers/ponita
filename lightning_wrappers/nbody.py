import pytorch_lightning as pl
import torch
import torchmetrics
from models.ponita import PONITA
from .scheduler import CosineWarmupScheduler


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

        # The metrics to log
        self.train_metric = torchmetrics.MeanSquaredError()
        self.valid_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()

        # Input/output specifications:
        in_channels_scalar = 1  # Charge, Velocity norm
        in_channels_vec = 1  # Velocity, rel_pos
        out_channels_scalar = 0  # None
        out_channels_vec = 1  # Output velocity

        # Make the model
        self.model = PONITA(in_channels_scalar,
                        args.hidden_dim,
                        out_channels_scalar,
                        args.layers,
                        input_dim_vec = in_channels_vec,
                        output_dim_vec = out_channels_vec,
                        radius=args.radius,
                        n=args.n,
                        M=args.M,
                        basis_dim=args.basis_dim,
                        degree=args.degree,
                        widening_factor=args.widening_factor,
                        layer_scale=args.layer_scale,
                        task_level='node')                        

    def forward(self, graph):
        _, pred = self.model(graph)
        return pred[...,0]  # only 1 vector is predicted

    def training_step(self, graph):
        pos_pred = graph.pos + self(graph)
        loss = torch.mean((pos_pred - graph.y)**2)
        self.train_metric(pos_pred, graph.y)
        return loss

    def on_training_epoch_end(self):
        self.log("train MSE", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pos_pred = graph.pos + self(graph)
        self.valid_metric(pos_pred, graph.y)  

    def on_validation_epoch_end(self):
        self.log("valid MSE", self.valid_metric, prog_bar=True)
    
    def test_step(self, graph, batch_idx):
        pos_pred = graph.pos + self(graph)
        self.test_metric(pos_pred, graph.y)  

    def on_test_epoch_end(self):
        self.log("test MSE", self.test_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {"optimizer": optimizer, "monitor": "val_loss"}
        # scheduler = CosineWarmupScheduler(optimizer, self.warmup, self.trainer.max_epochs)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}