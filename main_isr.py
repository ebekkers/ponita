import argparse
import os
import numpy as np
import torch

from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from lightning_wrappers.callbacks import EMA, EpochTimer
from lightning_wrappers.isr import PONITA_ISR
from torch_geometric.transforms import BaseTransform

from datasets.isr.dataset_isr import ISRDataReader
from datasets.isr.dataset_isr import PyGDataLoader

# TODO: do we need this?
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Sparsify(BaseTransform):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def __call__(self, graph):
        select = graph.x[:,0] > self.threshold
        graph.x = graph.x[select]
        graph.pos = graph.pos[select]
        if graph.batch is not None:
            graph.batch = graph.batch[select]
        graph.edge_index = None
        return graph
    
from torch_geometric.transforms import BaseTransform
class RemoveDuplicatePoints(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, graph):
        dists = (graph.pos[:,None,:] - graph.pos[None,:,:]).norm(dim=-1)
        dists = dists + 100. * torch.tril(torch.ones_like(dists), diagonal=0)
        min_dists = dists.min(dim=1)[0]
        select = min_dists > 0.
        graph.x = graph.x[select]
        graph.pos = graph.pos[select]
        graph.edge_index = None
        return graph
    

# ------------------------ Start of the main experiment script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Input arguments
    
    # Run parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--warmup', type=int, default=0,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-10,
                        help='weight decay')
    parser.add_argument('--log', type=eval, default=True,
                        help='logging flag')
    parser.add_argument('--enable_progress_bar', type=eval, default=True,
                        help='enable progress bar')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Num workers in dataloader')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # Train settings
    parser.add_argument('--train_augm', type=eval, default=True,
                        help='whether or not to use random rotations during training')
        
    # ISR Dataset
    parser.add_argument('--root', type=str, default="datasets/isr",
                        help='Data set location')
    # ISR Dataset
    parser.add_argument('--n_classes', type=str, default=10,
                        help='Number of sign classes')
    
    # Graph connectivity settings
    parser.add_argument('--radius', type=eval, default=None,
                        help='radius for the radius graph construction in front of the force loss')
    parser.add_argument('--loop', type=eval, default=True,
                        help='enable self interactions')
    
    # PONTA model settings
    parser.add_argument('--num_ori', type=int, default=8,
                        help='num elements of spherical grid')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='internal feature dimension')
    parser.add_argument('--basis_dim', type=int, default=256,
                        help='number of basis functions')
    parser.add_argument('--degree', type=int, default=3,
                        help='degree of the polynomial embedding')
    parser.add_argument('--layers', type=int, default=5,
                        help='Number of message passing layers')
    parser.add_argument('--widening_factor', type=int, default=4,
                        help='Number of message passing layers')
    parser.add_argument('--layer_scale', type=float, default=0,
                        help='Initial layer scale factor in ConvNextBlock, 0 means do not use layer scale')
    parser.add_argument('--multiple_readouts', type=eval, default=False,
                        help='Whether or not to readout after every layer')
    
    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use (assumes all are on one node)')
    
    # Arg parser
    args = parser.parse_args()
    
    # ------------------------ Device settings
    
    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()

    # ------------------------ Dataset
    
    # Load the dataset and set the dataset specific settings
    data = ISRDataReader('/home/oline/PONITA_SLR/datasets/isr/', batch_size=args.batch_size)

    # Create train, val, test splits


    # Make the dataloaders
    pyg_loader = PyGDataLoader(data, batch_size=args.batch_size)
    pyg_loader.build_loaders()
    
    # ------------------------ Load and initialize the model
    model = PONITA_ISR(args)

    # ------------------------ Weights and Biases logger
    if args.log:
        logger = pl.loggers.WandbLogger(project="PONITA-ISR", name=None, config=args, save_dir='logs')
    else:
        logger = None

    # ------------------------ Set up the trainer
    
    # Seed
    pl.seed_everything(args.seed, workers=True)
    
    # Pytorch lightning call backs
    callbacks = [EMA(0.99),
                 pl.callbacks.ModelCheckpoint(monitor='valid ACC', mode = 'max'),
                 EpochTimer()]
    if args.log: callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    
    # Initialize the trainer
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks, inference_mode=False, # Important for force computation via backprop
                         gradient_clip_val=0.5, accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar)
    
    # Do the training
    trainer.fit(model, pyg_loader.train_loader, pyg_loader.val_loader)
    
    # And test
    trainer.test(model, pyg_loader.test_loader, ckpt_path = "best")