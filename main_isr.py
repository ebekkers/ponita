import argparse
import os
import torch
import pytorch_lightning as pl
from lightning_wrappers.callbacks import EMA, EpochTimer
from lightning_wrappers.isr import PONITA_ISR
from torch_geometric.transforms import BaseTransform

from datasets.isr.pyg_dataloader_isr import ISRDataReader
from datasets.isr.pyg_dataloader_isr import PyGDataLoader


# TODO: do we need this?
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# ------------------------ Start of the main experiment script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Input arguments
    
    # Run parameters
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of epochs')
    parser.add_argument('--warmup', type=int, default=0,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-10,
                        help='weight decay')
    parser.add_argument('--temporal_weight_decay', type=float, default=1e-3,
                        help='weight decay on parameters in 1D temporal conv')
    parser.add_argument('--temporal_dropout_rate', type=float, default=0.1,
                        help='dropout rate on parameters in 1D temporal conv')
    parser.add_argument('--log', type=eval, default=True,
                        help='logging flag')
    parser.add_argument('--model_name', type=str, default='Ponita',
                        help='logging flag')
    parser.add_argument('--wandb_log_folder', type=str, default='Gloss_100',
                        help='logging flag')
    parser.add_argument('--enable_progress_bar', type=eval, default=True,
                        help='enable progress bar')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Num workers in dataloader')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # Train settings
    parser.add_argument('--train_augm', type=eval, default=False,
                        help='whether or not to use random rotations during training')
        
    # ISR Dataset settings
    ## Data location settings
    parser.add_argument('--root', type=str, default="datasets/isr",
                        help='Data set location')
    parser.add_argument('--root_metadata', type=str, default="wlasl_100.json",
                        help='Metadata json file location')
    parser.add_argument('--root_poses', type=str, default="wlasl_poses_pickle",
                        help='Pose data dir location')
    
    # Classification type settings
    parser.add_argument('--n_classes', type=str, default=988,
                        help='Number of sign classes')
    parser.add_argument('--temporal_configuration', type=str, default="spatio_temporal",
                        help='Temporal configuration of the graph. Options: spatio_temporal, per_frame') 
    
    ## Graph size parameter
    parser.add_argument('--n_nodes', type=int, default=27,
                        help='Number of nodes to use when reducing the graph - only 27 currently implemented') 
    
    # Graph connectivity settings
    parser.add_argument('--radius', type=eval, default=None,
                        help='radius for the radius graph construction in front of the force loss')
    parser.add_argument('--loop', type=eval, default=True,
                        help='enable self interactions')

    parser.add_argument('--kernel_size', type=int, default=9,
                        help='size of 1D conv kernel')    
    parser.add_argument('--stride', type=int, default=1,
                        help='size of 1D conv stride')    

    # PONTA model settings
    parser.add_argument('--num_ori', type=int, default=18,
                        help='num elements of spherical grid')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='internal feature dimension')
    parser.add_argument('--basis_dim', type=int, default=32,
                        help='number of basis functions')
    parser.add_argument('--degree', type=int, default=3,
                        help='degree of the polynomial embedding')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of message passing layers')
    parser.add_argument('--widening_factor', type=int, default=2,
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
    
    data_dir = os.path.dirname(__file__) + '/' + args.root
    data = ISRDataReader(data_dir, args)

    # Create train, val, test splits


    # Make the dataloaders
    pyg_loader = PyGDataLoader(data, args)
    
    
    # ------------------------ Load and initialize the model
    model = PONITA_ISR(args)
    

    # ------------------------ Weights and Biases logger
    if args.log:
        if args.model_name != '':
            logger = pl.loggers.WandbLogger(project=args.wandb_log_folder, name=args.model_name, config=args, save_dir='logs')
        else:
            logger = pl.loggers.WandbLogger(project=args.wandb_log_folder, name=None, config=args, save_dir='logs')
    else:
        logger = None

    # ------------------------ Set up the trainer
    
    # Seed
    pl.seed_everything(args.seed, workers=True)
    
    # Pytorch lightning call backs
    callbacks = [EMA(0.99),
                 pl.callbacks.ModelCheckpoint(monitor='val_acc', mode = 'max'),
                 EpochTimer()]
    if args.log: callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    
    # Initialize the trainer
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks, inference_mode=False, # Important for force computation via backprop
                         gradient_clip_val=0.5, accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar)
    
    # Do the training
    trainer.fit(model, pyg_loader.train_loader, pyg_loader.val_loader)
    
    # And test
    trainer.test(model, pyg_loader.test_loader, ckpt_path = "best")
