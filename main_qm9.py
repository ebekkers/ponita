import argparse
import os
import numpy as np
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from lightning_wrappers.callbacks import EMA, EpochTimer
from lightning_wrappers.qm9 import PONITA_QM9


# TODO: do we need this?
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# ------------------------ Start of the main experiment script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Input arguments
    
    # Run parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--warmup', type=int, default=10,
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
    
    # Test settings
    parser.add_argument('--repeats', type=int, default=5,
                        help='number of repeated forward passes at test-time')
    
    # QM9 Dataset
    parser.add_argument('--root', type=str, default="datasets/qm9",
                        help='Data set location')
    parser.add_argument('--target', type=str, default="alpha",
                        help='MD17 target')
    
    # Graph connectivity settings
    parser.add_argument('--radius', type=eval, default=1000.,
                        help='radius for the radius graph construction in front of the force loss')
    parser.add_argument('--loop', type=eval, default=True,
                        help='enable self interactions')
    
    # PONTA model settings
    parser.add_argument('--num_ori', type=int, default=-1,
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
    dataset = QM9(root=args.root)
    
    # Create train, val, test split (same random seed and splits as DimeNet)
    random_state = np.random.RandomState(seed=42)
    perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
    train_idx, val_idx, test_idx = perm[:110000], perm[110000:120000], perm[120000:]
    datasets = {'train': dataset[train_idx], 'val': dataset[val_idx], 'test': dataset[test_idx]}
    
    # Select the right target
    targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0',
           'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11, 12, 13, 14, 15])  # We will automatically replace U0 -> U0_atom etc.
    dataset.data.y = dataset.data.y[:, idx]
    dataset.data.y = dataset.data.y[:, targets.index(args.target)]

    # Make the dataloaders
    dataloaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == 'train'), num_workers=args.num_workers)
        for split, dataset in datasets.items()}
    
    # ------------------------ Load and initialize the model
    model = PONITA_QM9(args)
    model.set_dataset_statistics(datasets['train'])

    # ------------------------ Weights and Biases logger
    if args.log:
        logger = pl.loggers.WandbLogger(project="PONITA-QM9", name=args.target.replace(" ", "_"), config=args, save_dir='logs')
    else:
        logger = None

    # ------------------------ Set up the trainer
    
    # Seed
    pl.seed_everything(args.seed, workers=True)
    
    # Pytorch lightning call backs
    callbacks = [EMA(0.99),
                 pl.callbacks.ModelCheckpoint(monitor='valid MAE', mode = 'min'),
                 EpochTimer()]
    if args.log: callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    
    # Initialize the trainer
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks, inference_mode=False, # Important for force computation via backprop
                         gradient_clip_val=0.5, accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar)
    
    # Do the training
    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    
    # And test
    trainer.test(model, dataloaders['test'], ckpt_path = "best")
