import argparse
import os
import torch
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, Compose, RadiusGraph
import pytorch_lightning as pl
from lightning_wrappers.callbacks import EMA, EpochTimer
from lightning_wrappers.md17 import PONITA_MD17


# ------------------------ Some transforms specific to the rMD17 tasks
# One-hot encoding of atom type
class OneHotTransform(BaseTransform):
    def __init__(self, k=None):
        super().__init__()
        self.k = k

    def __call__(self, graph):
        if self.k is None:
            graph.x = torch.nn.functional.one_hot(graph.z).float()
        else:
            graph.x = torch.nn.functional.one_hot(graph.z, self.k).squeeze().float()

        return graph
# Unit conversion
class Kcal2meV(BaseTransform):
    def __init__(self):
        # Kcal/mol to meV
        self.conversion = 43.3634

    def __call__(self, graph):
        graph.energy = graph.energy * self.conversion
        graph.force = graph.force * self.conversion
        return graph


# ------------------------ Start of the main experiment script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Input arguments
    
    # Run parameters
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs')
    parser.add_argument('--warmup', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16,
                        help='weight decay')
    parser.add_argument('--log', type=eval, default=True,
                        help='logging flag')
    parser.add_argument('--enable_progress_bar', type=eval, default=False,
                        help='enable progress bar')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Num workers in dataloader')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # Train settings
    parser.add_argument('--train_augm', type=eval, default=True,
                        help='whether or not to use random rotations during training')
    parser.add_argument('--lambda_F', type=float, default=500.0,
                        help='coefficient in front of the force loss')
    
    # Test settings
    parser.add_argument('--repeats', type=int, default=5,
                        help='number of repeated forward passes at test-time')
    
    # MD17 Dataset
    parser.add_argument('--root', type=str, default="datasets",
                        help='Data set location')
    parser.add_argument('--target', type=str, default="revised aspirin",
                        help='MD17 target')
    
    # Graph connectivity settings
    parser.add_argument('--radius', type=eval, default=None,
                        help='radius for the radius graph construction in front of the force loss')
    parser.add_argument('--loop', type=eval, default=True,
                        help='enable self interactions')
    
    # PONTA model settings
    parser.add_argument('--num_ori', type=int, default=20,
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
    parser.add_argument('--multiple_readouts', type=eval, default=True,
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
    transform = [Kcal2meV(), OneHotTransform(9), RadiusGraph((args.radius or 1000.), loop=args.loop, max_num_neighbors=1000)]
    dataset = MD17(root=args.root, name=args.target, transform=Compose(transform))
    
    # Create train, val, test split
    test_idx = list(range(min(len(dataset),100000)))  # The whole dataset consist sof 100,000 samples
    train_idx = test_idx[::100]  # Select every other 100th sample for training
    del test_idx[::100]   # and remove these from the test set
    val_idx = train_idx[::20]  # Select every 20th sample from the train set for validation
    del train_idx[::20]  # and remove these from the train set

    # Dataset and loaders
    datasets = {'train': dataset[train_idx], 'val': dataset[val_idx], 'test': dataset[test_idx]}
    dataloaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == 'train'), num_workers=args.num_workers)
        for split, dataset in datasets.items()}
    
    # ------------------------ Load and initialize the model
    model = PONITA_MD17(args)
    model.set_dataset_statistics(datasets['train'])

    # ------------------------ Weights and Biases logger
    if args.log:
        logger = pl.loggers.WandbLogger(project="PONITA-MD17", name=args.target.replace(" ", "_"), config=args, save_dir='logs')
    else:
        logger = None

    # ------------------------ Set up the trainer
    
    # Seed
    pl.seed_everything(args.seed, workers=True)
    
    # Pytorch lightning call backs
    callbacks = [EMA(0.99),
                 pl.callbacks.ModelCheckpoint(monitor='valid MAE (energy)', mode = 'min'),
                 EpochTimer()]
    if args.log: callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    
    # Initialize the trainer
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks, inference_mode=False, # Important for force computation via backprop
                         gradient_clip_val=0.5, accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar)
    
    # Do the training
    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    
    # And test
    trainer.test(model, dataloaders['test'], ckpt_path = "best")
