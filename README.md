# ✨ 🐴 🔥 PONITA 

*Under construction...*


## What is this repository about?
This repository contains the code for the paper [Fast, Expressive SE(n) Equivariant Networks through Weight-Sharing in Position-Orientation Space](https://arxiv.org/abs/2310.02970). We propose **PONITA**: a simple fully convolutional SE(n) equivariant architecture. We developed it primarily for 3D point-cloud data, but the method is also applicable to 2D point clouds and 2D/3D images/volumes (though not yet with this repo). PONITA is an equivariant model that does not require working with steerable/Clebsch-Gordan methods, but has the same capabilities in that __it can handle scalars and vectors__ equally well. Moreover, since it does not depend on Clebsch-Gordan tensor products __PONITA is much faster__ than the typical steerable/tensor field network!

## About the name
PONITA is an acronym for Position-Orientation space Networks based on InvarianT Attributes. We believe this acronym is apt for the method for two reasons. Firstly, PONITA sounds like "bonita" ✨ which means pretty in Spanish, we personally think the architecture is pretty and elegant. Secondly, [Ponyta](https://bulbapedia.bulbagarden.net/wiki/Ponyta_(Pok%C3%A9mon)) 🐴 🔥 is a fire Pokémon which is known to be very fast, our method is fast as well.

## About the implementation
The PONITA model is provided in ```models/ponita.py```. In the forward pass it assumes a graph object with attributes ```graph.x``` which are the scalar features at each node, ```graph.pos``` their corresponding positions, and ```graph.y``` which are the targets at node level (```task_level='node'```) or at graph level (```task_level='graph'```). Optionally, the model can also take as input ```graph.vec``` which are vector features attached to each node. The model will always return two outputs: scalar ```output_scalar``` and ```output_vector```, in case no scalar or vector outputs are specified they will take on the value ```None```. Finally, an ```edge_index``` should be provided (in the form "source_to_target"). A connectiviy ```radius``` parameter can be provided in order to window the convolution kernels (default is None which means effectively the graph is fully connected, or the original edge_index is provided).

We relied on [torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) in our code.

The pieces of code used by PONITA are given in ```ponita/```. The most relevant code specific is given in
* ```ponita/geometry/invariants``` in which the unique and complete invariant attributes are computed
* ```ponita/nn/conv``` in which the (separable) group convolution is implemented.
* ```ponita/utils/to_from_sphere``` in which functions are give to embed scalars or vectors as a spherical signal and the other way around

## Testing PONITA
The paper results can be reproduced by running the following command for rMD17:

```python3 main_md17.py```

We did a sweep over all "revised *" targets and the seeds 0, 1, 2. Otherwise the defaut settings in ```main.py``` are used.

For the n-body experiments run

```python3 main_nbody.py```

We did a sweep over seeds 0, 1, 2 with the default parameters in ```main_nbody.py``` fixed.

_The EDM experiments will be added soon!_

## Conda environment
In order to run the code in this repository install the following conda environment
```
conda create --yes --name ponita python=3.10 numpy scipy matplotlib
conda activate ponita
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pyg==2.3.1 -c pyg -y
pip3 install wandb
pip3 install pytorch_lightning==1.8.6
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

## Work in progress
To be added soon:
* R3 functionality
* Equivariant diffusion experiments
* R2xS1 functionality

## Acknowledgements
The experimental setup builds upon the code bases of [EGNN repository](https://github.com/vgsatorras/egnn) and [EDM repository](https://github.com/ehoogeboom/e3_diffusion_for_molecules). The grid construction code is adapted from [Regular SE(3) Group Convolution](https://github.com/ThijsKuipers1995/gconv) library. We deeply thank the authors for open sourcing their codes. We are also very grateful to the developers of the amazing libraries [torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html), [pytorch lightning](https://lightning.ai/), and [weights and biases](https://https://wandb.ai/) !
