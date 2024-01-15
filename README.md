# ✨ 🐴 🔥 PONITA 

*Under construction...*

## What is this repository about?
This repository contains the code for the paper [Fast, Expressive SE(n) Equivariant Networks through Weight-Sharing in Position-Orientation Space](https://arxiv.org/abs/2310.02970). We propose **PONITA**: a simple fully convolutional SE(n) equivariant architecture. We developed it primarily for 3D point-cloud data, but the method is also applicable to 2D point clouds and 2D/3D images/volumes (though not yet with this repo). PONITA is an equivariant model that does not require working with steerable/Clebsch-Gordan methods, but has the same capabilities in that __it can handle scalars and vectors__ equally well. Moreover, since it does not depend on Clebsch-Gordan tensor products __PONITA is much faster__ than the typical steerable/tensor field network!

## About the name
PONITA is an acronym for Position-Orientation space Networks based on InvarianT Attributes. We believe this acronym is apt for the method for two reasons. Firstly, PONITA sounds like "bonita" ✨ which means pretty in Spanish, we personally think the architecture is pretty and elegant. Secondly, [Ponyta](https://bulbapedia.bulbagarden.net/wiki/Ponyta_(Pok%C3%A9mon)) 🐴 🔥 is a fire Pokémon which is known to be very fast, our method is fast as well.

## About the implementation

### The ponita model
The PONITA model is provided in ```models/ponita.py``` and is designed to support point clouds in 2D and 3D position space $\mathbb{R}^2$ and $\mathbb{R}^3$ out of the box. It further more supports input point clouds in 2D and 3D position-orientation spaces $\mathbb{R}^2 \times S^1$ and $\mathbb{R}^3 \times S^2$. 

The forward assumes a graph object with attributes
* ```graph.x```: which are the scalar features at each node, 
* ```graph.pos``` their corresponding positions, and 
* ```graph.y``` which are the targets at node level (```task_level='node'```) or at graph level (```task_level='graph'```). 
* ```graph.edge_index``` which defines how each node is connected to the other (source to target). However, the model also contains a ```radius``` option, which when specified is used to construct a radius graph which will overwrite (or define) the edge_index. It is also used to smoothly mask the message functions.

Optionally, the model can also take as input 
* ```graph.vec``` which are vector features attached to each node. 

The model will always return **two outputs**: scalar ```output_scalar``` and ```output_vector```, in case no scalar or vector outputs are specified they will take on the value ```None```. 

### The ponita layers

```PositionOrientationGraph(num_ori, radius)```, found in ```ponita\transforms\position_orientation_graph.py``` is a torch geometric transform that transforms the input graph to a graph in position orientation space. It will have two modes:
* **Fiber bundel mode**: When ```num_ori > 0``` a regular fiber bundle (assigning a grid of ```num_ori``` orientations per position) is generated from the provided input point cloud in $\mathbb{R}^2$ or $\mathbb{R}^3$. Subsequently the separable conv layers as described in the main-body in the text can be used. The fiber bundle interpretation is described in Appendix A.
* **Point cloud mode**: 
    * When ```num_ori = 0``` the graph will be treated as a point cloud on $\mathbb{R}^n$. Nothing really happens in this transform except for defining some variables for consistency with the position-orientation graphs.
    * When ```num_ori < -1``` the graph will be lifted to a position orientation graph by using the provided ```graph.edge_index```. In this transformation each edge will become a node which has a certain position (starting point of the edge) and orientation (the normalized direction vector from source to target position). Also an object ```graph.scatter_projection_index``` will be generated to be able to reduce the lifted graph back to the original position point cloud representation. Note for this setting ```graph.edge_index``` **is required** to be existing in the graph.

```SEnInvariantAttributes(separable, point_cloud)```, found in ```ponita\transforms\invariants.py``` is a torch geometric transform that adds invariant attributes to the graph. Only in fiber bundle mode can ```separable``` take on the value ```True```, as then one separate interactions spatial from those within the fiber. The ```separable``` option could also be set to ```False``` in the bundle mode to do full convolution over position-orientation space, but this requires quite some compute and memory. The option ```point_cloud``` switches between bundle mode (```point_cloud=False```) or point cloud mode (```point_cloud=True```). So, we have the following settings:
* ```separable=True, point_cloud=False```: generates ```graph.attr``` (shape = ```[num_edges, num_ori, 2]```) and ```graph.fiber_attr``` (shape = ```[num_ori, num_ori, 1]```).
* ```separable=False, point_cloud=False```: generates ```graph.attr``` (shape = ```[num_edges, num_ori, num_ori, 3]```)
* ```separable=False, point_cloud=True```: generates ```graph.attr``` (shape = ```[num_edges, d]```) with ```d=1``` for position graphs and ```d=3``` for position-orientation graphs in 2D or 3D.

```Conv(in_channels, out_channels, attr_dim, bias=True, aggr="add", groups=1)```, found in ```ponita\nn\conv.py``` implements convolution as a ```torch_geometric.nn.MessagePassing``` module. Considering the above graph constructions, this module operates on graphs in poin cloud mode. The forward requires a graph with ```graph.x``` (the features) and ```graph.edge_index``` and ```graph.attr``` (the pair-wise invariants, or embeddings thereof).

```FiberBundleConv(in_channels, out_channels, attr_dim, bias=True, aggr="add", separable=True, groups=1)```, also found in ```ponita\nn\conv.py``` implements convolution as a ```torch_geometric.nn.MessagePassing``` module over the fiber bundle. Considering the above graph constructions, this module operates on graphs in fiber bundle mode (a grid of orientations at each node). The forward requires a graph with ```graph.x``` (the features) and ```graph.edge_index``` (connecting the spatial base-points) and ```graph.attr``` and ```graph.fiber_attr``` (these are the pair-wise spatial or orientation-spatial attributes, or embeddings thereof), see SEnInvariantAttributes. 

```ConvNext(channels, conv, act=torch.nn.GELU(), layer_scale=1e-6, widening_factor=4)```, found in ```ponita\nn\convnext```, is a wrapper that turns the convolution layer ```conv``` into a ConvNext block.


### Back-end libraries

We relied on [torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) in our code.

## Reproducing PONITA results

### MD17 (3D point clouds)

The paper results can be reproduced by running the following command for rMD17:

```python3 main_md17.py```

We did a sweep over all "revised *" targets and the seeds 0, 1, 2. Otherwise the defaut settings in ```main.py``` are used. By setting ```num_ori=0``` the PNITA results are generated, with ```num_ori = 20``` the PONITA results are generated. The table shows our results compared to one of the seminal works on *steerable* equivariant convolutions: [NEQUIP](https://github.com/mir-group/nequip). Our methods, PNITA for position space convolutions and PONITA for position-orientation space convolutions, are based on invariants only. For comparison to other state-of-the-art-methods see our paper.

| |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | .
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-
| <td colspan=2>Azobenzene <td colspan=2>Aspirin <td colspan=2>Benzene <td colspan=2>Ethanol <td colspan=2>Malonaldehyde <td colspan=2>Naphthalene <td colspan=2>Paracetamol <td colspan=2>Salicyli cacid <td colspan=2>Toluene <td colspan=2>Uracil
| | E | F | E | F | E | F | E | F | E | F | E | F | E | F | E | F | E | F | E | F
| NEQUIP  | 2.3 | 8.2 | **0.7** | 2.9 | **0.04** | **0.3** | **0.4** | 2.8 | 0.8 | 5.1 | **0.2** | **1.3** | 1.4 | 5.9 | **0.7** | 4.0 | **0.3** | 1.6 | **0.4** | 3.1
| PNITA  | 4.7 | 16.3 | 3.2 | 12.2 | 0.2 | 0.4 | 0.7 | 4.1 | 0.9 | 5.1 | 1.1 | 5.6 | 2.8 | 11.4 | 1.7 | 8.6 | 0.6 | 3.4 | 0.9 | 5.6
| **PONITA** | **1.7** | **5.8** | **0.7** | **2.3** | 0.17 | **0.3** | **0.4** | **2.5** | **0.6** | **4.0** | 0.3 | **1.3** | **1.1** | **4.3** | **0.7** | **3.3** | **0.3** | **1.3** | **0.4** | **2.4**


### N-body (3D point clouds with input vectors)
For the n-body experiments run

```python3 main_nbody.py```

We did a sweep over seeds 0, 1, 2 with the default parameters in ```main_nbody.py``` fixed. In this case we cannot set ```num_ori=0``` as need to be able to handle input vectors. The mean squared error on predicted future positions of the particles is given below. Here we compared to the seminal works on this problem which use invariant feature representations ([EGNN](ref)) and steerable representations ([SEGNN](bla)). Again, in contrast to the steerable method SEGNN ours works with invariants only and thus does not require specialized tensor product operations.

| Method | MSE
|-|-
| EGNN | 0.0070
| SEGNN | 0.0043
| PONITA | 0.0043


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


## Acknowledgements
The experimental setup builds upon the code bases of [EGNN repository](https://github.com/vgsatorras/egnn) and [EDM repository](https://github.com/ehoogeboom/e3_diffusion_for_molecules). The grid construction code is adapted from [Regular SE(3) Group Convolution](https://github.com/ThijsKuipers1995/gconv) library. We deeply thank the authors for open sourcing their codes. We are also very grateful to the developers of the amazing libraries [torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html), [pytorch lightning](https://lightning.ai/), and [weights and biases](https://https://wandb.ai/) !
