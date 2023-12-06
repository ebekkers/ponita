import torch
from torch_geometric.transforms import BaseTransform, RadiusGraph
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce, remove_self_loops, add_self_loops
from ponita.geometry.rotation import uniform_grid_s2, random_matrix
from ponita.utils.to_from_sphere import scalar_to_sphere, vec_to_sphere
import torch_geometric


class PositionOrientationGraph(BaseTransform):
    """
    A PyTorch Geometric transform that lifts a point cloud in position space, as stored in a graph data object,
    to a position-orientation space fiber bundle. The grid of orientations sampled in each fiber is shared over 
    all nodes and node features (scalars and/or vectors) are locally lifted to this grid.

    Args:
        num_ori (int): Number of orientations used to discretize the sphere.
    """

    def __init__(self, num_ori, radius=None):
        super().__init__()

        # Discretization of the orientation grid
        self.num_ori = num_ori
        self.radius = radius

        # Grid type
        if num_ori > 0:
            self.ori_grid = uniform_grid_s2(num_ori)  # [num_ori, 3]
        if radius is not None:
            self.transform = RadiusGraph(radius, loop=True, max_num_neighbors=1000)

    def __call__(self, graph):
        """
        Apply the transform to the input graph.

        Args:
            graph (torch_geometric.data.Data): Input graph containing position (graph.pos),
                                                scalar features (optional, graph.x) with shape [num_nodes, num_features], 
                                                and vector features (optional, graph.vec) with shape [num_nodes, num_vec_features, 3]

        Returns:
            torch_geometric.data.Data: Updated graph with added orientation information (graph.ori_grid with shape [num_ori, n])
                                      and lifted feature (graph.f with shape [num_nodes, num_ori, num_vec + num_x]).
        """
        if self.num_ori == -1:
            graph = self.to_po_point_cloud(graph)
        else:
            graph = self.to_po_fiber_bundle(graph)
        loop = True  # Hard-code that self-interactions are always present
        if self.radius is not None:
            graph.edge_index = torch_geometric.nn.radius_graph(graph.pos[:,:graph.n], self.radius, graph.batch, loop, max_num_neighbors=1000)
        else:
            if loop:
                graph.edge_index = coalesce(add_self_loops(graph.edge_index)[0])
        return graph

    def to_po_fiber_bundle(self, graph):
        """
        Internal method to add orientation information to the input graph.

        Args:
            graph (torch_geometric.data.Data): Input graph containing position (graph.pos),
                                                scalar features (optional, graph.x with shape [num_nodes, num_scalars]), and
                                                vector features (optional, graph.vec with shape [num_nodes, num_vec, n]).

        Returns:
            torch_geometric.data.Data: Updated graph with added orientation information (graph.ori_grid) and features (graph.f).
        """
        graph.ori_grid = self.ori_grid.type_as(graph.pos)
        graph.num_ori = self.num_ori
        graph.n = graph.pos.size(1)

        # Lift input features to spheres
        inputs = []
        if hasattr(graph, "x"):    inputs.append(scalar_to_sphere(graph.x, graph.ori_grid))
        if hasattr(graph, "vec"):  inputs.append(vec_to_sphere(graph.vec, graph.ori_grid))
        graph.x = torch.cat(inputs, dim=-1)  # [num_nodes, num_ori, input_dim + input_dim_vec]

        # Return updated graph
        return graph

    def to_po_point_cloud(self, graph):
        
        graph.n = graph.pos.size(1)
        
        # -----------  The relevant items in the original graph

        edge_index = remove_self_loops(coalesce(graph.edge_index))[0]
        pos = graph.pos
        batch = graph.batch
        source, target = edge_index

        # ----------- Lifted positions (each original edge now becomes a node)

        pos_s, pos_t = pos[source], pos[target]
        dist = (pos_s - pos_t).norm(dim=-1, keepdim=True)
        ori_t = (pos_s - pos_t) / dist
        graph.pos = torch.cat([pos_t, ori_t], dim=-1)  # [6D position-orientation element]
        
        # ----------- Lift the edge_index

        # First, count number of orientations per base node (how many times a node is connected)
        # In edge_index we took the receiving node as "base node"
        num_base = pos.size(0)
        num_ori_at_base = edge_index[1].type(torch.float).histc(bins=num_base, min=0, max=num_base - 1).type(torch.int64)

        # The following is used as lookup table for connecting lifted idx to base idx
        # The corresponding lifted indices
        lifted_index = torch.arange(source.size(0), device=source.device)  # lifted idx
        # The transposed adjacency matrix
        adj_T = SparseTensor(row=target, col=source, value=lifted_index, sparse_sizes=(num_base, num_base))

        # At each source node in the base graph, we might have multiple associated lifted nodes
        # the following collects the ids of the lifted nodes which have source as base
        source_lift = adj_T[source].storage.value()

        # Then each of these lifted nodes sends towards the base at the target
        # The following are the indices of the target base nodes
        target_base = target[adj_T[source].storage.row()]

        # At these receiving base nodes we can have multiple lifted points
        # The following extracts the indices of the lifted nodes
        target_lift = adj_T[target_base].storage.value()
        
        # Finally we repeat the source indices by the number of lifted nodes at the receiving end
        source_lift = source_lift.repeat_interleave(num_ori_at_base[target_base])

        # The resulting edge_index in the lifted space
        edge_index_lift = torch.stack([source_lift, target_lift])
        graph.edge_index = coalesce(edge_index_lift, sort_by_row=False)

        # ----------- Lift the batch

        if hasattr(graph, "batch"):
            if graph.batch is not None:
                graph.batch = batch[edge_index[1]]

        # ----------- Lift the scalar and vector features, overwrite x
        inputs = []
        if hasattr(graph, "x"):    
            inputs.append(graph.x[edge_index[1]])
        if hasattr(graph, "vec"):  
            inputs.append(torch.einsum('bcd,bd->bc', graph.vec[edge_index[1]], graph.pos[:,3:]))
        graph.x = torch.cat(inputs, dim=-1)  # [num_lifted_nodes, num_channels]

        # ----------- Utility to be able to project back to the base node (e.g. via scatter collect)

        graph.scatter_projection_index = edge_index[1]
        

        return graph
        