import torch
from torch_geometric.transforms import BaseTransform
from ponita.geometry.rotation import uniform_grid_s2, random_matrix
from ponita.utils.to_from_sphere import scalar_to_sphere, vec_to_sphere


class PositionOrientationGraph(BaseTransform):
    """
    A PyTorch Geometric transform that lifts a point cloud in position space, as stored in a graph data object,
    to a position-orientation space fiber bundle. The grid of orientations sampled in each fiber is shared over 
    all nodes and node features (scalars and/or vectors) are locally lifted to this grid.

    Args:
        num_ori (int): Number of orientations used to discretize the sphere.
    """

    def __init__(self, num_ori):
        super().__init__()

        # Discretization of the orientation grid
        self.num_ori = num_ori

        # Grid type
        self.ori_grid = uniform_grid_s2(num_ori)  # [num_ori, 3]

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
        return self.to_po_fiber_bundle(graph)

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

        # Lift input features to spheres
        inputs = []
        if hasattr(graph, "x"):    inputs.append(scalar_to_sphere(graph.x, graph.ori_grid))
        if hasattr(graph, "vec"):  inputs.append(vec_to_sphere(graph.vec, graph.ori_grid))
        graph.f = torch.cat(inputs, dim=-1)  # [num_nodes, num_ori, input_dim + input_dim_vec]

        # Return updated graph
        return graph
