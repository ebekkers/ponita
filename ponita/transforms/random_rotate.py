import torch
from torch_geometric.transforms import BaseTransform
from ponita.geometry.rotation import random_matrix


class RandomRotate3D(BaseTransform):
    """
    A PyTorch Geometric transform that randomly rotates each point cloud in a batch of graphs
    by sampling rotations from a uniform distribution over the Special Orthogonal Group SO(3).

    Args:
        None
    """

    def __init__(self, attr_list):
        super().__init__()
        self.attr_list = attr_list

    def __call__(self, graph):
        """
        Apply the random rotation transform to the input graph.

        Args:
            graph (torch_geometric.data.Data): Input graph containing position (graph.pos),
                                                and optionally, batch information (graph.batch).

        Returns:
            torch_geometric.data.Data: Updated graph with randomly rotated positions.
                                      The positions in the graph are rotated using a random rotation matrix
                                      sampled from a uniform distribution over SO(3).
        """
        rand_SO3 = self.random_rotation(graph)
        return self.rotate_graph(graph, rand_SO3)

    def rotate_graph(self, graph, rand_SO3):
        for attr in self.attr_list:
            if hasattr(graph, attr):
                setattr(graph, attr, self.rotate_attr(getattr(graph, attr), rand_SO3))
        return graph
        
    def rotate_attr(self, attr, rand_SO3):
            rand_SO3 = rand_SO3.type_as(attr)
            if len(rand_SO3.shape)==3:
                if len(attr.shape)==2:
                    return torch.einsum('bij,bj->bi', rand_SO3, attr)
                else:
                    return torch.einsum('bij,bcj->bci', rand_SO3, attr)
            else:
                if len(attr.shape)==2:
                    return torch.einsum('ij,bj->bi', rand_SO3, attr)
                else:
                    return torch.einsum('ij,bcj->bci', rand_SO3, attr)
    
    def random_rotation(self, graph):
        if graph.batch is not None:
            batch_size = graph.batch.max() + 1
            random_SO3 = random_matrix(batch_size).to(graph.batch.device)
            random_SO3 = random_SO3[graph.batch]
        else:
            random_SO3 = random_matrix(1)[0]
        return random_SO3
