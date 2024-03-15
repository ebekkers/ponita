# Adopted from: https://github.com/AmitMY/pose-format/
import torch
import numpy as np
class ShearTransform:
    """
    Applies `2D shear <https://en.wikipedia.org/wiki/Shear_matrix>`_ transformation
    
    Args:
        shear_std (float): std to use for shear transformation. Default: 0.2
    """
    def __init__(self, shear_std: float=0.2):
        self.shear_std = shear_std
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, graph):
        """
        Applies shear transformation to the given data.

        Args:
            data (dict): input data

        Returns:
            dict: data after shear transformation
        """

        x = graph.pos
        self.shear_matrix = torch.eye(2).to(self.device)
        self.shear_matrix[0][1] = torch.tensor(
            np.random.normal(loc=0, scale=self.shear_std, size=1)[0]
        )
        # n_Frames, 27, 2
        graph.pos = torch.matmul(x, self.shear_matrix)
        return graph
        