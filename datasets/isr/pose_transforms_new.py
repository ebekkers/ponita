import torch
import random
import numpy as np


class Compose:
    """
    Compose a list of pose transforms
    
    Args:
        transforms (list): List of transforms to be applied.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: dict):
        """Applies the given list of transforms

        Args:
            x (dict): input data

        Returns:
            dict: data after the transforms
        """
        for transform in self.transforms:
            x = transform(x)
        return x

class CenterAndScaleNormalize:
    """
    Centers and scales the keypoints based on the referent points given.

    Args:
        reference_points_preset (str | None, optional): can be used to specify existing presets - `mediapipe_holistic_minimal_27` or `mediapipe_holistic_top_body_59`
        reference_point_indexes (list): shape(p1, p2); point indexes to use if preset is not given then
        scale_factor (int): scaling factor. Default: 1
        frame_level (bool): Whether to center and normalize at frame level or clip level. Default: ``False``
    """
    def __init__(
        self,
        reference_points_preset=None,
        reference_point_indexes=[3, 4],
        scale_factor=1,
        frame_level=False,
    ):

        
        self.reference_point_indexes = reference_point_indexes
        self.scale_factor = scale_factor
        self.frame_level = frame_level

    def __call__(self, x):
        """
        Applies centering and scaling transformation to the given data.

        Args:
            data (dict): input data

        Returns:
            dict: data after centering normalization
        """
        C, T, V = x.shape
        x = x.permute(1, 2, 0) #CTV->TVC

        if self.frame_level:
            for ind in range(x.shape[0]):
                center, scale = self.calc_center_and_scale_for_one_skeleton(x[ind])
                x[ind] -= center
                x[ind] *= scale
        else:
            center, scale = self.calc_center_and_scale(x)
            x = x - center
            x = x * scale

        x = x.permute(2, 0, 1) #TVC->CTV
        return x
    
    def calc_center_and_scale_for_one_skeleton(self, x):
        """
        Calculates the center and scale values for one skeleton.

        Args:
            x (torch.Tensor): Spatial keypoints at a timestep

        Returns:
            [float, float]: center and scale value to normalize for the skeleton
        """
        ind1, ind2 = self.reference_point_indexes
        point1, point2 = x[ind1], x[ind2]
        center = (point1 + point2) / 2
        dist = torch.sqrt(((point1 - point2) ** 2).sum(-1))
        scale = self.scale_factor / dist
        if torch.isinf(scale).any():
            return 0, 1  # Do not normalize
        return center, scale

    def calc_center_and_scale(self, x):
        """
        Calculates the center and scale value based on the sequence of skeletons.

        Args:
            x (torch.Tensor): all keypoints for the video clip.

        Returns:
            [float, float]: center and scale value to normalize
        """
        transposed_x = x.permute(1, 0, 2) # TVC -> VTC
        ind1, ind2 = self.reference_point_indexes
        points1 = transposed_x[ind1]
        points2 = transposed_x[ind2]

        points1 = points1.reshape(-1, points1.shape[-1])
        points2 = points2.reshape(-1, points2.shape[-1])

        center = torch.mean((points1 + points2) / 2, dim=0)
        mean_dist = torch.mean(torch.sqrt(((points1 - points2) ** 2).sum(-1)))
        scale = self.scale_factor / mean_dist
        if torch.isinf(scale).any():
            return 0, 1  # Do not normalize

        return center, scale