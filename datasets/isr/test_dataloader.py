import unittest
import torch 
from datasets.isr.pyg_dataloader_isr import SpatioTemporalGraphBuilder 

class TestSpatioTemporalGraphBuilder(unittest.TestCase):
    
    def test_build_edges(self):
        num_features = 2
        num_frames = 3
        num_nodes = 4
        inward_edges = [[0, 1], [0, 2], [2, 3]]  
        
        random_pos_data = torch.rand(num_features, num_frames, num_nodes)
        pos_shape = random_pos_data.shape

        st_edges = [[0, 1], [0, 2], [2, 3],
                    [4, 5], [4, 6], [6, 7],
                    [8, 9], [8, 10], [10, 11],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    [4, 8], [5, 9], [6, 10], [7, 11]
                    ]  
        st_edges = torch.tensor(st_edges).t().contiguous()
        


        graph_builder = SpatioTemporalGraphBuilder(random_pos_data, inward_edges)
    
        edges = graph_builder.build_spatiotemporal_edges()
        data = graph_builder.reshaped_data
        
        # check data shape
        assert(data.shape == (num_features, num_frames * num_nodes))

        # Check that the number of edges is correct
        self.assertEqual(len(edges), len(st_edges))  # Each edge appears twice (undirected graph)

        # Check that the edges are correct
        for edge in st_edges:
            self.assertIn(edge, st_edges)

if __name__ == '__main__':
    unittest.main()