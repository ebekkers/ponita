import json
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader



class ISRGraphDataLoader:
    def __init__(self, file_path, batch_size):
        self.file_path = os.path.join(file_path, 'subset_metadata.json')
        self.pkl_folder = os.path.join(file_path, 'subset_selection')
        self.batch_size = batch_size
        self.n_nodes = 27
        self.pose_indexes = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
        self.inward_edges = [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], 
                            [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], 
                            [7, 15], [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], 
                            [17, 23], [23, 24], [17, 25], [25, 26]]
        
        self.train_data, self.val_data, self.test_data = self._load_and_split_data()


    def _load_and_split_data(self):
        metadata = self._load_metadata()
        self.gloss_dict = self._load_gloss_pose_dict(metadata)
        data_dict = self._load_pkl_files()

        return self._split_dataset(data_dict)

    def _load_metadata(self):
        with open(self.file_path, 'r') as file:
            return json.load(file)

    def _load_gloss_pose_dict(self, metadata):
        gloss_dict = {}
        for item in metadata:  
            gloss = item['gloss']
            gloss_dict.setdefault(gloss, []).extend([(instance['video_id'], instance['split']) for instance in item['instances']])
        return gloss_dict

    def _load_pkl_files(self):
        data_dict = {}
        for gloss, metadata in self.gloss_dict.items():
            for vid_id, split in metadata:
                pkl_path = os.path.join(self.pkl_folder, f'{vid_id}.pkl')
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        graph_data = pickle.load(file)
                        kps = self._transform_data(graph_data["keypoints"][:, :, :2])
                        data_dict[vid_id] = {'label': gloss, 'node_pos': kps, 'split': split}
        return data_dict

    def _transform_data(self, kps):
        frames = torch.tensor(np.asarray(kps, dtype=np.float32)).permute(2, 0, 1)
        return self._downsample_data(frames)

    def _downsample_data(self, frames):
        return frames[:, :, self.pose_indexes]

    def _split_dataset(self, data_dict):
        train_data = {k: v for k, v in data_dict.items() if v['split'] == 'train'}
        val_data = {k: v for k, v in data_dict.items() if v['split'] == 'val'}
        test_data = {k: v for k, v in data_dict.items() if v['split'] == 'test'}
        return train_data, val_data, test_data

if __name__ == "__main__":
    graph_loader = ISRGraphDataLoader('/home/oline/PONITA_SLR/datasets/isr/', batch_size=32)
    print('Number of training points:', len(graph_loader.train_data))
    print('Number of validation points:', len(graph_loader.val_data))
    print('Number of test points:', len(graph_loader.test_data))















###############################################################
def make_pyg_loader(dataset, batch_size, shuffle, num_workers, radius, loop):
    
    
    """
    data_list = []
    radius = radius or 1000.
    radius_graph = RadiusGraph(radius, loop=loop, max_num_neighbors=1000)
    for data in dataset:
        loc, vel, edge_attr, charges, loc_end = data
        x = charges
        vec = vel[:,None,:]  # [num_pts, num_channels=1, 3]
        # Build the graph
        graph = Data(pos=loc, x=x, vec=vec, y=loc_end)
        graph = radius_graph(graph)
        # Append to the database list
        data_list.append(graph)
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    """