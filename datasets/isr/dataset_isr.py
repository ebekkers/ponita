import json
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader



class ISRDataReader:
    def __init__(self, data_dir, args):
        batch_size = args.batch_size
        self.file_path = os.path.join(data_dir, args.root_metadata)
        self.pkl_folder = os.path.join(data_dir, args.root_poses)
        self.batch_size = batch_size
        self.n_nodes = 27
        # Indexes for reduction of graph nodes
        self.pose_indexes = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]

        
        self.data_dict = self._load_and_split_data()


    def _load_and_split_data(self, seperate_temporal=True):
        metadata = self._load_metadata()
        self.gloss_dict = self._load_gloss_pose_dict(metadata)
        data_dict = self._load_pkl_files()
        
        # Decouple time and look at each frame as a singular instance
        if seperate_temporal:
            data_dict = self.seperate_temporal_dimension(data_dict)
            data_dict = self.clean_padding(data_dict)

        return data_dict

    ######################
    # A. Reading functionalities
    ######################

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
        labels = {word: index for index, word in enumerate(list(self.gloss_dict.keys()))}
        for gloss, metadata in self.gloss_dict.items():
            for vid_id, split in metadata:
                pkl_path = os.path.join(self.pkl_folder, f'{vid_id}.pkl')
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        graph_data = pickle.load(file)
                        kps = self._transform_data(graph_data["keypoints"][:, :, :2])
                        data_dict[vid_id] = {'label': labels[gloss], 'gloss': gloss, 'node_pos': kps, 'split': split}

        return data_dict
    
    ######################
    # B. Pre-processing functionalities
    ######################

    def _transform_data(self, kps):
        frames = torch.tensor(np.asarray(kps, dtype=np.float32)).permute(2, 0, 1)
        # TODO: Introduce the other transformations
        return self._downsample_data(frames)

    def _downsample_data(self, frames):
        """ Downsample pose graph based on the standard node selection from holistic 27 minimal nodes
        """
        return frames[:, :, self.pose_indexes]
    
    def seperate_temporal_dimension(self, data_dict):
        new_data_dict = {}
    
        for vid_id, data in data_dict.items():
            label = data['label']
            gloss = data['gloss']
            split = data['split']
            node_pos = data['node_pos']
            num_frames = node_pos.shape[1]

            for frame_idx in range(num_frames):
                new_key = f"{vid_id}_f_{frame_idx}"
                frame_data = node_pos[:, frame_idx, :]
                new_data_dict[new_key] = {
                    'label': label,
                    'gloss': gloss, 
                    'node_pos': frame_data,
                    'split': split
                }
        return new_data_dict
    
    def clean_padding(self, data_dict):
        """ During pose extraction with mediapipe padding is added for reasons beyond me
            This function removes all data points where all the node positions are zero
            Yeah ok turns out it was very few points, but still
        """
        filtered_data_dict = {}

        for key, data in data_dict.items():
            # Check if all elements in node_pos are zeros
            if not torch.all(data['node_pos'] == 0):
                filtered_data_dict[key] = data

        return filtered_data_dict       


class PyGDataLoader:
    def __init__(self, data, batch_size):
        self.data_dict = data.data_dict
        self.batch_size = batch_size
        self.inward_edges = [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], 
                            [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], 
                            [7, 15], [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], 
                            [17, 23], [23, 24], [17, 25], [25, 26]]

    def build_loaders(self):
        train_data, val_data, test_data = self._split_dataset(self.data_dict)
        self.train_loader  = self._load_data(train_data)
        self.val_loader = self._load_data(val_data, shuffle = False, split = 'val')
        self.test_loader = self._load_data(test_data, shuffle = False, split='test')

    def _split_dataset(self, data_dict):
        train_data = {k: v for k, v in data_dict.items() if v['split'] == 'train'}
        val_data = {k: v for k, v in data_dict.items() if v['split'] == 'val'}
        test_data = {k: v for k, v in data_dict.items() if v['split'] == 'test'}
        return train_data, val_data, test_data

    def _load_data(self, data_dict, shuffle = True, split = 'train'): 
        data_list = []
        for id, data in data_dict.items():
            pos = data['node_pos'].T
            y = data['label']
            edge_index = torch.tensor(self.inward_edges, dtype=torch.long).t().contiguous()
            data_list.append(Data(pos = pos, x = pos ,edge_index=edge_index, y=y))
           
        
        print('Number of ' + split + ' points:', len(data_list))
        
        return DataLoader(data_list, batch_size=self.batch_size, shuffle=shuffle)
    


if __name__ == "__main__":

    data = ISRDataReader('/home/oline/PONITA_SLR/datasets/isr/', batch_size=32)

    pyg_loader = PyGDataLoader(data, batch_size=32)
    pyg_loader.build_loaders()













