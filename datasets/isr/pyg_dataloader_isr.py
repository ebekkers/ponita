import json
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader



class ISRDataReader:
    def __init__(self, data_dir, args):
        print('Reading data...')
        
        self.args = args
        self.N_NODES = args.n_nodes

        file_path = os.path.join(data_dir, args.root_metadata)
        pickle_path = os.path.join(data_dir, args.root_poses)
        

        # Load metadata
        metadata = self._load_metadata(file_path)
        self._load_gloss_pose_dict(metadata)
        self.data_dict = self._load_pkl_files(pickle_path)
        
        # Treat each frame as individual data points
        if args.temporal_configuration == 'per_frame':
            self.data_dict = self._seperate_temporal_dimension(self.data_dict)

        
        # Build spatio temporal graph 
        elif args.temporal_configuration == 'spatio_temporal':
            self.data_dict = self._build_spatio_temporal_graph(self.data_dict)
        
        else:
            raise ValueError('Temporal configuration not recognized')
        

    def _load_metadata(self, file_path):
        """ Load the metadata from the json file
        """
        with open(file_path, 'r') as file:
            return json.load(file)

    def _load_gloss_pose_dict(self, metadata):
        """ Load the gloss dictionary from the metadata
        """
        self.gloss_dict = {}
        for item in metadata:  
            self.gloss_dict.setdefault(item['gloss'], []).extend(
                [(instance['video_id'], instance['split']) for instance in item['instances']])


    def _load_pkl_files(self, pickle_path):
        """ Load the pickle files and create a dictionary with the data
        """
        labels = {word: index for index, word in enumerate(self.gloss_dict.keys())}

        data_dict = {
            vid_id: {
                'label': labels[gloss],
                'gloss': gloss,
                'node_pos': self._transform_data(pickle.load(open(os.path.join(pickle_path, f'{vid_id}.pkl'), 'rb'))["keypoints"][:, :, :2]),
                'split': split
            }
            for gloss, metadata in self.gloss_dict.items()
            for vid_id, split in metadata
            if os.path.exists(os.path.join(pickle_path, f'{vid_id}.pkl'))
        }

        return data_dict

    
    #--------------------------------------
    # B. Pre-processing functionalities
    #--------------------------------------

    def _transform_data(self, kps):
        """ Apply selected transformations to the data
        """
        frames = torch.tensor(np.asarray(kps, dtype=np.float32)).permute(2, 0, 1)
        data = self._downsample_data(frames)

        # TODO: Introduce the other transformations
        
        return data

    def _downsample_data(self, frames):
        """ Downsample pose graph based on the standard node selection from holistic 27 minimal nodes
        """
        # Indexes for reduction of graph nodes of graph size 27 nodes, predefined in holistic mediapipe package 
        pose_indexes = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
        return frames[:, :, pose_indexes]

    #--------------------------------------
    # C. Graph construction functionalities
    #--------------------------------------    
    def _separate_temporal_dimension(self, data_dict):
        """
        Separates the temporal dimension of the data.
        
        For each video, it splits the data into individual frames and stores them 
        with a unique key representing each frame of the video.
        
        :param data_dict: A dictionary with video data, keyed by video ID.
        :return: A dictionary with data separated by individual frames.
        """
        separated_frame_dict = {}

        node_features = torch.tensor(np.eye(self.N_NODES))  

        for vid_id, data in data_dict.items():
            label, gloss, split = data['label'], data['gloss'], data['split']
            node_pos = data['node_pos']
            num_frames = node_pos.shape[1]

            for frame_idx in range(num_frames):
                new_key = f"{vid_id}_f_{frame_idx}"
                frame_data = node_pos[:, frame_idx, :]
                separated_frame_dict[new_key] = {
                    'label': label,
                    'gloss': gloss, 
                    'x': node_features,  
                    'node_pos': frame_data,
                    'split': split
                }

        return separated_frame_dict
    
    def _build_spatio_temporal_graph(self, data_dict):
        """
        Builds a spatio-temporal graph from the provided data dictionary.

        Each item in the data dictionary is transformed using the SpatioTemporalGraphBuilder,
        and the resulting spatio-temporal graph data is stored in a new dictionary.

        :param data_dict: A dictionary containing video data.
        :return: A dictionary containing the spatio-temporal graph data.
        """
        return {
            vid_id: {
                'label': data['label'],
                'gloss': data['gloss'],
                'x': st_builder.landmark_features.T,
                'node_pos': st_builder.reshaped_data,
                'edges': st_builder.reshaped_edges,
                'split': data['split']
            }
            for vid_id, data in data_dict.items()
            for st_builder in [SpatioTemporalGraphBuilder(data['node_pos'], self.args)]
        }
        
  
    
class SpatioTemporalGraphBuilder:
    def __init__(self, pos_data, args, inward_edges = None):
        """
        Initialize the graph builder with a fixed number of nodes and a list of inward edges.
        :param num_nodes: Number of nodes in each frame.
        :param inward_edges: List of edges in the format [source, destination].
        """
        
        self.pos_data = pos_data
        self.args = args

        if args.reduce_graph:
            self.pos_data = self._select_evenly_distributed_frames(self.pos_data, num_frames_to_select=self.args.n_frames)

        self.num_features = self.pos_data.shape[0]
        self.num_frames   = self.pos_data.shape[1]
        self.N_NODES    = args.n_nodes
        
        assert self.pos_data.shape[2] == self.N_NODES, "wrong configuration of number of nodes"

        self.tot_number_nodes = self.num_frames * self.N_NODES
        
        # Set inward edges    
        if inward_edges is None:
            ## Default holistic mediapipe edges
            self.inward_edges = [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], 
                                [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], 
                                [7, 15], [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], 
                                [17, 23], [23, 24], [17, 25], [25, 26]]
        else:
            self.inward_edges = inward_edges
        
        self.reshaped_edges = self._build_spatiotemporal_edges()
        #print(self.reshaped_edges.shape)
        self.reshaped_data = self._reshape_nodes()
        #print(self.reshaped_data.shape)
        self.landmark_features = self._build_node_features()
        #print(self.landmark_features.shape)
        #print('------------------------')

    def _select_evenly_distributed_frames(self, tensor, num_frames_to_select=5):
        n_frames = tensor.shape[1]
        
        # Calculate indices 
        indices = np.linspace(0, n_frames - 1, num_frames_to_select, dtype=int)
        
        # Select the frames using the calculated indices
        selected_frames = tensor[:, indices, :]

        return selected_frames
            
    def _reshape_nodes(self):
        return self.pos_data.reshape(self.pos_data.shape[0], -1)
    
    def _build_node_features(self):
        """
        Builds the node features for a given number of frames.
        :param num_frames: The number of frames in the data.
        :return: A tensor with the node features in the graph.
        """
        identity_matrix = np.eye(self.N_NODES)
        landmark_features = np.tile(identity_matrix, (1, self.num_frames))
    
        return torch.tensor(landmark_features)



    def _build_spatiotemporal_edges(self):
        """
        Builds the spatio-temporal edges for a given number of frames.
        :param num_frames: The number of frames in the data.
        :return: A tensor representing the edges in the graph.
        """
        edges = []

        # Adding spatial edges for each frame
        for frame in range(self.num_frames):
            frame_offset = frame * self.N_NODES
            for edge in self.inward_edges:
                n1 = frame_offset + edge[0]
                n2 = frame_offset + edge[1]
                edges.append([n1, n2])
                # Do we need edges both ways?
                #edges.append([dst, src])  

        # Adding temporal edges
        for frame in range(self.num_frames - 1):
            for node in range(self.N_NODES):
                n1 = frame * self.N_NODES + node
                n2 = (frame + 1) * self.N_NODES + node
                edges.append([n1, n2])
                
                #edges.append([dst, src])  # Connecting the node to the next frame

        return torch.tensor(edges).t().contiguous()
    
    




class PyGDataLoader:
    def __init__(self, data, args):
        print('Building dataloader...')
        self.data_dict = data.data_dict
        self.batch_size = args.batch_size
        self.args = args
        if args.temporal_configuration == 'per_frame':
            self.inward_edges = [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], 
                                [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], 
                                [7, 15], [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], 
                                [17, 23], [23, 24], [17, 25], [25, 26]]
            self.edge_index = torch.tensor(self.inward_edges, dtype=torch.long).t().contiguous()
        
        self.build_loaders()


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
            x = data['x']
            if self.args.temporal_configuration == 'spatio_temporal':
                self.edge_index = data['edges']
            data_list.append(Data(pos = pos, x = pos, edge_index= self.edge_index, y=y))
           
        
        print('Number of ' + split + ' points:', len(data_list))
        
        return DataLoader(data_list, batch_size=self.batch_size, shuffle=shuffle)
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ISR Dataset
    parser.add_argument('--root', type=str, default="",
                        help='Data set location')
    parser.add_argument('--root_metadata', type=str, default="subset_metadata.json",
                        help='Metadata json file location')
    parser.add_argument('--root_poses', type=str, default="subset_selection",
                        help='Pose data dir location')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--temporal_configuration', type=str, default="spatio_temporal",
                        help='Temporal configuration of the graph. Options: spatio_temporal, per_frame') 
        ## Graph size parameter
    parser.add_argument('--reduce_graph', type=bool, default=False,
                        help='Whether or not to reduce the graph to a limited number of frames') 
    # TODO: Find a better way to set this number
    parser.add_argument('--n_frames', type=float, default=10,
                        help='Number of frames to use for the spatio temporal graph (max 12)') 
    parser.add_argument('--n_nodes', type=int, default=27,
                        help='Number of nodes to use when reducing the graph - only 27 currently implemented') 
    # Arg parser
    args = parser.parse_args()

    data_dir = os.path.dirname(__file__) + '/' + args.root
    data = ISRDataReader(data_dir, args)

    pyg_loader = PyGDataLoader(data, args)
    pyg_loader.build_loaders()













