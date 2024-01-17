from dataset_isr import ISRDataReader
from dataset_isr import PyGDataLoader
import os    
import imageio
import networkx as nx
import matplotlib
import numpy as np 
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser()
# ISR Dataset
parser.add_argument('--root', type=str, default="",
                    help='Data set location')
parser.add_argument('--root_metadata', type=str, default="wlasl_new.json",
                    help='Metadata json file location')
parser.add_argument('--root_poses', type=str, default="wlasl_poses_pickle",
                    help='Pose data dir location')
parser.add_argument('--batch_size', type=int, default=5,
                    help='Batch size. Does not scale with number of gpus.')
parser.add_argument('--temporal_configuration', type=str, default="spatio_temporal",
                    help='Temporal configuration of the graph. Options: spatio_temporal, per_frame') 

# Train settings
parser.add_argument('--train_augm', type=eval, default=True,
                    help='whether or not to use random rotations during training')

# ISR Dataset
parser.add_argument('--n_classes', type=str, default=2000,
                    help='Number of sign classes')

parser.add_argument('--reduce_graph', type=bool, default=False,
                    help='') 
parser.add_argument('--n_frames', type=int, default=10,
                    help='Number of frames to use for the spatio temporal graph (max 12)') 

# Arg parser
args = parser.parse_args()

data_dir = os.path.dirname(__file__) + '/' + args.root
data = ISRDataReader(data_dir, args)

pyg_loader = PyGDataLoader(data, args)
pyg_loader.build_loaders()

if args.temporal_configuration == 'per_frame':
    inward_edges = [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], 
                    [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], 
                    [7, 15], [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], 
                    [17, 23], [23, 24], [17, 25], [25, 26]]


def build_png(filename = '11327_f_41'):
    x, y = pyg_loader.data_dict[filename]['node_pos']
    # Convert x and y to a dictionary of node positions
    
    positions = {i: (-x[i], -y[i]) for i in range(len(x))}

    # importing networkx 
    G = nx.Graph()
    # Add nodes and edges
    G.add_nodes_from(positions.keys())
    if args.temporal_configuration == 'per_frame':
        G.add_edges_from(inward_edges)
    else:
        print(pyg_loader.data_dict[filename]['edges'])
        reshaped_tensor = pyg_loader.data_dict[filename]['edges'].view(2, -1)

        # Convert the reshaped tensor to a list of tuples
        edges_list = [(int(src), int(dst)) for src, dst in zip(reshaped_tensor[0], reshaped_tensor[1])]
        G.add_edges_from(edges_list)
    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title("Connected Graph")
    plt.savefig('img/' + filename +'.png')

#build_png(filename = '11327')

def build_gif(vid_id = '13645'):
    frame_ids = [s for s in list(data.data_dict.keys()) if s.startswith(vid_id)]
    file_names = []

    for frame in frame_ids:
        x, y = pyg_loader.data_dict[frame]['node_pos']
        positions = {i: (x[i], y[i]) for i in range(len(x))}
    # Create and draw the graph
    G = nx.Graph()
    G.add_nodes_from(positions.keys())
    G.add_edges_from(inward_edges)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title(f"Connected Graph - {frame}")

    # Save the plot to a file
    file_name = f'graph_{frame}.png'
    plt.savefig(file_name)
    plt.close()

    # Append the file name to the list
    file_names.append(file_name)
    # Create a GIF from the saved images
    with imageio.get_writer('img/'+vid_id + '.gif', mode='I') as writer:
        for filename in file_names:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Optionally, delete the individual frame images
    for filename in file_names:
        os.remove(filename)

#build_gif()
        
def build_histogram(data_dict, dimension=1, num_bins=20, save_path='img/hist.png'):
    # Create an empty list to store the shapes of 'node_pos' entries
    shapes = []

    # Iterate through the dictionary
    for vid_id, data in data_dict.items():
        
        shapes.append(data['n_frames'])

    # Convert the list of shapes into a NumPy array for easier processing
    shapes = np.array(shapes)

    # Build a histogram of the specified dimension's shape
    plt.hist(shapes, bins=num_bins)
    plt.xlabel(f'Dimension {dimension} Shape')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of number of frames per vid')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

build_histogram(data.data_dict)