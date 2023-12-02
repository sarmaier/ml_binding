import glob
import numpy as np
import pickle
from make_interaction_gbsa_block import GbsaInteraction
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.data import DataLoader


def make_pickle(name, info):
    pickle_out = open(name + '.pickle', 'wb')
    pickle.dump(info, pickle_out)
    pickle_out.close()
    return


# Function to generate graph embeddings for a graph using GraphSAGE
def generate_graph_embedding(graph):
    # Convert NetworkX graph to PyTorch Geometric data
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.tensor([graph.nodes[node]['features'] for node in graph.nodes], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    # Define GraphSAGE model
    in_channels = x.size(1)  # Dynamically determine the number of input features
    hidden_channels = 32
    out_channels = 16
    num_layers = 2
    model = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels,
                     out_channels=out_channels, num_layers=num_layers)

    # Generate node embeddings
    model.eval()
    with torch.no_grad():
        x = model(data.x, data.edge_index)
        graph_embedding = global_mean_pool(x, data.batch)

    return graph_embedding.numpy()


# Example usage
if __name__ == "__main__":
    # Example: List of NetworkX graphs with attributes
    files = [x.split("_stripped_complex_ambpdb_08_28_23_ligand.xyz")[0] for x in glob.glob("*ligand.xyz")]
    graph_list = []

    # Create a few example graphs
    for pdb in files:
        print("working on . . . " + str(pdb))

        # call classes to make features for each block
        interaction_block = GbsaInteraction(pdb)
#       edges = interaction_block.edges
        adjacency = interaction_block.adjacency
        nodes = interaction_block.nodes

        # make networkx graph from numpy adjacency matrix
        g = nx.from_numpy_array(adjacency)

        # update dictionary of node attributes with labels
#        att_labels = {0: 'isligand', 1: 'ispositive', 2: 'isnegative', 3: 'ispolar', 4: 'isspecial',
#                      5: 'ishydrophobic', 6: 'vdw', 7: 'electrostatic', 8: 'polar_solv', 9: 'nonpolar_solv',
#                      10: 'total', 11: 'ca_distance'}
#        node_att = {x: {att_labels[i]: node_att[x][i] for i in att_labels} for x in node_att}
#        nx.set_node_attributes(g, node_att)
        node_att = {idx: val for idx, val in enumerate(nodes, start=0)}
        for node in g.nodes:
            g.nodes[node]['features'] = node_att[node]
        graph_list.append(g)

    # Generate graph embeddings for each graph
    embedding_list = [generate_graph_embedding(graph) for graph in graph_list]






    #nx.draw(g)
    #plt.show()
    	
#make_pickle('neural_net_features_09_20_23', feature_dict)
