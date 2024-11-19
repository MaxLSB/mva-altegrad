"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
G = nx.read_edgelist(
    "../datasets/CA-HepTh.txt",
    comments="#",
    delimiter="\t",
    create_using=nx.Graph(),
)
# change the path if needed

# Compute & print network characteristics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f"Number of nodes : {num_nodes}")
print(f"Number of edges : {num_edges}")


############## Task 2
connected_components = list(nx.connected_components(G))
num_components = len(connected_components)
print(f"Number of connected components : {num_components}")

largest_cc = max(connected_components, key=len)
giant_component = G.subgraph(largest_cc)


gcc_num_nodes = giant_component.number_of_nodes()
gcc_num_edges = giant_component.number_of_edges()
node_fraction = gcc_num_nodes / num_nodes
edge_fraction = gcc_num_edges / num_edges

print(f"Nodes in GCC : {gcc_num_nodes} ({node_fraction})")
print(f"Edges in GCC : {gcc_num_edges} ({edge_fraction})")
