"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):

    A = nx.adjacency_matrix(G)
    degrees = np.array([deg for _, deg in G.degree()], dtype=np.float64)
    D_inv = diags(1 / degrees)
    I = eye(len(G))
    Lrw = I - D_inv.dot(A)

    # Eigendecomposition
    eigenvalues, eigenvectors = eigs(Lrw, k=k + 1, which="SM")
    U = eigenvectors[:, 1:].real

    U_normalized = U / np.sqrt(np.sum(U**2, axis=1))[:, np.newaxis]

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(U_normalized)

    nodes = list(G.nodes())
    node_to_cluster = {nodes[i]: cluster_labels[i] for i in range(len(nodes))}

    return node_to_cluster


############## Task 4

file_path = "../datasets/CA-HepTh.txt"  # change this to your path to the dataset
G = nx.read_edgelist(file_path, comments="#", delimiter="\t", create_using=nx.Graph())

# Find gcc
connected_components = list(nx.connected_components(G))
largest_cc = max(connected_components, key=len)
giant_component = G.subgraph(largest_cc)


clusters = spectral_clustering(giant_component, k=50)

unique_clusters = set(clusters.values())
print(f"Number of unique clusters : {len(unique_clusters)}")

# We Display info about the clusters
cluster_sizes = {}
for cluster in clusters.values():
    cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1

print("\nCluster sizes :")
for cluster, size in sorted(cluster_sizes.items()):
    print(f"Cluster {cluster+1}: {size} nodes")


############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):

    m = G.number_of_edges()

    # We convert clustering to a dictionary of cluster nodes
    clusters = {}
    modularity = 0

    for node, cluster in clustering.items():
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(node)

    for cluster_nodes in clusters.values():
        lc = 0
        dc = 0

        for node in cluster_nodes:
            neighbors = set(G.neighbors(node))
            lc += len(neighbors.intersection(cluster_nodes)) / 2
            dc += G.degree(node)

        modularity += (lc / m) - (dc / (2 * m)) ** 2

    return modularity


############## Task 6

G = nx.read_edgelist(file_path, comments="#", delimiter="\t", create_using=nx.Graph())

# Find the gcc
connected_components = list(nx.connected_components(G))
largest_cc = max(connected_components, key=len)
giant_component = G.subgraph(largest_cc)
nodes = list(giant_component.nodes())

spectral_clusters = spectral_clustering(giant_component, k=50)
spectral_modularity = modularity(giant_component, spectral_clusters)
print("Spectral clustering modularity :", spectral_modularity)

random_clusters = {}
for node in nodes:
    random_clusters[node] = randint(1, 50)

random_modularity = modularity(giant_component, random_clusters)
print("Random clustering modularity :", random_modularity)
