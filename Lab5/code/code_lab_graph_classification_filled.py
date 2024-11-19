"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7


# load Mutag dataset
def load_dataset():

    dataset = TUDataset(root="../datasets", name="MUTAG")

    Gs = [to_networkx(data, to_undirected=True) for data in dataset]

    y = [data.y.item() for data in dataset]

    return Gs, y


Gs, y = load_dataset()

# Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(
    Gs, y, test_size=0.2, random_state=42
)


# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):
    all_paths = dict()
    sp_counts_train = dict()

    for i, G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    sp_counts_test = dict()

    for i, G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i, all_paths[length]] = sp_counts_train[i][length]

    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i, all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]

    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0, 1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0, 1)
    graphlets[2].add_edge(1, 2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0, 1)
    graphlets[3].add_edge(1, 2)
    graphlets[3].add_edge(0, 2)

    phi_train = np.zeros((len(G_train), 4))

    for i, G in enumerate(Gs_train):
        nodes = list(G.nodes())

        for _ in range(n_samples):

            if len(nodes) < 3:
                continue
            sampled_nodes = np.random.choice(nodes, 3, replace=False)

            subgraph = G.subgraph(sampled_nodes)

            for j, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(subgraph, graphlet):
                    phi_train[i, j] += 1

        # We normalize
        phi_train[i, :] /= n_samples

    phi_test = np.zeros((len(G_test), 4))

    # Compute feature map for test graphs
    for i, G in enumerate(Gs_test):
        nodes = list(G.nodes())

        for _ in range(n_samples):

            if len(nodes) < 3:
                continue

            sampled_nodes = np.random.choice(nodes, 3, replace=False)
            subgraph = G.subgraph(sampled_nodes)

            for j, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(subgraph, graphlet):
                    phi_test[i, j] += 1

        # We normalize
        phi_test[i, :] /= n_samples

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)


############## Task 9

K_train_gl, K_test_gl = graphlet_kernel(G_train, G_test)

clf_gl = SVC(kernel="precomputed")
clf_gl.fit(K_train_gl, y_train)

y_pred_gl = clf_gl.predict(K_test_gl)
accuracy_gl = accuracy_score(y_test, y_pred_gl)
print(f"Graphlet kernel accuracy : {accuracy_gl:.4f}")


############## Task 10

clf_sp = SVC(kernel="precomputed")
clf_sp.fit(K_train_sp, y_train)

y_pred_sp = clf_sp.predict(K_test_sp)

accuracy_sp = accuracy_score(y_test, y_pred_sp)
print(f"Shortest path kernel accuracy : {accuracy_sp:.4f}")


clf_gl = SVC(kernel="precomputed")
clf_gl.fit(K_train_gl, y_train)

y_pred_gl = clf_gl.predict(K_test_gl)
accuracy_gl = accuracy_score(y_test, y_pred_gl)
print(f"Graphlet kernel accuracy : {accuracy_gl:.4f}")
