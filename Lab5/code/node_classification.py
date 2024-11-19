"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt


# Loads the karate network
G = nx.read_weighted_edgelist(
    "../data/karate.edgelist", delimiter=" ", nodetype=int, create_using=nx.Graph()
)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt("../data/karate_labels.txt", delimiter=",", dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i, 0]] = class_labels[i, 1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network


def visualize(G, y):
    # Define colors for the nodes based on their class labels
    color_map = []
    for label in y:
        if label == 0:
            color_map.append("red")
        else:
            color_map.append("blue")

    pos = nx.spring_layout(G)
    nx.draw_networkx(
        G, pos, node_color=color_map, with_labels=True, node_size=500, font_size=10
    )
    plt.title("Karate Network Visualization")
    plt.show()


# Call the visualization function
visualize(G, y)


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i, :] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[: int(0.8 * n)]
idx_test = idx[int(0.8 * n) :]

X_train = embeddings[idx_train, :]
X_test = embeddings[idx_test, :]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"DeepWalk Embeddings Accuracy: {accuracy:.2f}")

############## Task 8
# Generates spectral embeddings

A = nx.adjacency_matrix(G)
D = diags(np.array(A.sum(axis=1)).flatten())
D_inv = diags(1 / np.array(A.sum(axis=1)).flatten())
L_rw = eye(n) - D_inv @ A

eigenvalues, eigenvectors = eigs(L_rw, k=2, which="SR")

spectral_embeddings = eigenvectors.real

X_train_spectral = spectral_embeddings[idx_train, :]
X_test_spectral = spectral_embeddings[idx_test, :]

clf_spectral = LogisticRegression(max_iter=1000)
clf_spectral.fit(X_train_spectral, y_train)

y_pred_spectral = clf_spectral.predict(X_test_spectral)
accuracy_spectral = accuracy_score(y_test, y_pred_spectral)
print(f"Spectral embeddings accuracy : {accuracy_spectral}")
