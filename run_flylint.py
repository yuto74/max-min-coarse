#!/usr/bin/env python3
#
# A GCN model for estimating supernodes in graph coarseening.
# Copyright (c) 2023, Hiroyuki Ohsaki.
# All rights reserved.
#
# $Id: run.py,v 1.13 2023/11/09 10:16:00 ohsaki Exp $
#
import glob
import os
import random
import statistics
import sys

from perlcompat import die, warn, getopts
#import tbdump
import graph_tools

os.environ["DGLBACKEND"] = "pytorch"

FEATURE_DIM = 6
HIDDEN_DIM = 64
N_CLASSES = 10

FILES = {}

def usage():
    die(f"""\
usage: {sys.argv[0]} [-vte] [-i file] [-o file] [-r dot-file]
  -v           verbose mode
  -t           train the model with training data
  -e           evalaute the model accuracy with test data
  -i file      load the trained model from FILE
  -o file      store the trained model as FILE
  -r dot-file  run super resolution for graph DOT-FILE
""")

import dgl
import dgl.nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

def import_graph(file):
    """Load a graph in DOT format from FILE."""
    g = graph_tools.Graph(directed=False)
    with open(file) as f:
        lines = f.readlines()
        g.import_dot(lines)
    # Renumber vertex IDs to make sure those are consecutive.
    h = graph_tools.Graph(directed=False)
    for v in g.vertices():
        vi = g.vertex_index(v)
        h.add_vertex(vi + 1)
    for u, v in g.edges():
        ui = g.vertex_index(u)
        vi = g.vertex_index(v)
        h.add_edge(ui + 1, vi + 1)

    with open(file) as f:
        if 'original degree distribution' in lines[0]:
            import re
            import ast
            degree = re.sub("// original degree distribution, ", "" , lines[0])
            degree = ast.literal_eval(degree)

    h.orig_degree = degree
    return h

def expand_graph(g, nvertices):
    """Inflate the number of vertices of graph G to NVERTICES."""
    h = g.copy_graph()
    h.orig_degree = g.orig_degree
    for v in range(1, nvertices + 1):
        h.add_vertex(v)
    return h

def parse_super_attribute(astr):
    """Parse a string of the `super' attribute and return a set of
    vertices."""
    # Example: '+1+7+12+4' -> {1, 7, 12, 4}
    vertices = set()
    if astr is None:
        return vertices
    for v in astr.split('+'):
        if not v:
            continue
        vertices.add(int(v))
    return vertices

def embed_features(g):
    """Record several vertex properties and metrics as its attribute."""
    for v in g.vertices():
        g.set_vertex_attribute(v, 'degree', g.degree(v))
        g.set_vertex_attribute(v, 'degreecent', g.degree_centrality(v))
        g.set_vertex_attribute(v, 'closeness', g.eigenvector_centrality(v))
        g.set_vertex_attribute(v, 'eigenvec', g.eigenvector_centrality(v))
        g.set_vertex_attribute(v, 'eccentr', g.eigenvector_centrality(v))
        g.set_vertex_attribute(v, 'ntriad', g.ntriads(v))

def compose_feature_and_label(g):
    """Compose matrices representing features and labels of all vertices in
    graph G."""
    features = []
    labels = []
    for v in g.vertices():
        # Construct a node feature vector.
        vec = []
        # Retrieve a dict for vertex attributes for better readability.
        attrs = g.get_vertex_attributes(v)
        for name in [
                'degree', 'degreecent', 'closeness', 'eigenvec', 'eccentr',
                'ntriad'
        ]:
            vec.append(attrs.get(name, 0))
        features.append(vec)
        # Node label is a one-hot vector indicating supernode.

    for i in range(1, N_CLASSES + 1):
        try:
            labels.append(float(g.orig_degree[i]))
        except:
            labels.append(.0)

    return features, labels

# # ----------------------------------------------------------------
def convert_to_dgl(g):
    """Convert a graph G in graph-tools format to DGL graph format."""
    h = dgl.DGLGraph()
    # NOTE: This code assumes vertices start from 1 and are conscutive.
    h.add_nodes(g.nvertices())
    for u, v in g.edges():
        # NOTE: DGL graph is always directional.
        ui = g.vertex_index(u)
        vi = g.vertex_index(v)
        h.add_edges([ui], [vi])
        h.add_edges([vi], [ui])
    # Note: GCN needs self loops for isolated nodes.
    h = dgl.add_self_loop(h)
    return h

def create_dgl_and_labels_from_graph(g):
    features, labels = compose_feature_and_label(g)
    # Conver to DGL format.
    g_dgl = convert_to_dgl(g)
    g_dgl.ndata['feat'] = torch.FloatTensor(features)
    return g_dgl, labels


# # ----------------------------------------------------------------
class MyDataset(object):
    def __init__(self, training, nvertices):
        super().__init__()
        self.nvertices = nvertices  # the size of the original graph
        self.graphs = []  # training/test graphs in DGL format.
        self.labels = []  # training/test labels as float tensor.
        # FIXME: Graphs preparation should be delayed for less memory consumption.
        self._load_all_data(training)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    @property
    def num_classes(self):
        return N_CLASSES

    def _load_all_data(self, training):
        i = 0
        for n, file in enumerate(glob.glob('data-coarse/*.dot')):
            # Split data for training and testing (i.e., non-training).
            # Reserve 20% for testing.
            if training and n % 5 == 0:
                continue
            if not training and n % 5 != 1:
                continue
            FILES[i] = file
            # Prepare graph data in graph-tools format.
            g = import_graph(file)
            # Update vertex features.
            embed_features(g)
            # N_VERTICES is the nuber of vertices in the original graph.
            g_expanded = g
            g, labels = create_dgl_and_labels_from_graph(g_expanded)
            self.graphs.append(g)
            self.labels.append(labels)
            i += 1
        # Dataset cannot be empty.
        assert self.graphs
        assert self.labels

def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_label = torch.FloatTensor(labels)
    return batched_graph, batched_label

# # ----------------------------------------------------------------
class GCN(nn.Module):
    # in_feats: the number of input features.
    # h_feats: the number of hidden features.
    # num_classes: the number of outputs from the model.
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        # Two layer GCNs with internal dimension of h_heats.
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
        self.conv2 = dgl.nn.GraphConv(h_feats, h_feats)
        self.linear = nn.Linear(h_feats, num_classes) # 全結合層
        self.pooling = dgl.nn.AvgPooling()

    def forward(self, g, in_feat):
        # 2 層 GCN
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)

        # pooling
        hg = self.pooling(g, h)
        logits = self.linear(hg)

        out = F.softmax(logits, dim=1)

        return out

def measure_accuray(logits, labels):
    # Prediction.
    pred = logits.argmax(1)
    # Ground truth.
    truth = labels.argmax(1)
    # The ratio of correct predictions.
    accuracy = (pred == truth).float().mean()
    return accuracy

def train_model(model, trainset, n_epochs=100):
    model.train()
    # Use PyTorch's DataLoader and the collate function defined before.
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=2,
                                              shuffle=True,
                                              collate_fn=collate)
    optimizer = optim.Adam(model.parameters(), lr=.01)
    loss_fn = nn.MSELoss()

    # Traing the model with batch graphs.
    epoch_losses = []
    for epoch in range(n_epochs):
        epoch_losses = []
        # In every epoch, go through all data for training.
        for n, (bg, labels) in enumerate(data_loader):
            # Forward propagation.
            logits = model(bg, bg.ndata['feat'])
            loss = loss_fn(logits, labels)
            accuracy = measure_accuray(logits, labels)
            
            # Backward propagation.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record the loss as Python number (item) after copying (detach).
            current_loss = loss.detach().item()
            epoch_losses.append(current_loss)
        avg_loss = statistics.mean(epoch_losses)
        print(f'epoch={epoch}\tloss={avg_loss:.4f}\taccuracy={accuracy:.4f}')

def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def test_model(model, testset):
    # Test the accuracy of the model.
    model.eval()
    # Convert a list of tuples to two lists.
    test_bg, test_labels = map(list, zip(*testset))
    # Create a batch graph.
    bg = dgl.batch(test_bg)
    # Vertically concatenate labels for all graphs.
    labels = torch.tensor(test_labels)
    # Prediction for test data.
    logits = model(bg, bg.ndata['feat'])

    # for i, (logit_vec, label_vec) in enumerate(zip(logits, labels)):
    #     print(f"[Sample {i}], file:{FILES[i]}")
    #     print("  Predicted:", logit_vec.detach().cpu().numpy())
    #     print("  Actual   :", label_vec.detach().cpu().numpy())            

    degree_dist = True
    r2 = False
    from collections import defaultdict
    if degree_dist:
        pred_sum = {i:.0 for i in range(1, 11)}
        true_sum = {i:.0 for i in range(1, 11)}
        # pred_sum = defaultdict(int)
        # true_sum = defaultdict(int)
        count = 0
        for i, (logit_vec, label_vec) in enumerate(zip(logits, labels)):
            for n, logit in enumerate(logit_vec.detach().cpu().numpy()):
                pred_sum[n + 1] += logit
            count += 1
                
        for i, (logit_vec, label_vec) in enumerate(zip(logits, labels)):
            for n, label in enumerate(label_vec.detach().cpu().numpy()):
                true_sum[n + 1] += label

        # print('option: set style data linespoints')
        # print("option: set xlabel \"degree\"")
        # print("option: set ylabel \"P(k)\"")
        # print('option: set xrange [0:10]')
        # print('option: set yrange [0:1]')
        print('name: average true')
        for n, true in enumerate(true_sum.values()):
            print(n + 1, true / count)
        
        print('name: average GNN')
        for n, pred in enumerate(pred_sum.values()):
            print(n + 1, pred / count)


    # # R2 score
    if r2:
        print('option: set xrange [0:]')
        print('option: set yrange [0:]')
        print('option: set style data points')
        print('name: R2 score')
        for i, (logit_vec, label_vec) in enumerate(zip(logits, labels)):
            print(i + 1, float(r2_score(label_vec, logit_vec).detach()))
    

def save_model(model, file):
    # Save the trained model.
    torch.save(model.state_dict(), file)

def load_model(model, file):
    # Save the trained model.
    model.load_state_dict(torch.load(file))

def run_degree(model, file, nvertices):

    model.eval()
    g = import_graph(file)
    embed_features(g)
    #g_expanded = expand_graph(g, nvertices)
    g_dgl, labels = create_dgl_and_labels_from_graph(g)
    logits = model(g_dgl, g_dgl.ndata['feat'])

    print('option: set style data linespoints')
    print('option: set xrange [0:]')
    print('option: set yrange [0:]')

    _logits = logits.detach().cpu().numpy()
    print('name: gnn')
    for n in range(0, 10):
        print(n+1, _logits[0][n])

    print('name: actual')
    for n in range(0, 10):
        print(n+1, labels[n])
        


def main():
    opt = getopts('vtei:n:o:r:') or usage()
    verbose = opt.v
    run_training = opt.t
    run_eval = opt.e
    nvertices = int(opt.n) if opt.n else 25
    load_model_file = opt.i
    save_model_file = opt.o
    run_degree_estimation = opt.r

    # Create a model.
    model = GCN(FEATURE_DIM, HIDDEN_DIM, N_CLASSES)
    #data = MyDataset(True, nvertices)

    if run_training:
        # Create training set.
        trainset = MyDataset(True, nvertices)
        train_model(model, trainset)
    if load_model_file:
        file = load_model_file
        load_model(model, file)
    if save_model_file:
        file = save_model_file
        save_model(model, file)
    if run_eval:
        # Create test set.
        testset = MyDataset(False, nvertices)
        test_model(model, testset)
    if run_degree_estimation:
        file = run_degree_estimation
        run_degree(model, file, nvertices)

if __name__ == "__main__":
    main()