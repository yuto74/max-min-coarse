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
import tbdump
import graph_tools

os.environ["DGLBACKEND"] = "pytorch"

FEATURE_DIM = 6
HIDDEN_DIM = 64
N_CLASSES = 2

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
    return h

def expand_graph(g, nvertices):
    """Inflate the number of vertices of graph G to NVERTICES."""
    h = g.copy_graph()
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
        if 'super' in attrs:
            labels.append((1, 0))
        else:
            labels.append((0, 1))
    return features, labels

# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
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
        for n, file in enumerate(glob.glob('data-coarse/*.dot')):
            # Split data for training and testing (i.e., non-training).
            # Reserve 20% for testing.
            if training and n % 5 == 0:
                continue
            if not training and n % 5 != 1:
                continue
            # Prepare graph data in graph-tools format.
            g = import_graph(file)
            # Update vertex features.
            embed_features(g)
            # N_VERTICES is the nuber of vertices in the original graph.
            g_expanded = expand_graph(g, self.nvertices)
            g, labels = create_dgl_and_labels_from_graph(g_expanded)
            self.graphs.append(g)
            self.labels.append(labels)
        # Dataset cannot be empty.
        assert self.graphs
        assert self.labels

def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_label = torch.FloatTensor(labels).view(-1, 2)
    return batched_graph, batched_label

# ----------------------------------------------------------------
class GCN(nn.Module):
    # in_feats: the number of input features.
    # h_feats: the number of hidden features.
    # num_classes: the number of outputs from the model.
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        # Two layer GCNs with internal dimension of h_heats.
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
        self.conv2 = dgl.nn.GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

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
    # Traing the model with batch graphs.
    epoch_losses = []
    for epoch in range(n_epochs):
        epoch_losses = []
        # In every epoch, go through all data for training.
        for n, (bg, labels) in enumerate(data_loader):
            # Forward propagation.
            logits = model(bg, bg.ndata['feat'])
            loss = F.cross_entropy(logits, labels)
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

def test_model(model, testset):
    # Test the accuracy of the model.
    model.eval()
    # Convert a list of tuples to two lists.
    test_bg, test_labels = map(list, zip(*testset))
    # Create a batch graph.
    bg = dgl.batch(test_bg)
    # Vertically concatenate labels for all graphs.
    labels = torch.tensor(test_labels).view(-1, 2)
    # Prediction for test data.
    logits = model(bg, bg.ndata['feat'])
    accuracy = measure_accuray(logits, labels)
    print(f'test_accuray={accuracy}')

def save_model(model, file):
    # Save the trained model.
    torch.save(model.state_dict(), file)

def load_model(model, file):
    # Save the trained model.
    model.load_state_dict(torch.load(file))

def super_resolution_old(model, file, nvertices):
    """直線上にどんどんノードが追加されてしまうバージョン"""
    def by_score(x):
        if x[0] in selected_vertices:
            return 10000
        else:
            return -x[1]

    model.eval()
    # Prepare graph data in graph-tools format.
    g = import_graph(file)
    selected_vertices = []
    while g.nvertices() < nvertices:
        # Update vertex features.
        embed_features(g)
        # FIXME: 「頂点を追加した G を粗視化後のグラフとみなす」が間違い
        g_dgl, labels = create_dgl_and_labels_from_graph(g)
        logits = model(g_dgl, g_dgl.ndata['feat'])
        # Find the vertex with the highest score that has not been attached.
        scores = sorted([(n, logits[n, 0]) for n in range(len(logits))],
                        key=by_score)
        # Now scores[0] has (index, maximum score).
        ui, score = scores[0]
        u = ui + 1
        # Note: the largest vertex identifer is (nvertices + 1).
        vi = g.nvertices()
        v = vi + 1
        g.add_edge(u, v)
        print(f'score={score}\tu={u}\tv={v}')
        selected_vertices.append(ui)

    with open('out.dot', 'w') as f:
        f.write(g.export_dot())

def super_resolution_society(model, file, nvertices):
    """放射状にノードが広がって追加されてしまうバージョン"""
    def by_score(x):
        if x[0] in selected_vertices:
            return -x[1] * .8
        else:
            return -x[1]

    model.eval()
    # Prepare graph data in graph-tools format.
    g = import_graph(file)
    selected_vertices = []
    while g.nvertices() < nvertices:
        # Update vertex features.
        embed_features(g)
        # FIXME: 「頂点を追加した G を粗視化後のグラフとみなす」が間違い
        g_dgl, labels = create_dgl_and_labels_from_graph(g)
        logits = model(g_dgl, g_dgl.ndata['feat'])
        # Find the vertex with the highest score that has not been attached.
        scores = sorted([(n, logits[n, 0]) for n in range(len(logits))],
                        key=by_score)
        while scores and g.nvertices() < nvertices:
            # Now scores[0] has (index, maximum score).
            ui, score = scores[0]
            u = ui + 1
            # Note: the largest vertex identifer is (nvertices + 1).
            vi = g.nvertices()
            v = vi + 1
            if not g.has_edge(u, v):
                g.add_edge(u, v)
                print(f'score={score}\tu={u}\tv={v}')
                selected_vertices.append(ui)
                selected_vertices.append(vi)
            scores = scores[1:]
        print()

    with open('out.dot', 'w') as f:
        f.write(g.export_dot())

def dump_graph(g, scores):
    out = ''
    max_score = max([v for v in scores])
    for v in sorted(g.vertices()):
        r = .01 / max_score * (scores[v - 1])
        out += f'define v{v} ellipse {r} {r} heat80\n'
    for u, v in sorted(g.edges()):
        out += f'define e{u}_{v} link v{u} v{v} 3 heat80\n'
    out += 'alpha /^e/ .7\n'
    out += 'spring -a /^v/\n'
    out += 'display\n'
    print(out)

def super_resolution(model, file, nvertices, discount_factor=.3):
    model.eval()
    # Prepare graph data in graph-tools format.
    g = import_graph(file)
    # Update vertex features.
    embed_features(g)
    g_dgl, labels = create_dgl_and_labels_from_graph(g)
    logits = model(g_dgl, g_dgl.ndata['feat'])

    scores = [0 for _ in range(nvertices)]
    min_score = min([logit[0] for logit in logits])
    for i, logit in enumerate(logits):
        # Use a relative score instead of absolute one since scores might be
        # negative. The minimal value of scores is 1.
        scores[i] = logit[0] - min_score + 1.

    dump_graph(g, scores)

    while g.nvertices() < nvertices:
        # The largest vertex ID is g.nvertices().
        u = g.nvertices() + 1
        g.add_vertex(u)
        level = min(100, 1 + int(u / 3))
        print(f'define v{u} ellipse .007 .007 heat{level}')

        # Randomly choose a node to connect with the probability proportional to their scores.
        v1 = random.choices(
            range(g.nvertices()), weights=scores[:g.nvertices()], k=1)[0] + 1
        g.add_edge(u, v1)
        # Adjust scores to distribute links to other nodes.
        scores[v1 - 1] -= scores[v1 - 1] * discount_factor
        scores[u - 1] += scores[v1 - 1] * discount_factor
        print(f'define e{u}_{v1} link v{u} v{v1} 3 heat{level}')

        if random.random() < .25:
            v2 = random.choices(
            range(g.nvertices()), weights=scores[:g.nvertices()], k=1)[0] + 1
            g.add_edge(u, v2)
            # Adjust scores to distribute links to other nodes.
            scores[v2 - 1] -= scores[v2 - 1] * discount_factor
            scores[u - 1] += scores[v2 - 1] * discount_factor
            print(f'define e{u}_{v2} link v{u} v{v2} 3 heat{level}')

        print('alpha /^e/ .7')
        print('spring -a /^v/')
        print('display')

    with open('out.dot', 'w') as f:
        f.write(g.export_dot())

def main():
    opt = getopts('vtei:n:o:r:') or usage()
    verbose = opt.v
    run_training = opt.t
    run_eval = opt.e
    nvertices = int(opt.n) if opt.n else 25
    load_model_file = opt.i
    save_model_file = opt.o
    run_super_res = opt.r

    # Create a model.
    model = GCN(FEATURE_DIM, HIDDEN_DIM, N_CLASSES)
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
    if run_super_res:
        file = run_super_res
        super_resolution(model, file, nvertices)

if __name__ == "__main__":
    main()
