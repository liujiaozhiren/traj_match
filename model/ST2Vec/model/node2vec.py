import numpy as np
import torch
from torch_geometric.nn import Node2Vec

import cfg
from model.ST2Vec.model.time2vec import proc_timeseq_embedding_nodeseq
from model.ST2Vec.model.traj2node import build_graph_from_trajs

device = cfg.device


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch(model, loader, optimizer):
    # Training with epoch iteration
    last_loss = 1
    print("Training node embedding with node2vec...")
    for i in range(100):
        loss = train(model, loader, optimizer)
        print('Epoch: {0} \tLoss: {1:.4f}'.format(i, loss))
        if abs(last_loss - loss) < 1e-5:
            break
        else:
            last_loss = loss


def read_graph(dataset):
    edge_df, node_df, edge_index, mapper = build_graph_from_trajs(dataset)
    # edge_index = edge_index.to_numpy()
    num_node = node_df.size
    return edge_index, num_node, mapper, edge_df

def infer_embeddings(model, num_nodes, device):
    model.eval()
    with torch.no_grad():
        node_features = model(torch.arange(num_nodes, device=device)).cpu().numpy()
    return node_features


def get_node_embedding(edge_index, num_node, device=cfg.device):
    # edge_index, num_node, mapper = read_graph(traj_list)
    edge_index = torch.LongTensor(edge_index).contiguous().to(device)
    model = Node2Vec(
        edge_index,
        embedding_dim=cfg.dim,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
        num_nodes=num_node
    ).to(device)
    embeddings = infer_embeddings(model, num_node, device)
    return embeddings



def load_netowrk(dataset):
    """
    load road network from file with Pytorch geometric data object
    :param dataset: the city name of road network
    :return: Pytorch geometric data object of the graph
    """

    edge_path = "./data/" + dataset + "/road/edge_weight.csv"
    node_embedding_path = "./data/" + dataset + "/node_features.npy"

    node_embeddings = np.load(node_embedding_path)
    df_dege = pd.read_csv(edge_path, sep=',')

    edge_index = df_dege[["s_node", "e_node"]].to_numpy()
    edge_attr = df_dege["length"].to_numpy()

    edge_index = torch.LongTensor(edge_index).t().contiguous()
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)



    road_network = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)

    return road_network