import networkx as nx


from model.trajGAT.node2vec.node2vec_mdl import learn_embeddings, Graph


class Parameter:
    def __init__(self, d_model) -> None:
        # self.output = "qtree.emb"
        self.dimensions = d_model
        self.walk_length = 80
        self.num_walks = 10
        self.window_size = 10
        self.iter = 1
        self.workers = 8
        self.p = 1
        self.q = 1
        self.weighted = False
        self.unweighted = True
        self.directed = False
        self.undirected = True



def node2vec_embed(id_edge_list, d_model):
    """
	Pipeline for representational learning for all nodes in a graph.
	"""
    node2vec_args = Parameter(d_model)
    nx_G = read_graph(id_edge_list, node2vec_args)
    G = Graph(nx_G, node2vec_args.directed, node2vec_args.p, node2vec_args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(node2vec_args.num_walks, node2vec_args.walk_length)

    all_vectors = learn_embeddings(walks, node2vec_args)

    return all_vectors


def read_graph(id_edge_list, node2vec_args):
    """
	Reads the input network in networkx.
	"""
    if node2vec_args.weighted:
        # G = nx.read_edgelist(input_path, nodetype=int, data=(("weight", float),), create_using=nx.DiGraph())
        G = nx.DiGraph(id_edge_list)  # 这里的 id_edge_list 需要带有权重
    else:
        G = nx.DiGraph(id_edge_list)

        for edge in G.edges():
            G[edge[0]][edge[1]]["weight"] = 1

    if not node2vec_args.directed:
        G = G.to_undirected()

    return G