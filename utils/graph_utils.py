import networkx as nx
import torch

# ---------- BUILD GRAPH ----------
def build_graph(df):
    G = nx.Graph()

    for _, row in df.iterrows():
        if 'embedding' not in row or row['embedding'] is None:
            continue

        G.add_node(
            row['post_id'],
            feature=row['embedding'],
            label=int(row['label'])
        )

    # 🔥 Connect posts from same user
    for user in df['user_id'].unique():
        posts = df[df['user_id'] == user]['post_id'].tolist()

        # Fully connect posts (better than chain 🔥)
        for i in range(len(posts)):
            for j in range(i + 1, len(posts)):
                G.add_edge(posts[i], posts[j])

    return G


# ---------- CONVERT TO PYTORCH ----------
def convert_to_pytorch(G):
    node_list = list(G.nodes())

    # 🔥 Fast mapping
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # ---------- FEATURES ----------
    features = []
    for n in node_list:
        emb = G.nodes[n]['feature']

        # Safety check
        if isinstance(emb, torch.Tensor):
            features.append(emb)
        else:
            features.append(torch.tensor(emb))

    x = torch.cat(features, dim=0).float()

    # ---------- LABELS ----------
    y = torch.tensor(
        [G.nodes[n]['label'] for n in node_list],
        dtype=torch.long
    )

    # ---------- EDGES ----------
    edge_index = []

    for u, v in G.edges():
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index.append([node_to_idx[v], node_to_idx[u]])  # bidirectional

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return x, edge_index, y