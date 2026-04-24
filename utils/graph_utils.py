import networkx as nx
import torch


# ---------- BUILD GRAPH ----------
def build_graph(df, max_edges_per_user=15):
    G = nx.Graph()

    # ---------- ADD NODES ----------
    for _, row in df.iterrows():
        emb = row.get("embedding", None)

        if emb is None:
            continue

        try:
            label = int(row["label"])
        except:
            continue

        G.add_node(
            row["post_id"],
            feature=emb,
            label=label
        )

    # ---------- ADD EDGES ----------
    for user in df["user_id"].unique():
        posts = df[df["user_id"] == user]["post_id"].tolist()

        # 🔥 shuffle to avoid bias
        # (important when limiting edges)
        posts = posts[:50]

        for i in range(len(posts)):
            for j in range(i + 1, min(len(posts), i + max_edges_per_user)):
                if posts[i] != posts[j]:
                    G.add_edge(posts[i], posts[j])

    return G


# ---------- CONVERT TO PYTORCH ----------
def convert_to_pytorch(G):
    node_list = list(G.nodes())

    if len(node_list) == 0:
        print("⚠️ Empty graph")
        return None, None, None

    # ---------- NODE INDEX ----------
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # ---------- FEATURES ----------
    features = []
    for n in node_list:
        emb = G.nodes[n]["feature"]

        if isinstance(emb, torch.Tensor):
            features.append(emb)
        else:
            features.append(torch.tensor(emb, dtype=torch.float32))

    try:
        x = torch.vstack(features).float()
    except Exception as e:
        print("Feature stacking error:", e)
        return None, None, None

    # ---------- LABELS ----------
    y = torch.tensor(
        [G.nodes[n]["label"] for n in node_list],
        dtype=torch.long
    )

    # ---------- EDGES ----------
    edge_index = []

    for u, v in G.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]

        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])  # bidirectional

    # ---------- SELF LOOPS ----------
    for i in range(len(node_list)):
        edge_index.append([i, i])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return x, edge_index, y