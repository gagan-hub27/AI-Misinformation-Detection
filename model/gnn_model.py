import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(GNN, self).__init__()

        # ---------- LAYERS ----------
        self.conv1 = GCNConv(input_dim, 128)
        self.bn1 = BatchNorm(128)

        self.conv2 = GCNConv(128, 64)
        self.bn2 = BatchNorm(64)

        self.conv3 = GCNConv(64, 2)

        self.dropout = 0.5

    def forward(self, x, edge_index):
        # ---------- LAYER 1 ----------
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # ---------- LAYER 2 ----------
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # ---------- RESIDUAL CONNECTION (NEW 🔥) ----------
        x_res = x1[:, :64] + x2   # match dimensions

        # ---------- OUTPUT ----------
        out = self.conv3(x_res, edge_index)

        return F.log_softmax(out, dim=1)