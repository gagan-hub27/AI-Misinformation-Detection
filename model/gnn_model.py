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

        # ---------- RESIDUAL PROJECTION ----------
        self.res_proj = torch.nn.Linear(128, 64)

        # ---------- DROPOUT ----------
        self.dropout = torch.nn.Dropout(p=0.4)  # 🔥 slightly reduced

        # ---------- NORMALIZATION (🔥 NEW) ----------
        self.layer_norm = torch.nn.LayerNorm(64)

        # ---------- INITIALIZATION ----------
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        if x is None or edge_index is None:
            return None

        # ---------- LAYER 1 ----------
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        # ---------- LAYER 2 ----------
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)

        # ---------- RESIDUAL CONNECTION (IMPROVED 🔥) ----------
        res = self.res_proj(x1)
        x_res = res + x2

        # 🔥 Stabilization
        x_res = self.layer_norm(x_res)
        x_res = F.relu(x_res)

        # ---------- OUTPUT ----------
        out = self.conv3(x_res, edge_index)

        return F.log_softmax(out, dim=1)