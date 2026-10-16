import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gep

from src.structure.gnn_net import DrugTargetNet

RESIDUE_PROJECTION_DIM = 64


class HybridNet(DrugTargetNet):
    """GNNNet with projected per-residue language model features appended to every node."""

    def __init__(
        self,
        embedding_dim: int,
        n_output: int = 1,
        num_features_pro: int = 33,
        num_features_mol: int = 78,
        output_dim: int = 128,
        dropout: float = 0.2,
        residue_projection_dim: int = RESIDUE_PROJECTION_DIM,
    ) -> None:
        super().__init__()

        self.n_output = n_output
        self._build_drug_branch(num_features_mol, output_dim)

        self.pro_residue_proj = nn.Linear(embedding_dim, residue_projection_dim)
        node_dim = num_features_pro + residue_projection_dim
        self.pro_conv1 = GCNConv(node_dim, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)

        self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)

        self._build_head(n_output, output_dim, dropout)

    def encode_protein(self, data_pro: Batch) -> torch.Tensor:
        target_edge_index, target_batch = data_pro.edge_index, data_pro.batch

        residues = self.relu(self.pro_residue_proj(data_pro.residue_emb.float()))
        xt = torch.cat((data_pro.x, residues), 1)

        xt = self.pro_conv1(xt, target_edge_index)
        xt = self.relu(xt)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        xt = gep(xt, target_batch)

        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)
        return xt
