import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import (
    GCNConv,
)
from torch_geometric.nn import (
    global_mean_pool as gep,
)


class DrugTargetNet(torch.nn.Module):
    """Drug graph branch and affinity head shared by every protein representation.

    Subclasses build the drug branch, then their protein branch, then the head, in that
    order, so parameter initialization and state_dict keys follow the original GNNNet.
    """

    def _build_drug_branch(self, num_features_mol: int, output_dim: int) -> None:
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

    def _build_head(self, n_output: int, output_dim: int, dropout: float) -> None:
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)

    def encode_drug(self, data_mol: Batch) -> torch.Tensor:
        mol_x, mol_edge_index, mol_batch = (
            data_mol.x,
            data_mol.edge_index,
            data_mol.batch,
        )

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        x = gep(x, mol_batch)

        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)
        return x

    def encode_protein(self, data_pro: Batch | torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, data_mol: Batch, data_pro: Batch | torch.Tensor) -> torch.Tensor:
        x = self.encode_drug(data_mol)
        xt = self.encode_protein(data_pro)

        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class GNNNet(DrugTargetNet):
    def __init__(
        self,
        n_output: int = 1,
        num_features_pro: int = 33,
        num_features_mol: int = 78,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super(GNNNet, self).__init__()

        print("GNNNet Loaded")
        self.n_output = n_output
        self._build_drug_branch(num_features_mol, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self._build_head(n_output, output_dim, dropout)

    def encode_protein(self, data_pro: Batch) -> torch.Tensor:
        target_x, target_edge_index, target_batch = (
            data_pro.x,
            data_pro.edge_index,
            data_pro.batch,
        )

        xt = self.pro_conv1(target_x, target_edge_index)
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
