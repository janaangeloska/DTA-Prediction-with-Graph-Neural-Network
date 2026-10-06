import torch
from torch import nn

from GNNNet import DrugTargetNet


class PLMNet(DrugTargetNet):
    def __init__(
        self,
        embedding_dim: int,
        n_output: int = 1,
        num_features_mol: int = 78,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.n_output = n_output
        self._build_drug_branch(num_features_mol, output_dim)

        self.pro_fc1 = nn.Linear(embedding_dim, 1024)
        self.pro_fc2 = nn.Linear(1024, output_dim)

        self._build_head(n_output, output_dim, dropout)

    def encode_protein(self, data_pro: torch.Tensor) -> torch.Tensor:
        xt = self.relu(self.pro_fc1(data_pro))
        xt = self.dropout(xt)
        xt = self.pro_fc2(xt)
        xt = self.dropout(xt)
        return xt
