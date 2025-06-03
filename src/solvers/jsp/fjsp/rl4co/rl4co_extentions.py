import torch
import torch.nn as nn

class MultiEncoder(nn.Module):
    def __init__(self, encoder_1, encoder_2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2

    def forward(self, td):

        hidden_1, _ = self.encoder_1(td)

        hidden_2, _ = self.encoder_2(td)

        hidden = (hidden_1[0] + hidden_2[0], hidden_1[1] + hidden_2[1])

        return hidden, None
    

class AdditionalMachineInfoInitEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        feature_name: str,
        feature_dim: int = 1,
        linear_bias: bool = True,
    ):
        super().__init__()
        self.feature_name = feature_name
        self.embed_dim = embed_dim
        self.init_embed = nn.Linear(feature_dim, embed_dim, linear_bias)

    def forward(self, td):
        bs, n_ops = td["is_ready"].shape
        ops_emb = torch.randn(size=(bs, n_ops, self.embed_dim), device=td.device)
        ma_emb = self.init_embed(td[self.feature_name].unsqueeze(2))
        n_machines = ma_emb.size(1)
        edge_emb = torch.randn(size=(bs, n_ops, n_machines, self.embed_dim), device=td.device)
        edges = td["ops_ma_adj"].transpose(1, 2).to(td.device)

        return ops_emb, ma_emb, edge_emb, edges
    
