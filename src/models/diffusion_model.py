import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    """
    Implementation of the Strudel-Model
    Featuring timestep-encoding and embedding bottleneck
    """
    def __init__(self, embed_dim=896, d_model=114, dim_feedforward=1024, nhead=12, num_layers=6, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.d_model = d_model

        self.timestep_projector = nn.Linear(d_model, d_model)
        self.embed_projector = nn.Linear(embed_dim, d_model)

        self.model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, activation='gelu'), num_layers=num_layers)
        
        self.output_projector = nn.Linear(d_model, embed_dim)
        

    def timestep_encoding(self, t, time_dim):
        inv_freq = 1. / (10000 ** (torch.arange(0, time_dim, 2, device=self.device).float() / time_dim))
        t_enc_a = torch.sin(t.repeat(1, time_dim // 2) * inv_freq)
        t_enc_b = torch.cos(t.repeat(1, time_dim // 2) * inv_freq)
        t_enc = torch.cat([t_enc_a, t_enc_b], dim=-1)
        return t_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float32) # Create features dimension which will be repeated later
        t_enc = self.timestep_encoding(t, self.d_model)
        t_enc = self.timestep_projector(t_enc)
        t_enc = t_enc.unsqueeze(0).repeat(x.shape[0], 1, 1)


        x = self.embed_projector(x)

        x = x + t_enc
        x = self.model(x)

        x = self.output_projector(x)
 
        return x

if __name__ == "__main__":
    d = DiffusionModel(10)
    x = torch.randn(5, 2, 10)
    t = torch.LongTensor([4]*x.shape[1])
    out = d(x, t)
    print(out.shape)

