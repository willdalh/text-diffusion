import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        

    def timestep_encoding(self, t, time_dim):
        inv_freq = 1. / (10000 ** (torch.arange(0, time_dim, 2, device=self.device).float() / time_dim))
        t_enc_a = torch.sin(t.repeat(1, time_dim // 2) * inv_freq)
        t_enc_b = torch.cos(t.repeat(1, time_dim // 2) * inv_freq)
        t_enc = torch.cat([t_enc_a, t_enc_b], dim=-1)
        return t_enc
