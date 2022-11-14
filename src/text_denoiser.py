import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
import torchtext.data.utils as ttdutils

from text_dataset import TextDataset

class TextDenoiser(nn.Module):
    def __init__(self, betas = (1e-4, 0.02), n_T = 1000, lr=3e-4, embed_dim=768):
        super(TextDenoiser, self).__init__()
        
        self.betas = betas
        self.n_T = n_T
        self.criterion = nn.MSELoss()
        self._load_schedule(self._get_schedule(*betas, n_T))


        train_iter = WikiText2(root="./data", split="train")
        tokenizer = ttdutils.get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        self.vocab = vocab
        data = [torch.LongTensor([vocab(tokenizer(item))]) for item in train_iter]
        data = tuple(filter(lambda x: x.numel() > 0, data))
        data = torch.cat(data, dim=1).squeeze(0)[:70000]
        dataset = TextDataset(data, seq_len=64)
        self.dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)

        self.embed_dim = embed_dim
        self.embedder = nn.Embedding(len(vocab), embed_dim)
        # self.model = nn.Transformer(d_model=embed_dim, nhead=12, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=3072, dropout=0.1, activation='gelu')s
        self.model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, dim_feedforward=3072, dropout=0.1, activation='gelu'), num_layers=6)
        self.decoder = nn.Linear(embed_dim, len(vocab))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.to()
        
        
    def _get_schedule(self, beta_start: float, beta_end: float, n_T: int):
        beta = torch.linspace(beta_start, beta_start, n_T + 1)
        sqrt_beta = torch.sqrt(beta)
        
        alpha = 1 - beta
        alphabar = torch.cumprod(alpha, dim=0)
        sqrt_alphabar = torch.sqrt(alphabar)
        sqrt_m_alphabar = torch.sqrt(1 - alphabar)
        oneover_sqrt_alpha = 1 /  torch.sqrt(alpha)
        malpha_over_sqrtmab = (1 - alpha) / sqrt_m_alphabar

        return {
            "beta": beta,
            "sqrt_beta": sqrt_beta,
            "alpha": alpha,
            "alphabar": alphabar,
            "sqrt_alphabar": sqrt_alphabar,
            "sqrt_m_alphabar": sqrt_m_alphabar,
            "oneover_sqrt_alpha": oneover_sqrt_alpha,
            "malpha_over_sqrtmab": malpha_over_sqrtmab
        }

    def _load_schedule(self, schedule):
        for k, v in schedule.items():
            self.register_buffer(k, v)

    def forward_process(self, x):
        x_emb = self.embedder(x)

        ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device) # TODO CHECK SHAPE FOR SEQ_LEN
        eps = self.model(x_emb)
        x_t = self.sqrt_alphabar[ts, None, None] * x_emb + self.sqrt_m_alphabar[ts, None, None] * eps
        pred_eps = self.model(x_t)
        noise_loss = self.criterion(eps, pred_eps)

        y = self.decoder(x_emb)
        y = F.log_softmax(y, dim=-1).permute(0, 2, 1)
        reconstruction_loss = F.cross_entropy(y, x)
        loss = noise_loss + reconstruction_loss
        return loss

    def sample_step(self, x, t):
        z = torch.randn_like(x).to(x.device) if t > 1 else 0
        tt = torch.LongTensor([t] * x.shape[1]).to(x.device)
        eps = self.model(x, tt)
        x = self.oneover_sqrt_alpha[t] * (x - eps * self.malpha_over_sqrtmab[t]) + z * self.sqrt_beta[t]
        return x

    def sample(self, device, n=1, seq_len=64, latents=None):
        self.eval()
        with torch.no_grad():
            x = torch.randn((seq_len, n, self.embed_dim), device=device) if latents is None else latents 
            for t in range(self.n_T, 0, -1):
                x = self.sample_step(x, t)
        indices = self.emb_to_indices(x)
        tokens = [self.vocab.lookup_tokens(i.tolist()) for i in indices.T]

        return [" ".join(token) for token in tokens]

    def emb_to_indices(self, x):
        return F.softmax(self.decoder(x), dim=-1).argmax(dim=-1)

    def run_epoch(self, device):
        self.train()
        losses = []
        loader = tqdm(self.dataloader)
        for i, x in enumerate(loader):
            x = x.to(device)
            self.optimizer.zero_grad()
            loss = self.forward_process(x)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        
        return np.mean(losses)
            

if __name__ == "__main__":
    denoiser = TextDenoiser()

