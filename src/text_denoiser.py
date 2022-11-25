import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import json
from argparse import Namespace

from get_dataset_and_vocab import get_dataset_and_vocab

from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
import torchtext.data.utils as ttdutils

from models.diffusion_model import DiffusionModel

from text_dataset import TextDataset

class TextDenoiser(nn.Module):
    def __init__(self, vocab, betas = (1e-4, 0.02), n_T = 1000, lr=3e-4, embed_dim=228, d_model=228, dim_feedforward=1024, nhead=12, num_layers=6, use_old_arch=False):
        super(TextDenoiser, self).__init__()
        
        self.betas = betas
        self.n_T = n_T
        self.criterion = nn.MSELoss()
        self._load_schedule(self._get_schedule(*betas, n_T))

       
        self.embed_dim = embed_dim
        self.embedder = nn.Embedding(len(vocab), embed_dim)
        # Freeze embedding weights
        # self.embedder.weight.requires_grad = False
        # self.model = nn.Transformer(d_model=embed_dim, nhead=12, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=3072, dropout=0.1, activation='gelu')s
        use_self_cond = False
        if use_old_arch:
            # Text-Model
            self.model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, activation='gelu'), num_layers=num_layers)
        else:
            # Strudel-Model
            self.model = DiffusionModel(embed_dim=embed_dim, d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, num_layers=num_layers)

        self.decoder = nn.Linear(embed_dim, len(vocab))
        self.decoder.weight = nn.Parameter(self.embedder.weight)
        self.decoder.weight.requires_grad = True

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.vocab = vocab

        self.use_old_arch = use_old_arch
        
        
    def _get_schedule(self, beta_start: float, beta_end: float, n_T: int):
        """Return a linear variance schedule"""
        beta = torch.linspace(beta_start, beta_end, n_T + 1)
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
        """Load variance schedule into the model"""
        for k, v in schedule.items():
            self.register_buffer(k, v)

    def noise(self, x, ts):
        """Corrupt x with noise at timesteps ts"""
        eps = torch.randn_like(x).to(x.device)
        x_noised = self.sqrt_alphabar[None, ts, None] * x + self.sqrt_m_alphabar[None, ts, None] * eps # Equation 3
        return x_noised, eps

    def forward_process_loss_self_cond(self, x):
        """Perform forward process and return losses when using self-conditioning"""
        x_emb = self.embedder(x)

        ts = torch.randint(1, self.n_T, (x.shape[1],)).to(x.device) 
        x_t, eps = self.noise(x_emb, ts)

        # Self conditioning
        eps_pred = torch.zeros_like(eps)
        if True:
            x_pred = self.sqrt_alphabar[None, ts, None] * x_emb + self.sqrt_m_alphabar[None, ts, None] * eps_pred # Equation 3
            inpt = torch.cat([x_t, x_pred], dim=-1)
            eps_pred = self.model(inpt) if self.use_old_arch else self.model(inpt, ts)
            eps_pred = eps_pred.detach()
        # print("x_emb.shape", x_emb.shape)
        # print("eps_pred.shape", eps_pred.shape)
        x_pred = self.sqrt_alphabar[None, ts, None] * x_emb + self.sqrt_m_alphabar[None, ts, None] * eps_pred # Equation 3
        inpt = torch.cat([x_t, x_pred], dim=-1)
        eps_pred = self.model(inpt) if self.use_old_arch else self.model(inpt, ts)
        noise_loss = self.criterion(eps_pred, eps)

        # pred_eps = self.model(x_t) if self.use_old_arch else self.model(x_t, ts)
        # noise_loss = self.criterion(eps, pred_eps)

        y = self.decoder(x_emb)
        reconstruction_loss = F.cross_entropy(y.permute(1, 2, 0), x.T) # y shape: (seq_len, batch_size, vocab_size) -> (batch_size, vocab_size, seq_len), x shape: (seq_len, batch_size) -> (batch_size, seq_len)
        loss = noise_loss + reconstruction_loss
        return loss, {"noise_loss": noise_loss, "reconstruction_loss": reconstruction_loss}

    def forward_process_loss(self, x):
        """Perform forward process and return losses"""
        x_emb = self.embedder(x)

        ts = torch.randint(1, self.n_T, (x.shape[1],)).to(x.device) 
        x_t, eps = self.noise(x_emb, ts)
        pred_eps = self.model(x_t) if self.use_old_arch else self.model(x_t, ts)
        noise_loss = self.criterion(eps, pred_eps)

        y = self.decoder(x_emb)
        reconstruction_loss = F.cross_entropy(y.permute(1, 2, 0), x.T) # y shape: (seq_len, batch_size, vocab_size) -> (batch_size, vocab_size, seq_len), x shape: (seq_len, batch_size) -> (batch_size, seq_len)
        loss = noise_loss + reconstruction_loss
        return loss, {"noise_loss": noise_loss, "reconstruction_loss": reconstruction_loss}

    def sample_step(self, x, t):
        """One step of the reverse process"""
        z = torch.randn_like(x).to(x.device) if t > 1 else 0
        tt = torch.LongTensor([t] * x.shape[1]).to(x.device)
        eps = self.model(x) if self.use_old_arch else self.model(x, tt)
        x = self.oneover_sqrt_alpha[t] * (x - eps * self.malpha_over_sqrtmab[t]) + z * self.sqrt_beta[t] # Equation 5
        return x

    def sample(self, device, n=1, seq_len=64, latents=None):
        """Sample normally"""
        self.eval()
        with torch.no_grad():
            x = torch.randn((seq_len, n, self.embed_dim), device=device) if latents is None else latents 
            for t in range(self.n_T, 0, -1):
                x = self.sample_step(x, t)
        indices = self.emb_to_indices(x)
        tokens = [self.vocab.lookup_tokens(i.tolist()) for i in indices.T]

        return [" ".join(token) for token in tokens]
    
    def sample_self_conditioned(self, device, n=1, seq_len=64, latents=None):
        """Sample using self-conditioning"""
        self.eval()
        with torch.no_grad():
            x = torch.randn((seq_len, n, self.embed_dim), device=device) if latents is None else latents 
            for t in range(self.n_T, 0, -1):
                tt = torch.LongTensor([t] * x.shape[1]).to(x.device)
                tt_prev = torch.LongTensor([t - 1] * x.shape[1]).to(x.device)

                x_pred = self.sqrt_alphabar[None, tt, None] * x + self.sqrt_m_alphabar[None, tt, None] * 0 # Equation 3
                inpt = torch.cat([x, x_pred], dim=-1)
                eps_pred = self.model(inpt) if self.use_old_arch else self.model(inpt, tt_prev)

                z = torch.randn_like(x).to(x.device) if t > 1 else 0
                # tt = torch.LongTensor([t] * x.shape[1]).to(x.device)
                # eps = self.model(x) if self.use_old_arch else self.model(x, tt)
                x = self.oneover_sqrt_alpha[t] * (x - eps_pred * self.malpha_over_sqrtmab[t]) + z * self.sqrt_beta[t] # Equation 5
        indices = self.emb_to_indices(x)
        tokens = [self.vocab.lookup_tokens(i.tolist()) for i in indices.T]
        return [" ".join(token) for token in tokens]



    def emb_to_indices(self, x):
        return F.softmax(self.decoder(x), dim=-1).argmax(dim=-1) 

    def emb_to_tokens(self, x):
        indices = self.emb_to_indices(x)
        return [self.vocab.lookup_tokens(i.tolist()) for i in indices.T]

    def emb_to_str(self, x):
        return [" ".join(token) for token in self.emb_to_tokens(x)]

    @staticmethod
    def load_from_training_log(log_dir, model_name, device):
        """Method for loading a model from a training log to be used for sampling"""
        with open(f"{log_dir}/args.json", "r") as f:
            args = json.load(f)
            args = Namespace(**args)
        vocab = torch.load(f"{log_dir}/vocab.pt")

        betas = (args.beta1, args.beta2)
        n_T = args.n_T
        
        use_old_arch = args.use_old_arch
        
        embed_dim = args.embed_dim
        
        embedder = nn.Embedding(len(vocab), embed_dim)
        if args.use_old_arch:
            model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.nhead, dim_feedforward=args.dim_feedforward, dropout=0.1, activation='gelu'), num_layers=args.num_layers)
        else:
            model = DiffusionModel(embed_dim=args.embed_dim, d_model=args.d_model, dim_feedforward=args.dim_feedforward, nhead=args.nhead, num_layers=args.num_layers)
        decoder = nn.Linear(args.embed_dim, len(vocab))

        denoiser = TextDenoiser(vocab, betas=betas, n_T=n_T, lr=args.lr, embed_dim=embed_dim, d_model=args.d_model, dim_feedforward=args.dim_feedforward, nhead=args.nhead, num_layers=args.num_layers, use_old_arch=use_old_arch)
        denoiser._load_schedule(denoiser._get_schedule(*betas, n_T))
        
        denoiser.embedder = embedder
        denoiser.model = model
        denoiser.decoder = decoder
        denoiser.load_state_dict(torch.load(f"{log_dir}/models/{model_name}", map_location=device))
        denoiser = denoiser.to(device)
        denoiser.eval()

        return denoiser

    def run_epoch(self, dataloader, device):
        """Train one iteration of the dataloader"""
        self.train()
        losses = []
        noise_losses = []
        reconstruction_losses = []

        loader = tqdm(dataloader)
        for i, x in enumerate(loader):
            # print(x.shape)
            x = x.to(device)
            self.optimizer.zero_grad()
            loss, loss_comp = self.forward_process_loss(x)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            noise_losses.append(loss_comp["noise_loss"].item())
            reconstruction_losses.append(loss_comp["reconstruction_loss"].item())
        
        return np.mean(losses), {"noise_loss": np.mean(noise_losses), "reconstruction_loss": np.mean(reconstruction_losses)}

    def cosine_similarity(self, x, y):
        return F.cosine_similarity(x, y, dim=-1)

            

if __name__ == "__main__":
    denoiser = TextDenoiser()

