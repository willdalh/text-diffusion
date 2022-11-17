from text_denoiser import TextDenoiser
from text_dataset import TextDataset
from torch.utils.data import DataLoader

from get_dataset_and_vocab import get_dataset_and_vocab

import torch

def collate_fn(batch):
    return torch.stack(batch, dim=1)

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, vocab = get_dataset_and_vocab(args.dataset, seq_len=args.seq_len, line_slice=args.line_slice)
    
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)


    text_denoiser = TextDenoiser(dataloader, vocab, betas=(args.beta1, args.beta2), lr=args.lr, n_T=args.n_T, embed_dim=args.embed_dim, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers)
    text_denoiser.train()
    text_denoiser = text_denoiser.to(device)

    torch.save(text_denoiser.vocab, args.log_dir + "/vocab.pt")

    previous_losses = []

    for epoch in range(args.epochs):
        loss, comp_dict = text_denoiser.run_epoch(device)

        loss_desc = f"Epoch {epoch} - Total loss: {loss}, Noise loss: {comp_dict['noise_loss']}, Reconstruction loss: {comp_dict['reconstruction_loss']}"
        previous_losses.append(loss_desc)
        print(f"\n{loss_desc}")
        if epoch % args.save_interval == 0 or epoch == args.epochs - 1 or epoch in [0, 1, 2, 3, 4, 5]:
            torch.save(text_denoiser.state_dict(), f"{args.log_dir}/models/saved_model.pt")

            sentences = text_denoiser.sample(device, 2, seq_len=args.seq_len)
            with open(f"{args.log_dir}/samples/samples.txt", "a") as f:
                f.write(f"Epoch {epoch}:\n")
                for sentence in sentences:
                    f.write(sentence)
                    f.write("\n")
                f.write("\n)
            
            with open(f"{args.log_dir}/losses.txt", "a") as f:
                for l in previous_losses:
                    f.write(f"{l}\n")
                previous_losses = []

            


