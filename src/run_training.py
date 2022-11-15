from text_denoiser import TextDenoiser
from text_dataset import TextDataset

import torch

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_denoiser = TextDenoiser(betas=(args.beta1, args.beta2), lr=args.lr, n_T=args.n_T, embed_dim=args.embed_dim)
    text_denoiser.train()
    text_denoiser = text_denoiser.to(device)

    torch.save(text_denoiser.vocab, args.log_dir + "/vocab.pt")

    for epoch in range(args.epochs):
        loss, comp_dict = text_denoiser.run_epoch(device)
        print(f"\nEpoch {epoch} - Total loss: {loss}, Noise loss: {comp_dict['noise_loss']}, Reconstruction loss: {comp_dict['reconstruction_loss']}")

        if epoch % args.save_interval == 0 or epoch == args.epochs - 1 or epoch in [0, 1, 2, 3, 4, 5]:
            torch.save(text_denoiser.state_dict(), f"{args.log_dir}/models/saved_model.pt")

            sentences = text_denoiser.sample(device, 2)
            with open(f"{args.log_dir}/samples/samples.txt", "a") as f:
                f.write(f"Epoch {epoch}:\n")
                for sentence in sentences:
                    f.write(sentence)
                    f.write("\n")
                f.write("\n\n")

            


