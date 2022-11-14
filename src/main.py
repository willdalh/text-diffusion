from text_denoiser import TextDenoiser
import argparse

def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(1e-4, 0.02))
    parser.add_argument("--n_T", type=int, default=1000)
    parser.add_argument("--embed_dim", type=int, default=768)
    args = parser.parse_args()
    main(args)