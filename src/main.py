from text_denoiser import TextDenoiser
import argparse
from run_training import run_training
import os
import shutil
import json
import time

def str_to_bool(s):
    return s in ["True", "true", "1"]

def main(args):
    # * Add additional args
    args.log_dir = "logs/" + args.log_name
    # args.im_shape = dataset_to_im_shape_map[args.dataset]

    # * Manipulate args
    if args.pretrained_emb is not None:
        args.freeze_emb = True

    # ! PROTECTED DIRS
    protected_dirs = []
    if args.log_name in protected_dirs:
        raise Exception(f"Log name {args.log_name} is protected, please choose another one")

    if not os.path.exists("logs"):
        os.mkdir("logs")

    # Remove if older than 30 seconds 
    # If not, keep the folder as it contains data created by SLURM
    if os.path.exists(args.log_dir) and (time.time() - os.path.getctime(args.log_dir)) > 30:
        shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)
    subdirs = ["models", "samples"]
    [os.makedirs(args.log_dir + "/" + subdir) for subdir in subdirs if not os.path.exists(args.log_dir + "/" + subdir)]

    with open(args.log_dir + "/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    run_training(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_name", default="train_test", help="The directory to log in", type=str)
    parser.add_argument("--dataset", default="wikitext2", help="The dataset to use", type=str)
    parser.add_argument("--line_slice", default=None, help="The number of lines to use from the dataset", type=int)
    parser.add_argument("--save_interval", default=20, help="The number of epochs between saving models", type=int)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=1e-4)
    parser.add_argument("--beta2", type=float, default=0.02)
    parser.add_argument("--n_T", type=int, default=1000)

    parser.add_argument("--seq_len", type=int, default=32)

    parser.add_argument("--embed_dim", type=int, default=228)
    parser.add_argument("--d_model", type=int, default=114)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=1024)

    parser.add_argument("--use_old_arch", type=str_to_bool, default=False)
    
    parser.add_argument("--pretrained_emb", type=str, default=None)
    parser.add_argument("--freeze_emb", type=str_to_bool, default=False)

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)




    args = parser.parse_args()
    main(args)
