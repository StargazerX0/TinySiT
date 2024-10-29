import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from transport import create_transport, Sampler
from train_utils import parse_transport_args
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_models, SiTBlock
import argparse
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import random


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features_dir, labels_dir, flip=0):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))
        self.flip = flip

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))

        if self.flip>0:
            if random.random() < self.flip:
                features = features[1:]
            else:
                features = features[:1]
        return torch.from_numpy(features), torch.from_numpy(labels)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--data-path", type=str, default="data/imagenet/train")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    parser.add_argument("--pruner", type=str, default='dense', choices=['dense', 'magnitude', 'random', 'sparsegpt', 'wanda'])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parse_transport_args(parser)

    args = parser.parse_args()

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    model.to(device)

    # Layer Expansion
    @torch.no_grad()
    def draw_activation_distribution(module, inputs, outputs):
        # N, L, D 
        print(module.layer_id)
        out_state = outputs
        # draw a histogram for the out_state
        plt.figure(figsize=(10,3))
        plt.hist(out_state.cpu().numpy().flatten(), bins=5000, density=True, color='#005a8c', edgecolor='black', linewidth=1.2)
        std = torch.std(out_state).item()
        # Highlight the max and min values
        plt.axvline(out_state.max().item(), linestyle='dashed', linewidth=3, color='#c76a4a')
        plt.axvline(out_state.min().item(), linestyle='dashed', linewidth=3, color='#86423e')

        # Highlight std, 2std, 3std
        plt.axvline(std, linestyle='dashed', linewidth=3, color='#004d4f')
        #plt.axvline(2*std, color='g', linestyle='dashed', linewidth=1)
        #plt.axvline(3*std, color='g', linestyle='dashed', linewidth=1)

        # mark all above values
        fontsize = 10
        offset = 1.5
        offset_y = 0
        plt.text(out_state.max().item()*1.2, offset_y, f" Max: {out_state.max().item():.2f}", rotation=90, verticalalignment='bottom', fontsize=fontsize)
        plt.text(out_state.min().item()*0.8, offset_y, f" Min: {out_state.min().item():.2f}", rotation=90, verticalalignment='bottom', fontsize=fontsize)
        plt.text(std*1.2, offset_y, f" std: {std:.2f}", rotation=90, verticalalignment='bottom', fontsize=fontsize)
        #plt.text(2*std+offset, offset_y, f"2*std: {2*std:.2f}", rotation=90, verticalalignment='bottom', fontsize=fontsize)
        #plt.text(3*std+offset, offset_y, f"3*std: {3*std:.2f}", rotation=90, verticalalignment='bottom', fontsize=fontsize)

        plt.xlabel("Activation Value")
        plt.ylabel("Density")
        plt.xscale('symlog')

        plt.xlim(out_state.min().item()*1.5, out_state.max().item()*1.5)
        plt.grid()
        plt.title(f"Layer {module.layer_id}")
        # remove white boundary
        os.makedirs("outputs/vis_activation", exist_ok=True)
        plt.savefig(f"outputs/vis_activation/pdf_activation_distribution_{module.layer_id}.png", bbox_inches='tight')
        plt.savefig(f"outputs/vis_activation/png_activation_distribution_{module.layer_id}.pdf", bbox_inches='tight')
        plt.close()


        # show token norm 256 x 256
        #plt.figure()
        #token_norm = out_state.norm(dim=-1, p=2) # N, L
        #token_norm = token_norm.view(-1, 256, 256)
        #plt.imshow(token_norm[0].cpu().numpy())
        #plt.colorbar()
        #os.makedirs("outputs/vis_token_norm", exist_ok=True)
        #plt.savefig(f"outputs/vis_token_norm/token_norm_{module.layer_id}.png")


    hooks = []
    for i, layer in enumerate(model.blocks):
        hooks.append(layer.register_forward_hook(draw_activation_distribution))
        layer.layer_id = i
        
    features_dir = f"{args.data_path}/imagenet256_features"
    labels_dir = f"{args.data_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir, flip=0.5)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  

    nbatches = 1
    num_timesteps = 250

    for i, (x, y) in enumerate(loader):
        if i == nbatches: break
        print(f"Batch {i+1}/{nbatches}")
        x = x.to(device)
        y = y.to(device)
        x = x.squeeze(dim=1)
        y = y.squeeze(dim=1)
        t = torch.randint(
            0, 
            num_timesteps, 
            (x.shape[0],), 
            device=device
        )

        with torch.no_grad():
            inputs_dict = {
                "x": x,
                "t": t,
                "y": y,
            }
            _ = model(**inputs_dict)

    for hook in hooks:
        hook.remove()
    