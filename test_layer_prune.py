import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import csv

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args

def center_crop_arr(pil_image, image_size):
    """Center crop implementation."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def main(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.global_seed)

    # Create model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    # Modify forward function to accept keep_layers
    def forward_with_layer_selection(self, x, t, y, keep_layers=None):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y 
        
        if keep_layers is None:
            keep_layers = range(len(self.blocks))
        for i, block in enumerate(self.blocks):
            if i in keep_layers:
                x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x
    
    model.forward = forward_with_layer_selection.__get__(model)
    
    # Load pretrained model if specified
    if args.load_weight is not None:
        initial_ckpt = torch.load(args.load_weight, map_location='cpu')
        if 'ema' in initial_ckpt:
            model.load_state_dict(initial_ckpt['ema'])
        elif 'model' in initial_ckpt:
            model.load_state_dict(initial_ckpt['model'])
        else:
            model.load_state_dict(initial_ckpt)
        print(f"Loaded model weights from {args.load_weight}")

    model.eval()

    # Setup VAE:
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Load only the first batch
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity;
    
    # Encode images to latent space
    with torch.no_grad():
        x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
        
    # Evaluation function
    def evaluate_combination(keeping_indices):
        with torch.no_grad():
            model_kwargs = dict(y=y, keep_layers=keeping_indices)
            loss_dict = transport.training_losses(model, x_latent, model_kwargs)
            loss = loss_dict["loss"].mean()
        return loss.item()

    # Evaluate removing one layer at a time
    results = []
    num_layers = len(model.blocks)
    for removed_layer in range(num_layers):
        keeping_indices = list(range(num_layers))
        keeping_indices.remove(removed_layer)
        loss = evaluate_combination(keeping_indices)
        results.append((removed_layer, loss))
        print(f"Removed layer {removed_layer}: Loss = {loss}")

    # Save results to CSV
    csv_file = "layer_removal_results.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Removed Layer", "Loss"])
        for removed_layer, loss in results:
            writer.writerow([removed_layer, loss])
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--load-weight", type=str, default=None)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)