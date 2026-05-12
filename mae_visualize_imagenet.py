"""
MAE Reconstruction Visualizer — ImageNet (GENERATED WITH CLAUDE)
-----------------------------------------
Shows a grid of: Original | Masked Input | Reconstruction

Usage:
  python mae_visualize.py                            # uses default checkpoint
  python mae_visualize.py --checkpoint my.pt         # custom checkpoint path
  python mae_visualize.py --images_dir /path/to/val  # sample from a folder
  python mae_visualize.py --n_images 8               # show more images
  python mae_visualize.py --mask_ratio 0.5           # try a different mask ratio
  python mae_visualize.py --no_save                  # don't write PNG to disk
  python mae_visualize.py --no_show                  # headless / HPC mode

Requires mae_imagenet.py to be in the same directory.
"""

import torch
import numpy as np
import argparse
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision import transforms

# ── Import everything from your training file ─────────────────────────────────
from mae_imagenet import (
    MaskedAutoencoder,
    DEVICE, PATCH_SIZE, IMG_SIZE, NUM_PATCHES,
    MASK_RATIO, ENC_DIM, DEC_DIM, SAVE_PATH,
)
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

IMAGENET_MEAN = np.array(IMAGENET_DEFAULT_MEAN)
IMAGENET_STD  = np.array(IMAGENET_DEFAULT_STD)

# ── Preprocessing (center-crop, no augmentation) ──────────────────────────────
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def unnormalize(tensor):
    """Convert a normalised (C,H,W) tensor to a displayable (H,W,C) numpy array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)   # (H, W, C)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def unpatchify(patches):
    """
    Convert patch tensor back to an image tensor.
    patches: (N_patches, PATCH_SIZE*PATCH_SIZE*3)
    returns: (3, IMG_SIZE, IMG_SIZE)
    """
    h = w = IMG_SIZE // PATCH_SIZE
    return rearrange(patches, '(h w) (p1 p2 c) -> c (h p1) (w p2)',
                     h=h, w=w, p1=PATCH_SIZE, p2=PATCH_SIZE, c=3)


def make_masked_image(img_tensor, mask):
    """
    Blank out masked patches so we can visualise what the encoder actually saw.

    img_tensor : (3, IMG_SIZE, IMG_SIZE) normalised
    mask       : (NUM_PATCHES,)  1 = masked, 0 = visible
    returns    : (3, IMG_SIZE, IMG_SIZE) with masked patches set to mid-grey
    """
    h = w = IMG_SIZE // PATCH_SIZE
    patches = rearrange(img_tensor, 'c (h p1) (w p2) -> (h w) (p1 p2 c)',
                        h=h, w=w, p1=PATCH_SIZE, p2=PATCH_SIZE)
    patches = patches.clone()
    # 0.5 in normalised space approximates mid-grey
    patches[mask.bool()] = 0.5
    return unpatchify(patches)


def load_images(img_paths):
    """Load a list of image paths and return a batched tensor on DEVICE."""
    tensors = []
    for p in img_paths:
        img = Image.open(p).convert('RGB')
        tensors.append(EVAL_TRANSFORM(img))
    return torch.stack(tensors).to(DEVICE)


@torch.no_grad()
def run_mae(model, imgs, mask_ratio=MASK_RATIO):
    """
    Explicit forward pass (mirrors run_mae in the CIFAR visualizer).
    Returns loss (float), pred (B, N, patch_dim), mask (B, N).
    """
    B = imgs.shape[0]

    # ── Encoder ──────────────────────────────────────────────────────────────
    tokens = model.encoder.patch_embed(imgs)
    tokens = tokens + model.encoder.pos_enc

    N      = tokens.shape[1]
    n_keep = int(N * (1 - mask_ratio))

    noise       = torch.rand(B, N, device=imgs.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :n_keep]
    visible  = torch.gather(tokens, 1,
                            ids_keep.unsqueeze(-1).expand(-1, -1, ENC_DIM))

    visible = model.encoder.blocks(visible)
    visible = model.encoder.norm(visible)

    # Build mask in original patch order (1 = masked, 0 = visible)
    mask = torch.ones(B, N, device=imgs.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)

    # ── Decoder ──────────────────────────────────────────────────────────────
    pred = model.decoder(visible, ids_restore)   # (B, N, patch_dim)

    # ── Loss (masked patches only) ────────────────────────────────────────────
    target = model.patch(imgs)                   # (B, N, patch_dim)
    loss   = ((pred - target) ** 2).mean(dim=-1) # (B, N)
    loss   = (loss * mask).sum() / mask.sum()

    return loss.item(), pred, mask


# ─────────────────────────────────────────────────────────────────────────────
# Main visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize(checkpoint_path, img_paths, mask_ratio=MASK_RATIO, save=True, show=True):

    # ── Load model ────────────────────────────────────────────────────────────
    model = MaskedAutoencoder().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # ── Load images ───────────────────────────────────────────────────────────
    imgs = load_images(img_paths)
    n    = len(img_paths)

    # ── Run MAE ───────────────────────────────────────────────────────────────
    loss, pred, mask = run_mae(model, imgs, mask_ratio=mask_ratio)
    print(f"Reconstruction loss on these {n} image(s): {loss:.4f}")

    # ── Build figure: 3 columns — Original | Masked input | Reconstruction ────
    fig, axes = plt.subplots(
        n, 3,
        figsize=(7, n * 2.4),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.35},
    )
    if n == 1:
        axes = axes[np.newaxis, :]   # ensure 2D indexing works for single image

    col_titles = [
        'Original',
        'Masked input\n(encoder saw this)',
        'Reconstruction\n(decoder output)',
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight='bold', pad=6)

    target_patches = model.patch(imgs)   # (B, N, patch_dim) — for per-image loss

    for i in range(n):
        img_tensor = imgs[i].cpu()
        mask_i     = mask[i].cpu()
        pred_i     = pred[i].cpu()

        # ── Column 0: original ────────────────────────────────────────────────
        ax = axes[i, 0]
        ax.imshow(unnormalize(img_tensor), interpolation='nearest')
        ax.set_ylabel(Path(img_paths[i]).name, fontsize=7, rotation=0,
                      labelpad=90, va='center')
        ax.set_xticks([]); ax.set_yticks([])

        # ── Column 1: masked input ────────────────────────────────────────────
        masked_img = make_masked_image(img_tensor, mask_i)
        axes[i, 1].imshow(unnormalize(masked_img), interpolation='nearest')
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])

        n_vis = int((mask_i == 0).sum().item())
        axes[i, 1].set_xlabel(f'{n_vis}/{NUM_PATCHES} patches visible',
                               fontsize=7, color='gray')

        # ── Column 2: reconstruction ──────────────────────────────────────────
        recon_img = unpatchify(pred_i)
        recon_np  = unnormalize(recon_img)
        axes[i, 2].imshow(recon_np, interpolation='nearest')
        axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])

        # Per-image masked reconstruction loss
        target_i   = target_patches[i].cpu()
        patch_loss = ((pred_i - target_i) ** 2).mean(dim=-1)
        img_loss   = (patch_loss * mask_i).sum() / mask_i.sum()
        axes[i, 2].set_xlabel(f'loss: {img_loss:.3f}', fontsize=7, color='gray')

    # ── Super-title ───────────────────────────────────────────────────────────
    n_masked = int(NUM_PATCHES * mask_ratio)
    fig.suptitle(
        f'MAE Reconstruction — ImageNet   '
        f'(mask ratio: {int(mask_ratio * 100)}%,  '
        f'{n_masked}/{NUM_PATCHES} patches masked)',
        fontsize=10, y=1.01,
    )

    plt.tight_layout()

    if save:
        out = 'mae_reconstructions.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved to {out}")

    if show:
        plt.show()

    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str,   default=SAVE_PATH,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--images',      nargs='+',  default=None,
                        help='One or more explicit image paths')
    parser.add_argument('--images_dir',  type=str,   default=None,
                        help='Directory to sample images from (used when --images is not set)')
    parser.add_argument('--n_images',    type=int,   default=6,
                        help='Number of images to visualize (when sampling from a directory)')
    parser.add_argument('--mask_ratio',  type=float, default=MASK_RATIO,
                        help='Fraction of patches to mask (default: training value)')
    parser.add_argument('--no_save',     action='store_true',
                        help='Do not save the figure to disk')
    parser.add_argument('--no_show',     action='store_true',
                        help='Do not call plt.show() — useful on headless servers')
    args = parser.parse_args()

    # ── Collect image paths ───────────────────────────────────────────────────
    if args.images:
        img_paths = args.images
    elif args.images_dir:
        all_imgs  = [str(p) for p in Path(args.images_dir).rglob('*')
                     if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        # Evenly spread across the directory for variety (mirrors CIFAR script)
        indices   = np.linspace(0, len(all_imgs) - 1, args.n_images, dtype=int)
        img_paths = [all_imgs[i] for i in indices]
    else:
        parser.error('Provide either --images or --images_dir')

    print(f"Reconstructing {len(img_paths)} image(s) | "
          f"mask_ratio={args.mask_ratio} | device={DEVICE}")

    visualize(
        checkpoint_path=args.checkpoint,
        img_paths=img_paths,
        mask_ratio=args.mask_ratio,
        save=not args.no_save,
        show=not args.no_show,
    )