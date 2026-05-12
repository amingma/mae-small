import torch
import torch.nn as nn
import math
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange
import time
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

DATA_PATH = "/scratch/bgvu/nma1/project/data/imagenet100"
BATCH_SIZE = 128
ENC_DIM = 768
DEC_DIM = 512
IMG_SIZE = 224
ENC_BLOCKS = 12
ENC_HEADS = 12
DEC_BLOCKS = 8
DEC_HEADS = 8
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
MASK_RATIO = 0.75
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 800
SAVE_PATH = 'mae_imagenet.pt'

class PatchEmbed(nn.Module):
    def __init__(self, emb_dim=ENC_DIM, patch_size=PATCH_SIZE):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=3, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

def sinusoidal_positional_encoding(num_patches, emb_dim):
    pos = torch.arange(num_patches).unsqueeze(1).float()
    dims = torch.arange(0, emb_dim, 2).float()
    angles = pos/(10000 ** (dims/emb_dim))
    pos_enc = torch.zeros((num_patches, emb_dim))
    pos_enc[:, 0::2] = torch.sin(angles)
    pos_enc[:, 1::2] = torch.cos(angles)
    return pos_enc.unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first = True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio, dim)
        )
    
    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attention(n, n, n)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.register_buffer('pos_enc', sinusoidal_positional_encoding(NUM_PATCHES, ENC_DIM))
        self.blocks = nn.Sequential(*[
            TransformerBlock(ENC_DIM, ENC_HEADS) for _ in range(ENC_BLOCKS)
        ])
        self.norm = nn.LayerNorm(ENC_DIM)
    
    def forward(self, x, mask_ratio = 0.75):
        B = x.shape[0]
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_enc

        N = tokens.shape[1]
        n_mask = int(mask_ratio * N)
        n_keep = N - n_mask

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :n_keep]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, ENC_DIM))

        visible = self.blocks(visible)
        visible = self.norm(visible)

        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return visible, ids_restore, mask
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(ENC_DIM, DEC_DIM)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, DEC_DIM))
        nn.init.normal_(self.mask_token, std=0.02)

        self.register_buffer('pos_enc', sinusoidal_positional_encoding(NUM_PATCHES, DEC_DIM))
        self.blocks = nn.Sequential(*[
            TransformerBlock(DEC_DIM, DEC_HEADS) for _ in range(DEC_BLOCKS)
        ])
        self.norm = nn.LayerNorm(DEC_DIM)
        self.pred = nn.Linear(DEC_DIM, PATCH_SIZE * PATCH_SIZE * 3)
    
    def forward(self, visible, ids_restore):
        B = visible.shape[0]
        n_keep = visible.shape[1]
        n_mask = NUM_PATCHES - n_keep

        x = self.embed(visible)

        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, DEC_DIM))
        x_full = x_full + self.pos_enc
        x_full = self.blocks(x_full)
        x_full = self.norm(x_full)
        x_full = self.pred(x_full)

        return x_full
    
class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def patch(self, imgs):
        x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = PATCH_SIZE, p2 = PATCH_SIZE)
        return x
    
    def forward(self, imgs, mask_ratio=0.75):
        visible, ids_restore, mask = self.encoder(imgs, mask_ratio)
        pred = self.decoder(visible, ids_restore)
        target = self.patch(imgs)

        loss = ((pred - target)**2).mean(dim=-1)
        loss = (loss * mask).sum()/mask.sum()

        return loss, pred, mask

def get_train_loader(data_path, batch_size=BATCH_SIZE, num_workers=16):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    print(f"Pretrain: {len(dataset):,} images across {len(dataset.classes)} classes")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=True, persistent_workers=True)

def get_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * epoch/warmup_epochs
    progress = (epoch - warmup_epochs)/(total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    print("Training")

    train_loader = get_train_loader(data_path=DATA_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MaskedAutoencoder()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
    model.train()
    model.to(device)

    scaler = torch.amp.GradScaler()
    for ep in range(EPOCHS):
        lr = get_lr(ep+1, 50, 800, 1.5e-4) * BATCH_SIZE/256
        set_lr(opt, lr)
        total_loss = 0
        epoch_start = time.time()
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss, _, _ = model(imgs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
        epoch_time = time.time() - epoch_start
        avg = total_loss / len(train_loader)

        print(f"Epoch {ep+1:3d}/{EPOCHS} | "
              f"Train Loss: {avg:.3f} | "
              f"Time: {epoch_time:.1f}s")
    
    torch.save(model.state_dict(), SAVE_PATH)

