import math
import os
import random
import torch
import torch.nn as nn
import time
from einops import rearrange
from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from timm.utils.model_ema import ModelEmaV3
from timm.layers.drop import DropPath
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_PATH = "/scratch/bgvu/nma1/project/data/imagenet100"
IMG_SIZE = 224
PATCH_SIZE = 16
ENC_DIM = 768
ENC_HEADS = 12
ENC_BLOCKS = 12
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
DROP_PATH_RATE = 0.2
CHECKPOINT_PATH = "mae_imagenet.pt"

random.seed(42)

class PatchEmbed(nn.Module):
    def __init__(self, emb_dim=ENC_DIM, patch_size=PATCH_SIZE):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=3, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x
    
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
        self.drop_path = DropPath(drop_prob = DROP_PATH_RATE)
    
    def forward(self, x):
        n = self.norm1(x)
        x = x + self.drop_path(self.attention(n, n, n)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
def sinusoidal_positional_encoding(num_patches, emb_dim):
    pos = torch.arange(num_patches).unsqueeze(1).float()
    dims = torch.arange(0, emb_dim, 2).float()
    angles = pos/(10000 ** (dims/emb_dim))
    pos_enc = torch.zeros((num_patches, emb_dim))
    pos_enc[:, 0::2] = torch.sin(angles)
    pos_enc[:, 1::2] = torch.cos(angles)
    return pos_enc.unsqueeze(0)

class ViTBase(nn.Module):
    def __init__(self, num_classes = 100):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.register_buffer('pos_enc', sinusoidal_positional_encoding(NUM_PATCHES, ENC_DIM))
        self.blocks = nn.Sequential(*[
            TransformerBlock(ENC_DIM, ENC_HEADS) for _ in range(ENC_BLOCKS)
        ])
        self.norm = nn.LayerNorm(ENC_DIM)
        self.head = nn.Linear(ENC_DIM, num_classes)
    
    def forward(self, x):
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_enc
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)
    
def get_train_loader(data_path, num_classes=100, batch_size=128, num_workers=16):
    train_transform = create_transform(
        input_size=224,
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1',
        re_prob=0.25,
        re_mode='pixel',
        interpolation='bicubic',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )

    dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)

    # Pick num_classes random classes and keep all their samples
    chosen_classes = random.sample(range(len(dataset.classes)), num_classes)
    chosen_set     = set(chosen_classes)
    label_map      = {orig: new for new, orig in enumerate(sorted(chosen_classes))}
    indices        = [i for i, (_, label) in enumerate(dataset.samples) if label in chosen_set]

    for i in indices:
        path, orig            = dataset.samples[i]
        dataset.samples[i]    = (path, label_map[orig])
        dataset.targets[i]    = label_map[orig]

    subset = torch.utils.data.Subset(dataset, indices)
    print(f"Train: {len(subset):,} images, {num_classes} classes")

    return DataLoader(subset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=True, persistent_workers=True), label_map


def get_val_loader(data_path, label_map, batch_size=128, num_workers=8):
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    dataset  = datasets.ImageFolder(os.path.join(data_path, 'validation'), transform=val_transform)
    chosen_set = set(label_map.keys())
    indices    = [i for i, (_, label) in enumerate(dataset.samples) if label in chosen_set]

    for i in indices:
        path, orig            = dataset.samples[i]
        dataset.samples[i]    = (path, label_map[orig])
        dataset.targets[i]    = label_map[orig]

    subset = torch.utils.data.Subset(dataset, indices)
    print(f"Val:   {len(subset):,} images, {len(label_map)} classes")

    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
def get_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        return base_lr * epoch/warmup_epochs
    progress = (epoch - warmup_epochs)/(total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def build_optimizer(model, base_lr, weight_decay, beta1, beta2):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith('bias') or 'pos_embed' in name or 'cls_token' in name:
            no_decay.append(param)
        else:
            decay.append(param)
        
    param_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0}
    ]

    return torch.optim.AdamW(param_groups, lr = base_lr, betas=(beta1, beta2))

def build_mixup(num_classes, mixup, cutmix):
    return Mixup(
        mixup_alpha=mixup,
        cutmix_alpha=cutmix,
        prob = 1.0,
        switch_prob = 0.5, 
        mode = 'batch', 
        label_smoothing = 0.1,
        num_classes = num_classes
    )

def load_mae_encoder_weights(vit_model, mae_checkpoint_path, device):
    mae_state = torch.load(mae_checkpoint_path, map_location=device)
    
    # Extract only the encoder weights, stripping the "encoder." prefix
    encoder_weights = {
        k.replace("encoder.", ""): v
        for k, v in mae_state.items()
        if k.startswith("encoder.")
    }
    
    # Load into ViT, strict=False so the classification head is left randomly initialized
    missing, unexpected = vit_model.load_state_dict(encoder_weights, strict=False)
    print(f"Missing keys:    {missing}")      # expect: ['head.weight', 'head.bias']
    print(f"Unexpected keys: {unexpected}")   # expect: []
    
    return vit_model

def train(model, train_loader, val_loader, device, num_classes, 
          total_epochs, warmup_epochs, base_lr, weight_decay):
    optimizer = build_optimizer(model, base_lr, weight_decay, beta1=0.9, beta2=0.999)
    mixup_fn = build_mixup(num_classes, mixup=0.8, cutmix=1.0)
    scaler = torch.amp.GradScaler()
    criterion = SoftTargetCrossEntropy()
    ema_model = ModelEmaV3(model, decay = 0.9999, device=device)

    actual_batch_size = 128
    target_batch_size = 1024
    accumulation_steps = target_batch_size // actual_batch_size

    for epoch in range(total_epochs):
        start_time = time.time()
        lr = get_lr(epoch, warmup_epochs, total_epochs, base_lr)
        set_lr(optimizer, lr)
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for step, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            images, targets = mixup_fn(images, targets)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets) / accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1)%accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema_model.update(model)
            total_loss += loss.item() * accumulation_steps
        
        if len(train_loader) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema_model.update(model)
    
        avg_loss = total_loss / len(train_loader)

        ema_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                with torch.amp.autocast("cuda"):
                    logits = ema_model.module(images)
                    preds = logits.argmax(dim = 1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
        val_acc = correct/total
        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch} | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} | Time: {epoch_time:.1f}")
    return model, ema_model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model =ViTBase().to(device)
    model = load_mae_encoder_weights(model, CHECKPOINT_PATH, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    train_loader, label_map = get_train_loader(data_path=DATA_PATH)
    val_loader = get_val_loader(data_path=DATA_PATH, label_map = label_map)

    model, ema_model = train(model, train_loader, val_loader, device, 100, 200, 5, 1e-3, 0.05)

    torch.save(model.state_dict(), 'vit-b_mae.pt')
    print("Saved model to vit-b_mae.pt")