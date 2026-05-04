import torch
import torch.nn as nn
from einops import rearrange
from timm.data.mixup import Mixup
from timm.loss.cross_entropy import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

IMG_SIZE = 224
PATCH_SIZE = 16
ENC_DIM = 768
ENC_HEADS = 12
ENC_BLOCKS = 12
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2

class PatchEmbed(nn.Module):
    def __init__(self, emb_dim=ENC_DIM, patch_size=PATCH_SIZE):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=3, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return 
    
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
    
class ViTBase(nn.Module):
    def __init__(self, num_classes = 100):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, ENC_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + NUM_PATCHES, ENC_DIM))
        self.blocks = nn.ModuleList(*[
            TransformerBlock(ENC_DIM, ENC_HEADS) for _ in range(ENC_BLOCKS)
        ])
        self.norm = nn.LayerNorm(ENC_DIM)
        self.head = nn.Linear(ENC_DIM, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        return self.head(cls_out)
    
def get_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        return base_lr * epoch/warmup_epochs
    progress = (total_epochs - epoch)/(total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + progress)

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def build_optimizer(model, base_lr, weight_decay, beta1, beta2):
    decay, no_decay = []
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

def train(model, train_loader, val_loader, device, num_classes, 
          total_epochs, warmup_epochs, base_lr, weight_decay, 
          drop_path_rate):
    optimizer = build_optimizer(model, base_lr, weight_decay, beta1=0.9, beta2=0.95)
    mixup_fn = build_mixup(num_classes, mixup=0.8, cutmix=1.0)
    scaler = torch.amp.GradScaler()
    criterion = SoftTargetCrossEntropy()

    actual_batch_size = 128
    target_batch_size = 1024
    accumulation_steps = target_batch_size // actual_batch_size

    best_acc = 0.0

    for epoch in range(total_epochs):
        lr = get_lr(epoch, warmup_epochs, total_epochs, base_lr)
        set_lr(optimizer, lr)
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for step, (imgs, targets) in enumerate(train_loader):
            imgs.to(device)
            targets.to(device)

            images, targets = mixup_fn(images, targets)
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, targets) / accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1)%accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps
    
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch} | Loss: {avg_loss:.4f}")
    return model