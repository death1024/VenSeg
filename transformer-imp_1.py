import os
import glob
import random
import math
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gather_image_mask_pairs(data_dir, image_suffix="_crop.nii.gz"):
    images = sorted(
        f for f in glob.glob(os.path.join(data_dir, f"*{image_suffix}"))
        if "mask" not in os.path.basename(f)
    )
    pairs = []
    for img_path in images:
        base_name = img_path.replace(image_suffix, "")
        corr_path = base_name + "_mask_corr.nii.gz"
        crop_path = base_name + "_mask_crop.nii.gz"

        if os.path.exists(corr_path):
            pairs.append((img_path, corr_path))
        elif os.path.exists(crop_path):
            pairs.append((img_path, crop_path))
        else:
            print(f"[Skipping] No mask found for {img_path}")
    return pairs


def split_data(pairs, train_ratio=0.8):
    random.shuffle(pairs)
    train_size = int(len(pairs) * train_ratio)
    train_list = pairs[:train_size]
    val_list = pairs[train_size:]
    return train_list, val_list


def random_flip_3d(image, label):
    if random.random() < 0.5:
        image = image.flip(dims=[1])  # flip D
        label = label.flip(dims=[1])
    if random.random() < 0.5:
        image = image.flip(dims=[2])  # flip H
        label = label.flip(dims=[2])
    if random.random() < 0.5:
        image = image.flip(dims=[3])  # flip W
        label = label.flip(dims=[3])
    return image, label


def random_3d_rotation(image, label, max_angle=15):
    img_np = image[0].cpu().numpy() 
    lbl_np = label[0].cpu().numpy()
    D, H, W = img_np.shape
    center = np.array([D / 2, H / 2, W / 2])

    rx = math.radians(random.uniform(-max_angle, max_angle))
    ry = math.radians(random.uniform(-max_angle, max_angle))
    rz = math.radians(random.uniform(-max_angle, max_angle))

    # 构造旋转矩阵
    cx, sx = math.cos(rx), math.sin(rx)
    Rx = np.array([
        [1,  0,   0],
        [0, cx,  -sx],
        [0, sx,   cx],
    ])
    cy, sy = math.cos(ry), math.sin(ry)
    Ry = np.array([
        [ cy, 0,  sy],
        [  0, 1,   0],
        [-sy, 0,  cy],
    ])
    cz, sz = math.cos(rz), math.sin(rz)
    Rz = np.array([
        [ cz, -sz, 0],
        [ sz,  cz, 0],
        [  0,   0, 1],
    ])
    R = Rz @ Ry @ Rx
    offset = center - R @ center

    rotated_img = affine_transform(
        img_np, R, offset=offset, order=1, mode='nearest'
    )
    rotated_lbl = affine_transform(
        lbl_np, R, offset=offset, order=0, mode='nearest'
    )
    rotated_img = torch.from_numpy(rotated_img).unsqueeze(0).to(image.device, dtype=image.dtype)
    rotated_lbl = torch.from_numpy(rotated_lbl).unsqueeze(0).to(label.device, dtype=label.dtype)
    return rotated_img, rotated_lbl

# DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff

def compute_metrics(logits, targets, smooth=1e-5):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == targets).float().mean()

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
    return acc.item(), dice.item()


class VentriclesDataset(Dataset):
    def __init__(self,
                 pairs_list,
                 patch_size=(64, 64, 64),
                 num_patches_per_volume=10,
                 augment=False,
                 max_angle=15):
        self.pairs_list = pairs_list
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.augment = augment
        self.max_angle = max_angle

        self.images = []
        self.labels = []
        for (img_path, lbl_path) in pairs_list:
            img_nib = nib.load(img_path)
            lbl_nib = nib.load(lbl_path)
            img_data = img_nib.get_fdata().astype(np.float32)
            lbl_data = lbl_nib.get_fdata().astype(np.float32)

            # 归一化
            mn, mx = img_data.min(), img_data.max()
            if mx - mn > 1e-6:
                img_data = (img_data - mn) / (mx - mn)

            # 二值化
            lbl_data[lbl_data >= 0.5] = 1
            lbl_data[lbl_data < 0.5] = 0

            self.images.append(img_data)
            self.labels.append(lbl_data)

        self.total_count = len(self.pairs_list) * self.num_patches_per_volume

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        vol_idx = idx // self.num_patches_per_volume
        img_data = self.images[vol_idx]
        lbl_data = self.labels[vol_idx]
        D, H, W = img_data.shape
        psD, psH, psW = self.patch_size

        d1 = random.randint(0, max(D - psD, 0)) if D >= psD else 0
        h1 = random.randint(0, max(H - psH, 0)) if H >= psH else 0
        w1 = random.randint(0, max(W - psW, 0)) if W >= psW else 0
        d2, h2, w2 = d1 + psD, h1 + psH, w1 + psW

        patch_img = img_data[d1:d2, h1:h2, w1:w2]
        patch_lbl = lbl_data[d1:d2, h1:h2, w1:w2]

        patch_img = np.expand_dims(patch_img, axis=0)  
        patch_lbl = np.expand_dims(patch_lbl, axis=0)

        patch_img = torch.from_numpy(patch_img).float()
        patch_lbl = torch.from_numpy(patch_lbl).float()

        if self.augment:
            patch_img, patch_lbl = random_flip_3d(patch_img, patch_lbl)
            patch_img, patch_lbl = random_3d_rotation(patch_img, patch_lbl, self.max_angle)

        return patch_img, patch_lbl

# 下采样
class PatchMerging3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduction = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        out = self.reduction(x)
        return out


# Swin
def window_partition_3d(x, window_size):
    B, C, D, H, W = x.shape
    Wd, Wh, Ww = window_size
    x = x.view(B, C,
               D // Wd, Wd,
               H // Wh, Wh,
               W // Ww, Ww)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    x = x.view(-1, Wd * Wh * Ww, C)
    return x


def window_reverse_3d(windows, window_size, B, C, D, H, W):
    Wd, Wh, Ww = window_size
    num_win_d = D // Wd
    num_win_h = H // Wh
    num_win_w = W // Ww

    x = windows.view(B, num_win_d, num_win_h, num_win_w, Wd, Wh, Ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, C, D, H, W)
    return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(Bn, N, C)
        out = self.proj(out)
        return out


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, in_channels, window_size=(4, 4, 4), shift_size=(0, 0, 0),
                 num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = WindowAttention3D(dim=in_channels, num_heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(in_channels)
        hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        Sd, Sh, Sw = self.shift_size
        if any(s > 0 for s in self.shift_size):
            x_shifted = torch.roll(x, shifts=(-Sd, -Sh, -Sw), dims=(2, 3, 4))
        else:
            x_shifted = x

        # LN + window partition
        x_for_ln = x_shifted.permute(0, 2, 3, 4, 1)  
        x_for_ln = self.norm1(x_for_ln)
        x_for_ln = x_for_ln.permute(0, 4, 1, 2, 3).contiguous() 

        w = window_partition_3d(x_for_ln, self.window_size)  
        # Swin 注意力
        attn_out = self.attn(w)
        # 翻转一下窗口
        merged = window_reverse_3d(
            attn_out, self.window_size,
            B, C, D, H, W
        )
        if any(s > 0 for s in self.shift_size):
            merged = torch.roll(merged, shifts=(Sd, Sh, Sw), dims=(2, 3, 4))

        x = x + merged  

        # MLP
        x_for_ln2 = x.permute(0, 2, 3, 4, 1)
        x_for_ln2 = self.norm2(x_for_ln2)
        x_for_ln2 = x_for_ln2.permute(0, 4, 1, 2, 3).contiguous()

        Bf, Cf, Df, Hf, Wf = x_for_ln2.shape
        x_2d = x_for_ln2.view(Bf, Cf, -1).transpose(1, 2).reshape(-1, Cf)

        mlp_out = self.mlp(x_2d)
        mlp_out = mlp_out.view(Bf, Df * Hf * Wf, Cf).transpose(1, 2)
        mlp_out = mlp_out.view(Bf, Cf, Df, Hf, Wf)

        x = x + mlp_out
        return x


# 全局自注意力 (VT)
class ViTBlock3D(nn.Module):
    def __init__(self, in_channels, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(in_channels)
        hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        B, C, D, H, W = x.shape
        x_for_ln = x.permute(0, 2, 3, 4, 1).contiguous()  
        x_2d = x_for_ln.view(B, D * H * W, C)

        # LN
        x_ln = self.norm1(x_2d)
        # MHSA
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x_2d = x_2d + attn_out

        # MLP
        x_ln2 = self.norm2(x_2d)
        mlp_out = self.mlp(x_ln2)
        x_2d = x_2d + mlp_out

        # reshape back
        x_new = x_2d.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return x_new


# 先门控 + 残差融合，再上采样
class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 门控
        self.gate_conv = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1)
        self.gate_bn   = nn.BatchNorm3d(in_ch)
        self.gate_act  = nn.Sigmoid()

        # 融合后卷积
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
        )

        # 最后转置卷积
        self.up_conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x, skip):
        g = self.gate_conv(skip)
        g = self.gate_bn(g)
        g = self.gate_act(g)

        x = x + skip * g
        x = self.conv(x)
        x = self.up_conv(x)  
        return x


class AdvancedSwin3DNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 base_ch=48,
                 window_size=(4, 4, 4),
                 shift_size=(2, 2, 2),
                 depths=[2, 2, 2],
                 out_channels=1,
                 dropout=0.1):
        super().__init__()

        # Encoder
        self.in_conv = nn.Conv3d(in_channels, base_ch, kernel_size=3, stride=2, padding=1)
        self.in_bn   = nn.BatchNorm3d(base_ch)
        self.in_act  = nn.ReLU(inplace=True)

        self.stage0_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                in_channels=base_ch,
                window_size=window_size,
                shift_size=(0,0,0) if blk_idx%2==0 else shift_size,
                dropout=dropout
            ) for blk_idx in range(depths[0])
        ])
        self.down0 = PatchMerging3D(base_ch, base_ch*2) 

        self.stage1_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                in_channels=base_ch*2,
                window_size=window_size,
                shift_size=(0,0,0) if blk_idx%2==0 else shift_size,
                dropout=dropout
            ) for blk_idx in range(depths[1])
        ])
        self.down1 = PatchMerging3D(base_ch*2, base_ch*4)  

        self.stage2_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                in_channels=base_ch*4,
                window_size=window_size,
                shift_size=(0,0,0) if blk_idx%2==0 else shift_size,
                dropout=dropout
            ) for blk_idx in range(depths[2])
        ])

        self.bottleneck = nn.ModuleList([
            ViTBlock3D(base_ch*4, num_heads=8, mlp_ratio=4.0, dropout=dropout),
            ViTBlock3D(base_ch*4, num_heads=8, mlp_ratio=4.0, dropout=dropout),
        ])

        # Decoder 
        self.up_block2 = UpBlock3D(in_ch=base_ch*4, out_ch=base_ch*2)
        self.up_block1 = UpBlock3D(in_ch=base_ch*2, out_ch=base_ch)
        self.up_block0 = UpBlock3D(in_ch=base_ch, out_ch=base_ch//2)

        self.out_conv = nn.Conv3d(base_ch//2, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B,1,D,H,W)
        """
        # Encoder
        x = self.in_conv(x)
        x = self.in_bn(x)
        x = self.in_act(x)

        for blk in self.stage0_blocks:
            x = blk(x)
        skip0 = x 
        x = self.down0(x)  

        for blk in self.stage1_blocks:
            x = blk(x)
        skip1 = x  
        x = self.down1(x) 

        for blk in self.stage2_blocks:
            x = blk(x)
        skip2 = x 

        # Bottleneck 
        for blk in self.bottleneck:
            x = blk(x)  
            
        # Decoder 
        x = self.up_block2(x, skip2)   

        x = self.up_block1(x, skip1)   

        x = self.up_block0(x, skip0)   

        logits = self.out_conv(x)    
        return logits


def train_3d_transformer_advanced(
    train_list,
    val_list=None,
    patch_size=(64, 64, 64),
    num_patches_per_volume=10,
    epochs=50,
    batch_size=1,
    lr=1e-4,
    augment=True,
    max_angle=15,
    dropout=0.1,
    device="cuda",
    save_path="advanced_swin3d_ventricles.pth",
    use_scheduler=True,
    early_stopping_patience=None
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 数据集
    train_dataset = VentriclesDataset(
        pairs_list=train_list,
        patch_size=patch_size,
        num_patches_per_volume=num_patches_per_volume,
        augment=augment,
        max_angle=max_angle
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = None
    if val_list is not None and len(val_list) > 0:
        val_dataset = VentriclesDataset(
            pairs_list=val_list,
            patch_size=patch_size,
            num_patches_per_volume=2, 
            augment=False,
            max_angle=0
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    model = AdvancedSwin3DNet(
        in_channels=1,
        base_ch=48,
        window_size=(4, 4, 4),
        shift_size=(2, 2, 2),
        depths=[2, 2, 2],      # 3个 stage
        out_channels=1,
        dropout=dropout
    ).to(device)

    dice_loss_fn = DiceLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    def total_loss(logits, targets):
        return dice_loss_fn(logits, targets) + bce_loss_fn(logits, targets)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    train_loss_list = []
    train_acc_list = []
    train_dice_list = []

    val_loss_list = []
    val_acc_list = []
    val_dice_list = []

    best_val_dice = 0.0
    best_epoch = 0
    no_improve_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_val = 0.0
        total_acc_val = 0.0
        total_dice_val = 0.0
        count = 0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = total_loss(logits, lbls)
            loss.backward()
            optimizer.step()

            total_loss_val += loss.item()
            accv, dicev = compute_metrics(logits, lbls)
            total_acc_val += accv
            total_dice_val += dicev
            count += 1

        avg_loss = total_loss_val / count
        avg_acc = total_acc_val / count
        avg_dice = total_dice_val / count

        train_loss_list.append(avg_loss)
        train_acc_list.append(avg_acc)
        train_dice_list.append(avg_dice)

        if scheduler:
            scheduler.step()

        # 验证
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_total_acc = 0.0
            val_total_dice = 0.0
            val_count = 0
            with torch.no_grad():
                for vimgs, vlbls in val_loader:
                    vimgs = vimgs.to(device)
                    vlbls = vlbls.to(device)
                    vout = model(vimgs)
                    vloss = total_loss(vout, vlbls)
                    val_total_loss += vloss.item()
                    vacc, vdice = compute_metrics(vout, vlbls)
                    val_total_acc += vacc
                    val_total_dice += vdice
                    val_count += 1

            avg_val_loss = val_total_loss / val_count
            avg_val_acc = val_total_acc / val_count
            avg_val_dice = val_total_dice / val_count

            val_loss_list.append(avg_val_loss)
            val_acc_list.append(avg_val_acc)
            val_dice_list.append(avg_val_dice)

            # Check best
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)
                no_improve_count = 0
            else:
                no_improve_count += 1

            print(f"[Epoch {epoch}/{epochs}] "
                  f"Train Loss={avg_loss:.4f} Acc={avg_acc:.4f} Dice={avg_dice:.4f} | "
                  f"Val Loss={avg_val_loss:.4f} Acc={avg_val_acc:.4f} Dice={avg_val_dice:.4f}")

            # 早停
            if early_stopping_patience is not None and no_improve_count >= early_stopping_patience:
                print(f"Val dice连续 {early_stopping_patience} 次未提升, 提前停止.")
                break
        else:
            print(f"[Epoch {epoch}/{epochs}] "
                  f"Train Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Dice={avg_dice:.4f}")
            torch.save(model.state_dict(), save_path)

    if val_loader is not None:
        print(f"Best Val Dice={best_val_dice:.4f} at epoch={best_epoch}, saved to {save_path}")
    else:
        print(f"No validation set used. Model saved to {save_path}")

    epochs_range = range(1, len(train_loss_list) + 1)
    plt.figure()
    plt.plot(epochs_range, train_loss_list, label="Train Loss")
    plt.plot(epochs_range, train_dice_list, label="Train Dice")
    if val_loader is not None:
        plt.plot(epochs_range, val_loss_list, label="Val Loss")
        plt.plot(epochs_range, val_dice_list, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Advanced 3D Transformer (3-down-3-up) Training Curves")
    plt.legend()
    plt.savefig("advanced_swin3d_training_curves.png")
    plt.show()

if __name__ == "__main__":
    set_seed(1)

    data_dir = "./data"
    pairs = gather_image_mask_pairs(data_dir, image_suffix="_crop.nii.gz")
    print("Total pairs found:", len(pairs))

    train_list, val_list = split_data(pairs, train_ratio=0.8)
    print("Train samples:", len(train_list))
    print("Val samples:", len(val_list))

    train_3d_transformer_advanced(
        train_list=train_list,
        val_list=val_list,
        patch_size=(64, 64, 64),
        num_patches_per_volume=5,
        epochs=40,
        batch_size=1,
        lr=1e-4,
        augment=True,
        max_angle=15,
        dropout=0.1,
        device="cuda",
        save_path="advanced_swin3d_ventricles_best.pth",
        use_scheduler=True,
        early_stopping_patience=50
    )
