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

def gather_image_mask_pairs(
    data_dir,
    image_suffix="_crop.nii.gz"
):

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
    train_size = int(len(pairs)*train_ratio)
    train_list = pairs[:train_size]
    val_list   = pairs[train_size:]
    return train_list, val_list

def random_flip_3d(image, label):
    if random.random() < 0.5:
        image = image.flip(dims=[1]) 
        label = label.flip(dims=[1])
    if random.random() < 0.5:
        image = image.flip(dims=[2])  
        label = label.flip(dims=[2])
    if random.random() < 0.5:
        image = image.flip(dims=[3])  
        label = label.flip(dims=[3])
    return image, label

def random_3d_rotation(image, label, max_angle=15):
    img_np = image[0].cpu().numpy()  
    lbl_np = label[0].cpu().numpy()

    D, H, W = img_np.shape
    center = np.array([D/2, H/2, W/2])

    rx = math.radians(random.uniform(-max_angle, max_angle))
    ry = math.radians(random.uniform(-max_angle, max_angle))
    rz = math.radians(random.uniform(-max_angle, max_angle))

    # 构造旋转矩阵 (Rz * Ry * Rx)
    cx, sx = math.cos(rx), math.sin(rx)
    Rx = np.array([
        [1,   0,   0],
        [0,  cx, -sx],
        [0,  sx,  cx],
    ])
    cy, sy = math.cos(ry), math.sin(ry)
    Ry = np.array([
        [ cy, 0,  sy],
        [  0, 1,   0],
        [-sy, 0,  cy],
    ])
    cz, sz = math.cos(rz), math.sin(rz)
    Rz = np.array([
        [ cz, -sz,  0],
        [ sz,  cz,  0],
        [  0,   0,  1],
    ])
    R = Rz @ Ry @ Rx

    # offset
    c_trans = center - R @ center

    rotated_img = affine_transform(
        img_np, R, offset=c_trans, order=1, mode='nearest'
    )
    rotated_lbl = affine_transform(
        lbl_np, R, offset=c_trans, order=0, mode='nearest'
    )

    rotated_img = torch.from_numpy(rotated_img).unsqueeze(0).to(image.device, dtype=image.dtype)
    rotated_lbl = torch.from_numpy(rotated_lbl).unsqueeze(0).to(label.device, dtype=label.dtype)
    return rotated_img, rotated_lbl

# Dice Loss
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

class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch, embed_dim=512, patch_size=(4,4,4)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  
        B, ED, Dp, Hp, Wp = x.shape
        x = x.flatten(2) 
        x = x.transpose(1, 2)
        return x, (Dp, Hp, Wp)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout_p=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_p, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        # Self-Attention
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1)
        x = x + attn_out
        # MLP
        x2 = self.norm2(x)
        mlp_out = self.mlp(x2)
        x = x + mlp_out
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, depth=4, num_heads=8, mlp_ratio=4.0, dropout_p=0.1):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout_p))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TransUNet3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        patch_size=(4,4,4),
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout_p=0.1,
        decoder_channels=(512,256),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size=patch_size)
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads, mlp_ratio, dropout_p)

        self.up_blocks = nn.ModuleList()
        prev_ch = embed_dim
        for ch in decoder_channels:
            block = nn.Sequential(
                nn.ConvTranspose3d(prev_ch, ch, kernel_size=2, stride=2),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True),
            )
            self.up_blocks.append(block)
            prev_ch = ch

        self.out_conv = nn.Conv3d(prev_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # Patch Embedding
        tokens, (Dp, Hp, Wp) = self.patch_embed(x)  
        # Transformer
        trans_out = self.transformer(tokens)      
        # Reshape 
        B, N, ED = trans_out.shape
        x_enc = trans_out.transpose(1,2).view(B, ED, Dp, Hp, Wp)

        # Decoder
        dec = x_enc
        for up_block in self.up_blocks:
            dec = up_block(dec)

        out = self.out_conv(dec)
        return out

class VentriclesDataset(Dataset):
    def __init__(
        self,
        pairs_list,
        patch_size=(64,64,64),
        num_patches_per_volume=10,
        augment=False,
        max_angle=15
    ):
        self.pairs_list = pairs_list
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.augment = augment
        self.max_angle = max_angle

        self.images = []
        self.labels = []
        for (img_path, label_path) in pairs_list:
            img_nib = nib.load(img_path)
            lbl_nib = nib.load(label_path)

            img_data = img_nib.get_fdata().astype(np.float32)
            lbl_data = lbl_nib.get_fdata().astype(np.float32)

            mn, mx = img_data.min(), img_data.max()
            if mx - mn > 1e-6:
                img_data = (img_data - mn) / (mx - mn)

            lbl_data[lbl_data >= 0.5] = 1
            lbl_data[lbl_data < 0.5] = 0

            self.images.append(img_data)
            self.labels.append(lbl_data)

        self.total_count = len(self.pairs_list) * self.num_patches_per_volume

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        volume_idx = idx // self.num_patches_per_volume
        img_data = self.images[volume_idx]
        lbl_data = self.labels[volume_idx]

        D, H, W = img_data.shape
        d_size, h_size, w_size = self.patch_size

        # 随机裁剪
        d1 = random.randint(0, max(D - d_size, 0)) if D>=d_size else 0
        h1 = random.randint(0, max(H - h_size, 0)) if H>=h_size else 0
        w1 = random.randint(0, max(W - w_size, 0)) if W>=w_size else 0
        d2, h2, w2 = d1 + d_size, h1 + h_size, w1 + w_size

        patch_img = img_data[d1:d2, h1:h2, w1:w2]
        patch_lbl = lbl_data[d1:d2, h1:h2, w1:w2]

        patch_img = np.expand_dims(patch_img, axis=0)  
        patch_lbl = np.expand_dims(patch_lbl, axis=0)

        patch_img = torch.from_numpy(patch_img).float()
        patch_lbl = torch.from_numpy(patch_lbl).float()

        # 数据增广
        if self.augment:
            patch_img, patch_lbl = random_flip_3d(patch_img, patch_lbl)
            patch_img, patch_lbl = random_3d_rotation(patch_img, patch_lbl, self.max_angle)

        return patch_img, patch_lbl

def train_3d_transformer(
    train_list,
    val_list=None,
    patch_size=(64,64,64),
    num_patches_per_volume=10,
    epochs=50,
    batch_size=1,
    lr=1e-4,
    augment=True,
    max_angle=15,
    dropout_p=0.1,
    device="cuda",
    save_path="transformer_ventricles_best.pth",
    use_scheduler=True,
    early_stopping_patience=None
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    train_dataset = VentriclesDataset(
        pairs_list=train_list,
        patch_size=patch_size,
        num_patches_per_volume=num_patches_per_volume,
        augment=augment,
        max_angle=max_angle
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = None
    if val_list and len(val_list) > 0:
        val_dataset = VentriclesDataset(
            pairs_list=val_list,
            patch_size=patch_size,
            num_patches_per_volume=2,  
            augment=False,
            max_angle=0
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    model = TransUNet3D(
        in_channels=1,
        out_channels=1,
        patch_size=(4,4,4),   
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout_p=dropout_p,
        decoder_channels=(512,256)
    ).to(device)

    dice_loss_fn = DiceLoss()
    bce_loss_fn  = nn.BCEWithLogitsLoss()
    def total_loss_fn(logits, labels):
        return dice_loss_fn(logits, labels) + bce_loss_fn(logits, labels)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    train_loss_list = []
    train_acc_list  = []
    train_dice_list = []

    val_loss_list   = []
    val_acc_list    = []
    val_dice_list   = []

    best_val_dice      = 0.0
    best_epoch         = 0
    no_improve_counter = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        total_acc  = 0.0
        total_dice = 0.0
        count_batch= 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks= masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = total_loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count_batch += 1

            # metrics
            acc_val, dice_val = compute_metrics(outputs, masks)
            total_acc  += acc_val
            total_dice += dice_val

        avg_train_loss = total_loss / count_batch
        avg_train_acc  = total_acc  / count_batch
        avg_train_dice = total_dice / count_batch

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        train_dice_list.append(avg_train_dice)

        if scheduler:
            scheduler.step()

        # 验证
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_total_acc  = 0.0
            val_total_dice = 0.0
            val_count_batch= 0

            with torch.no_grad():
                for val_imgs, val_masks in val_loader:
                    val_imgs = val_imgs.to(device)
                    val_masks= val_masks.to(device)

                    val_outputs = model(val_imgs)
                    vloss = total_loss_fn(val_outputs, val_masks)
                    val_total_loss += vloss.item()
                    val_count_batch += 1

                    vacc, vdice = compute_metrics(val_outputs, val_masks)
                    val_total_acc  += vacc
                    val_total_dice += vdice

            avg_val_loss = val_total_loss / val_count_batch
            avg_val_acc  = val_total_acc / val_count_batch
            avg_val_dice = val_total_dice / val_count_batch

            val_loss_list.append(avg_val_loss)
            val_acc_list.append(avg_val_acc)
            val_dice_list.append(avg_val_dice)

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            print(f"[Epoch {epoch}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, Dice: {avg_train_dice:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, Dice: {avg_val_dice:.4f}")

            # 早停
            if early_stopping_patience is not None:
                if no_improve_counter >= early_stopping_patience:
                    print(f"验证集Dice连续 {early_stopping_patience} 次未提升, 提前停止训练.")
                    break
        else:
            print(f"[Epoch {epoch}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, Dice: {avg_train_dice:.4f}")
            torch.save(model.state_dict(), save_path)

    if val_loader is not None:
        print(f"Best val dice: {best_val_dice:.4f} (epoch={best_epoch}), saved in {save_path}")
    else:
        print(f"No validation set used. Model saved to {save_path}")

    epochs_range = range(1, len(train_loss_list)+1)
    plt.figure()
    # Loss
    plt.plot(epochs_range, train_loss_list, label="Train Loss")
    if val_loader is not None:
        plt.plot(epochs_range, val_loss_list, label="Val Loss")
    # Acc
    plt.plot(epochs_range, train_acc_list, label="Train Acc")
    if val_loader is not None:
        plt.plot(epochs_range, val_acc_list, label="Val Acc")
    # Dice
    plt.plot(epochs_range, train_dice_list, label="Train Dice")
    if val_loader is not None:
        plt.plot(epochs_range, val_dice_list, label="Val Dice")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("3D Transformer (TransUNet) Training Curves")
    plt.legend()
    plt.savefig("transformer_3d_training_metrics.png")
    plt.show()

if __name__ == "__main__":
    set_seed(1) 
    data_dir = "./data"

    pairs = gather_image_mask_pairs(data_dir=data_dir, image_suffix="_crop.nii.gz")
    print("Total paired samples:", len(pairs))

    train_list, val_list = split_data(pairs, train_ratio=0.8)
    print("Train samples:", len(train_list))
    print("Val samples:", len(val_list))

    train_3d_transformer(
        train_list=train_list,
        val_list=val_list,
        patch_size=(64,64,64),
        num_patches_per_volume=10,
        epochs=50,
        batch_size=1,             
        lr=1e-4,
        augment=True,
        max_angle=15,
        dropout_p=0.1,
        device="cuda:0",
        save_path="transformer_ventricles_best.pth",
        use_scheduler=True,         
        early_stopping_patience=20
    )
