import os
import glob
import math
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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
        image = image.flip(dims=[1])  
        label = label.flip(dims=[1])
    if random.random() < 0.5:
        image = image.flip(dims=[2])  
        label = label.flip(dims=[2])
    if random.random() < 0.5:
        image = image.flip(dims=[3]) 
        label = label.flip(dims=[3])
    return image, label


def random_gamma(image):
    gamma = random.uniform(0.7, 1.5)
    return image ** gamma


def random_affine_3d(image, label, max_angle=15, max_scale=0.1, max_shift=0.1):
   
    angle_d = math.radians(random.uniform(-max_angle, max_angle))
    angle_h = math.radians(random.uniform(-max_angle, max_angle))
    angle_w = math.radians(random.uniform(-max_angle, max_angle))
    # 随机缩放
    scale = 1.0 + random.uniform(-max_scale, max_scale)
    # 随机平移
    shift_d = random.uniform(-max_shift, max_shift)
    shift_h = random.uniform(-max_shift, max_shift)
    shift_w = random.uniform(-max_shift, max_shift)

    img_np = image[0].cpu().numpy() 
    lbl_np = label[0].cpu().numpy()
    D, H, W = img_np.shape
    center = np.array([D/2, H/2, W/2])

    # 构造旋转矩阵
    def rot_x(a):
        return np.array([[1,0,0],
                         [0, math.cos(a), -math.sin(a)],
                         [0, math.sin(a), math.cos(a)]])
    def rot_y(a):
        return np.array([[math.cos(a),0,math.sin(a)],
                         [0,1,0],
                         [-math.sin(a),0,math.cos(a)]])
    def rot_z(a):
        return np.array([[math.cos(a),-math.sin(a),0],
                         [math.sin(a), math.cos(a),0],
                         [0,0,1]])
    R = rot_x(angle_d) @ rot_y(angle_h) @ rot_z(angle_w)
    R *= scale

    # 平移
    offset = center - R @ center + np.array([shift_d*D, shift_h*H, shift_w*W])

    from scipy.ndimage import affine_transform
    img_rot = affine_transform(
        img_np, R, offset=offset, order=1, mode='nearest'
    )
    lbl_rot = affine_transform(
        lbl_np, R, offset=offset, order=0, mode='nearest'
    )
    # 返回Tensor
    img_t = torch.from_numpy(img_rot).unsqueeze(0).to(image.device, dtype=image.dtype)
    lbl_t = torch.from_numpy(lbl_rot).unsqueeze(0).to(label.device, dtype=label.dtype)
    return img_t, lbl_t


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
                 patch_size=(64,64,64),
                 num_patches_per_volume=5,
                 augment=False):
        self.pairs_list = pairs_list
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.augment = augment

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

        self.total_count = len(self.pairs_list)* self.num_patches_per_volume

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        vol_idx = idx // self.num_patches_per_volume
        img_data = self.images[vol_idx]
        lbl_data = self.labels[vol_idx]
        D, H, W = img_data.shape
        psD, psH, psW = self.patch_size

        # 随机裁块
        d1 = random.randint(0, max(D-psD, 0)) if D>psD else 0
        h1 = random.randint(0, max(H-psH, 0)) if H>psH else 0
        w1 = random.randint(0, max(W-psW, 0)) if W>psW else 0
        d2, h2, w2 = d1+psD, h1+psH, w1+psW
        patch_img = img_data[d1:d2, h1:h2, w1:w2]
        patch_lbl = lbl_data[d1:d2, h1:h2, w1:w2]

        patch_img = torch.from_numpy(patch_img)[None,...].float()  
        patch_lbl = torch.from_numpy(patch_lbl)[None,...].float()

        if self.augment:
            # 随机 flip
            patch_img, patch_lbl = random_flip_3d(patch_img, patch_lbl)
            # 随机 gamma
            patch_img = random_gamma(patch_img)
            # 随机仿射
            patch_img, patch_lbl = random_affine_3d(patch_img, patch_lbl,
                                                    max_angle=15, max_scale=0.1, max_shift=0.1)

        return patch_img, patch_lbl

class MLPBlock(nn.Module):
    def __init__(self, embed_dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * expansion)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class LayerNormChannel(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        B, C, D, H, W = x.shape
        xp = x.permute(0,2,3,4,1).contiguous()
        xp = self.ln(xp)
        xp = xp.permute(0,4,1,2,3).contiguous()
        return xp

# 局部注意力 (Swin) + Deformable offset
class DeformWindowAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=(4,4,4), dropout=0.1, n_deform=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, 4, dropout)
        self.n_deform = n_deform
        self.offset_fc = nn.Linear(dim, n_deform*3)  # 3D offset

    def forward(self, x):
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        # 如果不能整除再做padding
        x_win = x.view(B, C, D//wd, wd, H//wh, wh, W//ww, ww)
        x_win = x_win.permute(0,2,4,6,3,5,7,1).contiguous()
        bnw = B*(D//wd)*(H//wh)*(W//ww)
        n_tokens = wd*wh*ww
        x_win = x_win.view(bnw, n_tokens, C)

        # LN + QKV
        x_ln = self.ln1(x_win)
        qkv = self.qkv(x_ln).reshape(bnw, n_tokens, 3, self.num_heads, self.head_dim)
        q = qkv[:,:,0]
        k = qkv[:,:,1]
        v = qkv[:,:,2]
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        # deform offset 
        offset = self.offset_fc(x_ln) 

        # dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.permute(0,2,1,3).reshape(bnw, n_tokens, C)

        out = self.proj(out)
        out = self.dropout(out)
        x_win = x_win + out

        # MLP
        x_ln2 = self.ln2(x_win)
        mlp_out = self.mlp(x_ln2)
        x_win = x_win + mlp_out
        # reshape回原形状
        x_win = x_win.view(B, (D//wd), (H//wh), (W//ww), wd, wh, ww, C)
        x_win = x_win.permute(0,7,1,4,2,5,3,6).contiguous()
        x_out = x_win.view(B, C, D, H, W)
        return x_out

# 全局注意力 (ViT)
class GlobalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, 4, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D*H*W
        xp = x.permute(0,2,3,4,1).reshape(B, N, C)  
        x_ln = self.ln1(xp)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        attn_out = self.dropout(attn_out)
        xp = xp + attn_out
        x_ln2 = self.ln2(xp)
        mlp_out = self.mlp(x_ln2)
        mlp_out = self.dropout(mlp_out)
        xp = xp + mlp_out
        x_out = xp.view(B,D,H,W,C).permute(0,4,1,2,3).contiguous()
        return x_out

# Encoder
class MultiScaleEncoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, depths=(2,2,2), dropout=0.1):
        super().__init__()
        self.init_conv = nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1)
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        curr_dim = base_ch
        for i, d in enumerate(depths):
            stage_blocks = nn.ModuleList([
                nn.ModuleDict({
                    "local": DeformWindowAttentionBlock(curr_dim, num_heads=4, window_size=(2,4,4), dropout=dropout),
                    "global": GlobalAttentionBlock(curr_dim, num_heads=4, dropout=dropout)
                }) for _ in range(d)
            ])
            self.stages.append(stage_blocks)
            # 下采样 conv
            next_dim = curr_dim*2 if i < len(depths)-1 else curr_dim*2
            self.downs.append(nn.Conv3d(curr_dim, next_dim, kernel_size=2, stride=2))
            curr_dim = next_dim
        self.out_dim = curr_dim

    def forward(self, x):
        x = self.init_conv(x)  
        skips = []
        for i, blocks in enumerate(self.stages):
            for blk_dict in blocks:
                x_local = blk_dict["local"](x)
                x_global = blk_dict["global"](x)
                x = x + x_local + x_global
            skips.append(x)
            x = self.downs[i](x)  # 下采样
        return skips, x

class GatedCrossScaleBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # 空间注意力
        self.spa_conv = nn.Conv3d(dim, 1, kernel_size=3, padding=1)
        # 通道注意力
        self.mlp_c1 = nn.Linear(dim, dim//4)
        self.mlp_c2 = nn.Linear(dim//4, dim)
        self.drop = nn.Dropout(dropout)
        self.ln = LayerNormChannel(dim)

    def forward(self, dec_x, skip):
        # 空间门控
        spa = torch.sigmoid(self.spa_conv(skip))  
        skip_gated = skip * spa
        # 通道门控
        b,c,d,h,w = skip_gated.shape
        gap = skip_gated.view(b,c,-1).mean(-1)  
        gap = F.relu(self.mlp_c1(gap))
        gap = self.drop(gap)
        gap = torch.sigmoid(self.mlp_c2(gap)).view(b,c,1,1,1)
        skip_gated = skip_gated * gap
        # 融合
        x = dec_x + skip_gated
        x = self.ln(x)
        return x

class MultiScaleDecoder(nn.Module):
    def __init__(self, depths=(2,2,2), base_ch=32, dropout=0.1):
        super().__init__()
        curr_dim = base_ch*(2**len(depths))  # encoder最后的通道
        self.up_stages = nn.ModuleList()
        for i in reversed(range(len(depths))):
            next_dim = base_ch*(2**i)
            block = nn.ModuleList([
                nn.ConvTranspose3d(curr_dim, next_dim, kernel_size=2, stride=2),
                GatedCrossScaleBlock(next_dim, dropout=dropout)
            ])
            self.up_stages.append(block)
            curr_dim = next_dim
        self.out_dim = curr_dim

    def forward(self, skips, x):
        for i, stage in enumerate(self.up_stages):
            up_conv = stage[0]
            gate_blk = stage[1]
            x = up_conv(x)
            skip = skips[len(skips)-1 - i]
            x = gate_blk(x, skip)
        return x


# 网络
class VentriclesTransformerModel(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, depths=(2,2,2), out_ch=1, dropout=0.1):
        super().__init__()
        self.encoder = MultiScaleEncoder(in_ch, base_ch, depths, dropout)
        self.decoder = MultiScaleDecoder(depths, base_ch, dropout)
        self.out_conv = nn.Conv3d(self.decoder.out_dim, out_ch, kernel_size=1)

    def forward(self, x):
        skips, x_enc = self.encoder(x)
        x_dec = self.decoder(skips, x_enc)
        logits = self.out_conv(x_dec)  
        return logits

def train_ventricles_transformer(
    train_list,
    val_list=None,
    patch_size=(64,64,64),
    num_patches_per_volume=5,
    epochs=50,
    batch_size=1,
    lr=1e-4,
    augment=True,
    device="cuda",
    save_path="transformer_ventricle.pth",
    use_scheduler=False,
    early_stopping_patience=None
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    train_dataset = VentriclesDataset(
        pairs_list=train_list,
        patch_size=patch_size,
        num_patches_per_volume=num_patches_per_volume,
        augment=augment
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = None
    if val_list is not None and len(val_list)>0:
        val_dataset = VentriclesDataset(
            pairs_list=val_list,
            patch_size=patch_size,
            num_patches_per_volume=2,
            augment=False
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    model = VentriclesTransformerModel(
        in_ch=1,
        base_ch=32,
        depths=(2,2,2),
        out_ch=1,
        dropout=0.1
    ).to(device)

    dice_loss_fn = DiceLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    def total_loss(logits, targets):
        return bce_loss_fn(logits, targets) + dice_loss_fn(logits, targets)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) if use_scheduler else None

    train_loss_list, train_acc_list, train_dice_list = [], [], []
    val_loss_list, val_acc_list, val_dice_list = [], [], []

    best_val_dice = 0.0
    best_epoch = 0
    no_improve_count = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss_val, total_acc_val, total_dice_val = 0,0,0
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

        if val_loader is not None:
            model.eval()
            val_total_loss, val_total_acc, val_total_dice = 0,0,0
            val_count = 0
            with torch.no_grad():
                for vimgs, vlbls in val_loader:
                    vimgs = vimgs.to(device)
                    vlbls = vlbls.to(device)
                    logits_val = model(vimgs)
                    vloss = total_loss(logits_val, vlbls)
                    val_total_loss += vloss.item()
                    vacc, vdice = compute_metrics(logits_val, vlbls)
                    val_total_acc += vacc
                    val_total_dice += vdice
                    val_count += 1
            avg_val_loss = val_total_loss / val_count
            avg_val_acc = val_total_acc / val_count
            avg_val_dice = val_total_dice / val_count
            val_loss_list.append(avg_val_loss)
            val_acc_list.append(avg_val_acc)
            val_dice_list.append(avg_val_dice)

            # check best
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

            if early_stopping_patience and no_improve_count >= early_stopping_patience:
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


    epochs_range = range(1, len(train_loss_list)+1)
    plt.figure()
    plt.plot(epochs_range, train_loss_list, label="Train Loss")
    plt.plot(epochs_range, train_dice_list, label="Train Dice")
    if val_loader is not None:
        plt.plot(epochs_range, val_loss_list, label="Val Loss")
        plt.plot(epochs_range, val_dice_list, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Transformer-based Ventricles Segmentation Training")
    plt.legend()
    plt.savefig("transformer_ventricles_training.png")
    plt.show()


# 推理
def mc_dropout_inference(model, x, num_samples=5):

    model.train()  
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(x)
            prob = torch.sigmoid(logits)
            preds.append(prob.cpu().numpy())
    preds = np.stack(preds, axis=0)  
    mean_pred = preds.mean(axis=0)   
    var_pred = preds.var(axis=0)    
    return mean_pred, var_pred


if __name__ == "__main__":
    set_seed(1)
    data_dir = "./data"
    pairs = gather_image_mask_pairs(data_dir, image_suffix="_crop.nii.gz")
    print("Total pairs found:", len(pairs))

    train_list, val_list = split_data(pairs, train_ratio=0.8)
    print("Train samples:", len(train_list))
    print("Val samples:", len(val_list))

    train_ventricles_transformer(
        train_list=train_list,
        val_list=val_list,
        patch_size=(64,64,64),
        num_patches_per_volume=5,
        epochs=40,
        batch_size=1,
        lr=1e-4,
        augment=True,
        device="cuda",
        save_path="transformer_ventricles_best.pth",
        use_scheduler=False,
        early_stopping_patience=20
    )

