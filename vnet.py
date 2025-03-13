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
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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

    cx, sx = math.cos(rx), math.sin(rx)
    Rx = np.array([
        [1,   0,    0],
        [0,  cx,  -sx],
        [0,  sx,   cx]
    ])
    # Ry
    cy, sy = math.cos(ry), math.sin(ry)
    Ry = np.array([
        [ cy,  0,  sy],
        [  0,  1,   0],
        [-sy,  0,  cy]
    ])
    # Rz
    cz, sz = math.cos(rz), math.sin(rz)
    Rz = np.array([
        [ cz, -sz,  0],
        [ sz,  cz,  0],
        [  0,   0,  1]
    ])

    R = Rz @ Ry @ Rx

    c_trans = center - R @ center

    # 图像三线性插值
    rotated_img = affine_transform(
        img_np,
        R,
        offset=c_trans,
        order=1,   
        mode='nearest'
    )
    # 标签最近邻插值
    rotated_lbl = affine_transform(
        lbl_np,
        R,
        offset=c_trans,
        order=0,   # 最近邻
        mode='nearest'
    )

    rotated_img = torch.from_numpy(rotated_img).unsqueeze(0)
    rotated_lbl = torch.from_numpy(rotated_lbl).unsqueeze(0)

    device = image.device
    rotated_img = rotated_img.to(device, dtype=image.dtype)
    rotated_lbl = rotated_lbl.to(device, dtype=label.dtype)
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


class VNetResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super(VNetResidualBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(out_ch)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(out_ch)

        self.dropout = nn.Dropout3d(p=dropout_p)
        self.skip_conv = None
        if in_ch != out_ch:
            self.skip_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_conv is not None:
            residual = self.skip_conv(residual)

        out = out + residual
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        return out

def make_n_res_blocks(in_ch, out_ch, n_block, dropout_p=0.0):
    blocks = []
    # 第一个块 
    blocks.append(VNetResidualBlock(in_ch, out_ch, dropout_p=dropout_p))
    # 后面的块
    for _ in range(n_block-1):
        blocks.append(VNetResidualBlock(out_ch, out_ch, dropout_p=dropout_p))
    return nn.Sequential(*blocks)

class VNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_p=0.1):
        super(VNet3D, self).__init__()
        # Encoder
        # C=16, 1个残差块
        self.enc_level1_pre  = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1, stride=1)
        self.enc_level1_bn   = nn.BatchNorm3d(16)
        self.enc_level1_block= make_n_res_blocks(16, 16, n_block=1, dropout_p=dropout_p)

        # 下采样1
        self.down_conv1 = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.down_bn1   = nn.BatchNorm3d(32)
        # C=32, 2个残差块
        self.enc_level2_block= make_n_res_blocks(32, 32, n_block=2, dropout_p=dropout_p)

        # 下采样2
        self.down_conv2 = nn.Conv3d(32, 64, kernel_size=2, stride=2)
        self.down_bn2   = nn.BatchNorm3d(64)
        # C=64, 3个残差块
        self.enc_level3_block= make_n_res_blocks(64, 64, n_block=3, dropout_p=dropout_p)

        # 下采样3
        self.down_conv3 = nn.Conv3d(64, 128, kernel_size=2, stride=2)
        self.down_bn3   = nn.BatchNorm3d(128)
        # C=128, 3个残差块
        self.enc_level4_block= make_n_res_blocks(128, 128, n_block=3, dropout_p=dropout_p)

        # 下采样4
        self.down_conv4 = nn.Conv3d(128, 256, kernel_size=2, stride=2)
        self.down_bn4   = nn.BatchNorm3d(256)
        # C=256, 3个残差块 (bottleneck)
        self.enc_level5_block= make_n_res_blocks(256, 256, n_block=3, dropout_p=dropout_p)

        # Decoder 
        self.up4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.up4_bn = nn.BatchNorm3d(128)
        self.dec_level4_block = make_n_res_blocks(256, 128, n_block=3, dropout_p=dropout_p)

        # 128->64
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up3_bn = nn.BatchNorm3d(64)
        self.dec_level3_block = make_n_res_blocks(128, 64, n_block=3, dropout_p=dropout_p)

        # 64->32
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.up2_bn = nn.BatchNorm3d(32)
        self.dec_level2_block = make_n_res_blocks(64, 32, n_block=2, dropout_p=dropout_p)

        # 32->16
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.up1_bn = nn.BatchNorm3d(16)
        self.dec_level1_block = make_n_res_blocks(32, 16, n_block=1, dropout_p=dropout_p)

        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc_level1_pre(x)    
        x1 = self.enc_level1_bn(x1)
        x1 = F.relu(x1, inplace=True)
        x1_out = self.enc_level1_block(x1)  

        d1 = self.down_conv1(x1_out)  
        d1 = self.down_bn1(d1)
        d1 = F.relu(d1, inplace=True)
        x2_out = self.enc_level2_block(d1)  

        d2 = self.down_conv2(x2_out)  
        d2 = self.down_bn2(d2)
        d2 = F.relu(d2, inplace=True)
        x3_out = self.enc_level3_block(d2)

        d3 = self.down_conv3(x3_out)  
        d3 = self.down_bn3(d3)
        d3 = F.relu(d3, inplace=True)
        x4_out = self.enc_level4_block(d3)

        d4 = self.down_conv4(x4_out)  
        d4 = self.down_bn4(d4)
        d4 = F.relu(d4, inplace=True)
        x5_out = self.enc_level5_block(d4) 

        u4 = self.up4(x5_out)         
        u4 = self.up4_bn(u4)
        u4 = F.relu(u4, inplace=True)
        cat4 = torch.cat([u4, x4_out], dim=1)  
        y4_out = self.dec_level4_block(cat4)   

        u3 = self.up3(y4_out)         
        u3 = self.up3_bn(u3)
        u3 = F.relu(u3, inplace=True)
        cat3 = torch.cat([u3, x3_out], dim=1)   
        y3_out = self.dec_level3_block(cat3)    

        u2 = self.up2(y3_out)        
        u2 = self.up2_bn(u2)
        u2 = F.relu(u2, inplace=True)
        cat2 = torch.cat([u2, x2_out], dim=1)   
        y2_out = self.dec_level2_block(cat2)    

        u1 = self.up1(y2_out)         
        u1 = self.up1_bn(u1)
        u1 = F.relu(u1, inplace=True)
        cat1 = torch.cat([u1, x1_out], dim=1)   
        y1_out = self.dec_level1_block(cat1)    

        out = self.out_conv(y1_out)   
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

            # 归一化
            mn, mx = img_data.min(), img_data.max()
            if mx - mn > 1e-6:
                img_data = (img_data - mn) / (mx - mn)

            # 二值化标签
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
        d1 = random.randint(0, max(D - d_size, 0)) if D >= d_size else 0
        h1 = random.randint(0, max(H - h_size, 0)) if H >= h_size else 0
        w1 = random.randint(0, max(W - w_size, 0)) if W >= w_size else 0

        d2 = d1 + d_size
        h2 = h1 + h_size
        w2 = w1 + w_size

        patch_img = img_data[d1:d2, h1:h2, w1:w2]
        patch_lbl = lbl_data[d1:d2, h1:h2, w1:w2]

        patch_img = np.expand_dims(patch_img, axis=0)  
        patch_lbl = np.expand_dims(patch_lbl, axis=0)

        patch_img = torch.from_numpy(patch_img).float()
        patch_lbl = torch.from_numpy(patch_lbl).float()

        # 数据增广
        if self.augment:
            patch_img, patch_lbl = random_flip_3d(patch_img, patch_lbl)
            patch_img, patch_lbl = random_3d_rotation(patch_img, patch_lbl, max_angle=self.max_angle)

        return patch_img, patch_lbl

def train_3d_vnet(
    train_list,
    val_list=None,
    patch_size=(64,64,64),
    num_patches_per_volume=10,
    epochs=50,
    batch_size=2,
    lr=1e-3,
    augment=True,
    max_angle=15,
    dropout_p=0.1,
    save_path="vnet_ventricles.pth",
    device="cuda",
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

    model = VNet3D(in_channels=1, out_channels=1, dropout_p=dropout_p).to(device)

    dice_loss_fn = DiceLoss()
    bce_loss_fn  = nn.BCEWithLogitsLoss()
    optimizer    = optim.Adam(model.parameters(), lr=lr)

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

    best_val_dice   = 0.0
    best_epoch      = 0
    no_improve_count= 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_dice = 0.0
        count_batch = 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            # (Dice + BCE)
            dice_l = dice_loss_fn(outputs, masks)
            bce_l  = bce_loss_fn(outputs, masks)
            loss   = dice_l + bce_l

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count_batch += 1

            acc, dice = compute_metrics(outputs, masks)
            total_acc  += acc
            total_dice += dice

        avg_train_loss = total_loss / count_batch
        avg_train_acc  = total_acc / count_batch
        avg_train_dice = total_dice / count_batch

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        train_dice_list.append(avg_train_dice)

        if scheduler:
            scheduler.step()

        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_total_acc  = 0.0
            val_total_dice = 0.0
            val_count_batch = 0

            with torch.no_grad():
                for val_imgs, val_masks in val_loader:
                    val_imgs = val_imgs.to(device)
                    val_masks = val_masks.to(device)

                    val_out = model(val_imgs)
                    vdice_l = dice_loss_fn(val_out, val_masks)
                    vbce_l  = bce_loss_fn(val_out, val_masks)
                    vloss   = vdice_l + vbce_l

                    val_total_loss += vloss.item()
                    val_count_batch += 1

                    acc, dice = compute_metrics(val_out, val_masks)
                    val_total_acc  += acc
                    val_total_dice += dice

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
                no_improve_count = 0
            else:
                no_improve_count += 1

            print(f"[Epoch {epoch}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, Dice: {avg_train_dice:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, Dice: {avg_val_dice:.4f}")

            # 早停
            if early_stopping_patience is not None:
                if no_improve_count >= early_stopping_patience:
                    print(f"验证集Dice连续 {early_stopping_patience} 次未提升, 提前停止训练.")
                    break

        else:
            torch.save(model.state_dict(), save_path)
            print(f"[Epoch {epoch}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, Dice: {avg_train_dice:.4f}")

    if val_loader is not None:
        print(f"Best val dice: {best_val_dice:.4f} (epoch {best_epoch}), model saved to {save_path}")
    else:
        print(f"Model saved to {save_path} (no validation set used)")

    epochs_range = range(1, len(train_loss_list)+1)
    plt.figure()

    plt.plot(epochs_range, train_loss_list, label="Train Loss")
    if val_loader is not None:
        plt.plot(epochs_range, val_loss_list, label="Val Loss")

    plt.plot(epochs_range, train_acc_list, label="Train Acc")
    if val_loader is not None:
        plt.plot(epochs_range, val_acc_list, label="Val Acc")

    plt.plot(epochs_range, train_dice_list, label="Train Dice")
    if val_loader is not None:
        plt.plot(epochs_range, val_dice_list, label="Val Dice")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("VNet 3D Training (Loss / Accuracy / Dice)")
    plt.legend()
    plt.savefig("vnet_training_metrics.png")
    plt.show()

if __name__ == "__main__":
    set_seed(1)

    data_dir = "./data"

    pairs = gather_image_mask_pairs(data_dir=data_dir, image_suffix="_crop.nii.gz")
    print("Total paired samples:", len(pairs))

    train_list, val_list = split_data(pairs, train_ratio=0.8)
    print("Train samples:", len(train_list))
    print("Val samples:", len(val_list))

    train_3d_vnet(
        train_list=train_list,
        val_list=val_list,
        patch_size=(64,64,64),
        num_patches_per_volume=10,
        epochs=50,
        batch_size=2,
        lr=1e-3,
        augment=True,
        max_angle=15,
        dropout_p=0.1,              # 残差块中加Dropout
        save_path="vnet_ventricles_best.pth",
        device="cuda:0",
        use_scheduler=True,         # StepLR
        early_stopping_patience=10  
    )
