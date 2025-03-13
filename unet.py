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

def random_rotate_3d(image, label, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    theta = math.radians(angle)

    c, d, h, w = image.shape
    center_x = (h - 1) / 2.0
    center_y = (w - 1) / 2.0

    rotated_image = torch.zeros_like(image)
    rotated_label = torch.zeros_like(label)

    device = image.device
    cos_val = math.cos(theta)
    sin_val = math.sin(theta)

    for z in range(d):
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        xx_t = xx - center_x
        yy_t = yy - center_y

        xx_r = cos_val * xx_t - sin_val * yy_t + center_x
        yy_r = sin_val * xx_t + cos_val * yy_t + center_y

        xx_norm = 2.0 * xx_r / (w - 1) - 1.0
        yy_norm = 2.0 * yy_r / (h - 1) - 1.0

        grid = torch.stack((xx_norm, yy_norm), dim=-1).unsqueeze(0)  

        slice_img = image[:, z, :, :].unsqueeze(0)
        slice_lbl = label[:, z, :, :].unsqueeze(0)

        rotated_slice_img = F.grid_sample(slice_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        rotated_slice_lbl = F.grid_sample(slice_lbl, grid, mode='nearest',  padding_mode='border', align_corners=True)

        rotated_image[:, z, :, :] = rotated_slice_img[0]
        rotated_label[:, z, :, :] = rotated_slice_lbl[0]

    return rotated_image, rotated_label

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
        # Acc
        acc = (preds == targets).float().mean()

        # Dice 
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)

    return acc.item(), dice.item()

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout3d(p=dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32, dropout_p=0.0):
        super(UNet3D, self).__init__()

        self.enc1 = ConvBlock3D(in_channels, base_ch, dropout_p=dropout_p)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock3D(base_ch, base_ch*2, dropout_p=dropout_p)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock3D(base_ch*2, base_ch*4, dropout_p=dropout_p)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock3D(base_ch*4, base_ch*8, dropout_p=dropout_p)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock3D(base_ch*8, base_ch*16, dropout_p=dropout_p)

        self.up4 = nn.ConvTranspose3d(base_ch*16, base_ch*8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(base_ch*16, base_ch*8, dropout_p=dropout_p)

        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_ch*8, base_ch*4, dropout_p=dropout_p)

        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_ch*4, base_ch*2, dropout_p=dropout_p)

        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_ch*2, base_ch, dropout_p=dropout_p)

        self.out_conv = nn.Conv3d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        up4 = self.up4(b)
        cat4 = torch.cat([up4, e4], dim=1)
        d4 = self.dec4(cat4)

        up3 = self.up3(d4)
        cat3 = torch.cat([up3, e3], dim=1)
        d3 = self.dec3(cat3)

        up2 = self.up2(d3)
        cat2 = torch.cat([up2, e2], dim=1)
        d2 = self.dec2(cat2)

        up1 = self.up1(d2)
        cat1 = torch.cat([up1, e1], dim=1)
        d1 = self.dec1(cat1)

        out = self.out_conv(d1)
        return out

class VentriclesDataset(Dataset):
    def __init__(
        self,
        pairs_list,                
        patch_size=(64,64,64),
        num_patches_per_volume=10,
        augment=False
    ):
        self.pairs_list = pairs_list
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.augment = augment

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
            patch_img, patch_lbl = random_rotate_3d(patch_img, patch_lbl, max_angle=15)

        return patch_img, patch_lbl

def train_3d_unet(
    train_list,
    val_list=None,
    patch_size=(64,64,64),
    num_patches_per_volume=10,
    epochs=50,
    batch_size=2,
    lr=1e-3,
    augment=True,
    dropout_p=0.0,
    save_path="unet_ventricles_final.pth",
    device="cuda",
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
    if val_list and len(val_list) > 0:
        val_dataset = VentriclesDataset(
            pairs_list=val_list,
            patch_size=patch_size,
            num_patches_per_volume=2, 
            augment=False
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    model = UNet3D(in_channels=1, out_channels=1, base_ch=32, dropout_p=dropout_p).to(device)
    dice_loss_fn = DiceLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器
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

            # Dice + BCE
            dice_l = dice_loss_fn(outputs, masks)
            bce_l = bce_loss_fn(outputs, masks)
            loss = dice_l + bce_l

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count_batch += 1
            acc, dice = compute_metrics(outputs, masks)
            total_acc += acc
            total_dice += dice

        avg_train_loss = total_loss / count_batch
        avg_train_acc = total_acc / count_batch
        avg_train_dice = total_dice / count_batch

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        train_dice_list.append(avg_train_dice)

        if scheduler is not None:
            scheduler.step()

        # 验证
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_total_acc = 0.0
            val_total_dice = 0.0
            val_count_batch = 0

            with torch.no_grad():
                for val_imgs, val_masks in val_loader:
                    val_imgs = val_imgs.to(device)
                    val_masks = val_masks.to(device)

                    val_outputs = model(val_imgs)
                    vdice_l = dice_loss_fn(val_outputs, val_masks)
                    vbce_l = bce_loss_fn(val_outputs, val_masks)
                    vloss = vdice_l + vbce_l

                    val_total_loss += vloss.item()
                    val_count_batch += 1

                    acc, dice = compute_metrics(val_outputs, val_masks)
                    val_total_acc += acc
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
                    print(f"Val Dice 连续 {early_stopping_patience} 次没有提升, 提前停止训练.")
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
    plt.title("Training Curves (Loss / Accuracy / Dice)")
    plt.legend()
    plt.savefig("training_metrics_unet.png")
    plt.show()

if __name__ == "__main__":
    set_seed(1)

    data_dir = "./data"  
    pairs = gather_image_mask_pairs(data_dir=data_dir, image_suffix="_crop.nii.gz")
    print("Total paired samples:", len(pairs))

    train_list, val_list = split_data(pairs, train_ratio=0.8)
    print("Train samples:", len(train_list))
    print("Val samples:", len(val_list))

    train_3d_unet(
        train_list=train_list,
        val_list=val_list,
        patch_size=(64,64,64),
        num_patches_per_volume=10,
        epochs=50,
        batch_size=2,
        lr=1e-3,
        augment=True,               
        dropout_p=0.1,              
        save_path="unet_ventricles_final.pth",
        device="cuda:0",          
        use_scheduler=True,        
        early_stopping_patience=20
    )
