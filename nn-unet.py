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

# 随机种子
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



# 数据增广
def random_flip_3d(image, label):
    if random.random() < 0.5:
        image = image.flip(dims=[1]) 
        label = label.flip(dims=[1])
    if random.random() < 0.5:
        image = image.flip(dims=[2])  
        label = label.flip(dims=[2])
    if random.random() < 0.5:
        image = image.flip(dims=[3])  # 三个维度的翻转
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

    # 旋转矩阵 R = Rz * Ry * Rx
    cx, sx = math.cos(rx), math.sin(rx)
    Rx = np.array([
        [1,    0,   0],
        [0,   cx,  -sx],
        [0,   sx,   cx]
    ])
    cy, sy = math.cos(ry), math.sin(ry)
    Ry = np.array([
        [ cy,  0,   sy],
        [  0,  1,   0],
        [-sy,  0,   cy]
    ])
    cz, sz = math.cos(rz), math.sin(rz)
    Rz = np.array([
        [ cz, -sz,   0],
        [ sz,  cz,   0],
        [  0,   0,   1]
    ])
    R = Rz @ Ry @ Rx

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


# 二分类
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff


# 计算指标的函数
def compute_metrics(logits, targets, smooth=1e-5):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == targets).float().mean()

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)

    return acc.item(), dice.item()


# nnUNet
def conv_instancenorm_lrelu(in_ch, out_ch):
    """
    (Conv3D -> InstanceNorm3d -> LeakyReLU)
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_ch),
        nn.LeakyReLU(negative_slope=0.01, inplace=True)
    )

class DoubleConv(nn.Module):
    """
    两次 conv_instancenorm_lrelu
    """
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            conv_instancenorm_lrelu(in_ch, out_ch),
            conv_instancenorm_lrelu(out_ch, out_ch)
        )
    def forward(self, x):
        return self.block(x)

class nnUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32, deep_supervision=True):
        super(nnUNet3D, self).__init__()
        self.deep_supervision = deep_supervision

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.down1 = nn.Conv3d(base_ch, base_ch*2, kernel_size=2, stride=2)

        self.enc2 = DoubleConv(base_ch*2, base_ch*2)
        self.down2 = nn.Conv3d(base_ch*2, base_ch*4, kernel_size=2, stride=2)

        self.enc3 = DoubleConv(base_ch*4, base_ch*4)
        self.down3 = nn.Conv3d(base_ch*4, base_ch*8, kernel_size=2, stride=2)

        self.enc4 = DoubleConv(base_ch*8, base_ch*8)
        self.down4 = nn.Conv3d(base_ch*8, base_ch*16, kernel_size=2, stride=2)

        # Bottleneck
        self.enc5 = DoubleConv(base_ch*16, base_ch*16)

        # Decoder
        self.up4 = nn.ConvTranspose3d(base_ch*16, base_ch*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_ch*16, base_ch*8)

        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch*2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, out_channels, kernel_size=1)

        # 监督的分支
        if deep_supervision:
            self.ds4 = nn.Conv3d(base_ch*8, out_channels, kernel_size=1)
            self.ds3 = nn.Conv3d(base_ch*4, out_channels, kernel_size=1)
            self.ds2 = nn.Conv3d(base_ch*2, out_channels, kernel_size=1)
            

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   
        p1 = self.down1(e1) 

        e2 = self.enc2(p1)  
        p2 = self.down2(e2) 

        e3 = self.enc3(p2)  
        p3 = self.down3(e3) 

        e4 = self.enc4(p3)  
        p4 = self.down4(e4)

        b  = self.enc5(p4)  
        # Decoder
        up4 = self.up4(b)               
        cat4 = torch.cat([up4, e4], 1)  
        d4 = self.dec4(cat4)           

        up3 = self.up3(d4)             
        cat3 = torch.cat([up3, e3], 1)
        d3 = self.dec3(cat3)           

        up2 = self.up2(d3)
        cat2 = torch.cat([up2, e2], 1)
        d2 = self.dec2(cat2)          

        up1 = self.up1(d2)
        cat1 = torch.cat([up1, e1], 1)
        d1 = self.dec1(cat1)           

        out_main = self.out_conv(d1)  
        if self.deep_supervision:
            ds_out4 = self.ds4(d4)     
            ds_out3 = self.ds3(d3)    
            ds_out2 = self.ds2(d2)    
            return out_main, ds_out4, ds_out3, ds_out2
        else:
            return out_main


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
        d1 = random.randint(0, max(D - d_size, 0)) if D>=d_size else 0
        h1 = random.randint(0, max(H - h_size, 0)) if H>=h_size else 0
        w1 = random.randint(0, max(W - w_size, 0)) if W>=w_size else 0

        d2, h2, w2 = d1 + d_size, h1 + h_size, w1 + w_size

        patch_img = img_data[d1:d2, h1:h2, w1:w2]
        patch_lbl = lbl_data[d1:d2, h1:h2, w1:w2]

        patch_img = np.expand_dims(patch_img, axis=0)  # (1,D,H,W)
        patch_lbl = np.expand_dims(patch_lbl, axis=0)

        patch_img = torch.from_numpy(patch_img).float()
        patch_lbl = torch.from_numpy(patch_lbl).float()

        # 数据增广
        if self.augment:
            patch_img, patch_lbl = random_flip_3d(patch_img, patch_lbl)
            patch_img, patch_lbl = random_3d_rotation(patch_img, patch_lbl, max_angle=self.max_angle)

        return patch_img, patch_lbl



class MultiOutputLoss(nn.Module):
    # 对每个分支计算 (Dice + BCE)，再用不同权重加权求和。
    def __init__(self, weights=[1.0, 0.5, 0.25, 0.125], smooth=1e-5):
        super(MultiOutputLoss, self).__init__()
        self.weights = weights
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, logits, target):
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth)
        return 1.0 - dice_coeff

    def forward(self, net_outputs, labels):
        """
        net_outputs: (out_main, ds_out4, ds_out3, ds_out2)
        labels: (B,1,D,H,W)
        """
        total_loss = 0.0
        for i, out in enumerate(net_outputs):
            out_shape = out.shape[2:]  
            # 标签做 nearest 下采样
            lbl_down = F.interpolate(labels, size=out_shape, mode='nearest')

            dice_l = self.dice_loss(out, lbl_down)
            bce_l  = self.bce(out, lbl_down)

            w = self.weights[i] if i < len(self.weights) else 0.0
            total_loss += w * (dice_l + bce_l)

        return total_loss


def train_3d_nnunet(
    train_list,
    val_list=None,
    patch_size=(64,64,64),
    num_patches_per_volume=10,
    epochs=50,
    batch_size=2,
    lr=1e-3,
    augment=True,
    max_angle=15,
    deep_supervision=True,
    save_path="nnunet_ventricles_best.pth",
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

    model = nnUNet3D(
        in_channels=1,
        out_channels=1,
        base_ch=32,
        deep_supervision=deep_supervision
    ).to(device)

    if deep_supervision:
        criterion = MultiOutputLoss(weights=[1.0, 0.5, 0.25, 0.125])
    else:
        dice_loss_fn = DiceLoss()
        bce_loss_fn  = nn.BCEWithLogitsLoss()
        def single_output_loss(logits, labels):
            return dice_loss_fn(logits, labels) + bce_loss_fn(logits, labels)
        criterion = single_output_loss

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    # 记录指标
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
        total_acc  = 0.0
        total_dice = 0.0
        count_batch= 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks= masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs) 

            if deep_supervision:
                loss = criterion(outputs, masks)
                main_out = outputs[0]  
            else:
                loss = criterion(outputs, masks)
                main_out = outputs

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count_batch += 1

            acc, dice = compute_metrics(main_out, masks)
            total_acc  += acc
            total_dice += dice

        avg_train_loss = total_loss / count_batch
        avg_train_acc  = total_acc  / count_batch
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
            val_count_batch= 0

            with torch.no_grad():
                for val_imgs, val_masks in val_loader:
                    val_imgs = val_imgs.to(device)
                    val_masks= val_masks.to(device)

                    val_outs = model(val_imgs)
                    if deep_supervision:
                        vloss = criterion(val_outs, val_masks)
                        main_out = val_outs[0]
                    else:
                        vloss = criterion(val_outs, val_masks)
                        main_out = val_outs

                    val_total_loss += vloss.item()
                    val_count_batch += 1

                    acc, dice = compute_metrics(main_out, val_masks)
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
                    print(f"验证集Dice连续 {early_stopping_patience} 个epoch无提升, 提前结束训练.")
                    break

        else:
            torch.save(model.state_dict(), save_path)
            print(f"[Epoch {epoch}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, Dice: {avg_train_dice:.4f}")

    if val_loader is not None:
        print(f"Best val dice: {best_val_dice:.4f} (epoch={best_epoch}), model saved to {save_path}")
    else:
        print(f"No validation set. Model saved to {save_path}")


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
    plt.title("3D nnU-Net Training Curves (Loss / Accuracy / Dice)")
    plt.legend()
    plt.savefig("nnunet_3d_training_metrics.png")
    plt.show()


if __name__ == "__main__":
    set_seed(1)

    data_dir = "./data"

    pairs = gather_image_mask_pairs(data_dir=data_dir, image_suffix="_crop.nii.gz")
    print("Total paired samples:", len(pairs))

    train_list, val_list = split_data(pairs, train_ratio=0.8)
    print("Train samples:", len(train_list))
    print("Val samples:", len(val_list))

    train_3d_nnunet(
        train_list=train_list,
        val_list=val_list,
        patch_size=(64,64,64),
        num_patches_per_volume=10,
        epochs=50,
        batch_size=2,
        lr=1e-3,
        augment=True,
        max_angle=15,
        deep_supervision=True,        
        save_path="nnunet_ventricles_best.pth",
        device="cuda:0",
        use_scheduler=True,           
        early_stopping_patience=10    
    )
