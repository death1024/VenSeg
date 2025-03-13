import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.cluster import KMeans


def set_seed(seed=2023):
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

#划分训练验证
def split_data(pairs, train_ratio=0.8):
    random.shuffle(pairs)
    train_size = int(len(pairs)*train_ratio)
    train_list = pairs[:train_size]
    val_list   = pairs[train_size:]
    return train_list, val_list


class Brain3DDataset(Dataset):
    def __init__(self, pairs, transform=None):
        super().__init__()
        self.transform = transform
        self.data_list = []  # 存放元信息

        for img_path, msk_path in pairs:
            self.data_list.append((img_path, msk_path))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, msk_path = self.data_list[idx]
        # 读取 NIfTI
        img_nii = nib.load(img_path)
        msk_nii = nib.load(msk_path)

        img_data = img_nii.get_fdata()  # shape: [Dx, Hy, Wz] 
        msk_data = msk_nii.get_fdata()

        masked_data = img_data * (msk_data > 0)

        # 归一化：减均值 / 标准差
        mean_val = masked_data.mean()
        std_val = masked_data.std()
        if std_val > 1e-5:
            masked_data = (masked_data - mean_val) / (std_val + 1e-8)

        # 转成 [1, D, H, W]
        vol_tensor = torch.tensor(masked_data, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            vol_tensor = self.transform(vol_tensor)

        return vol_tensor


class ResBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_channels)

        # 如果 in/out 不匹配或 stride>1，需要投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(Encoder3D, self).__init__()
        # 第一级
        self.layer1 = ResBlock3D(in_channels, base_channels, stride=1)
        # 下采样 + 第二级
        self.layer2 = ResBlock3D(base_channels, base_channels*2, stride=2)
        # 下采样 + 第三级
        self.layer3 = ResBlock3D(base_channels*2, base_channels*4, stride=2)
        # 下采样 + 第四级
        self.layer4 = ResBlock3D(base_channels*4, base_channels*8, stride=2)

    def forward(self, x):
        # x: [B, 1, D, H, W]
        x1 = self.layer1(x)  
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]

class Decoder3D(nn.Module):
    def __init__(self, base_channels=32):
        super(Decoder3D, self).__init__()
        self.up1 = nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.res1 = ResBlock3D(base_channels*8, base_channels*4, stride=1)

        self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.res2 = ResBlock3D(base_channels*4, base_channels*2, stride=1)

        self.up3 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.res3 = ResBlock3D(base_channels*2, base_channels, stride=1)

        self.out_conv = nn.Conv3d(base_channels, 1, kernel_size=1, stride=1)  
    
    def forward(self, features):
        # features = [x1, x2, x3, x4]
        x1, x2, x3, x4 = features

        d1 = self.up1(x4)  
        cat1 = torch.cat([d1, x3], dim=1)  # 用一个跳跃连接
        d1 = self.res1(cat1)

        d2 = self.up2(d1)
        cat2 = torch.cat([d2, x2], dim=1)
        d2 = self.res2(cat2)

        d3 = self.up3(d2)
        cat3 = torch.cat([d3, x1], dim=1)
        d3 = self.res3(cat3)

        out = self.out_conv(d3)  # [B, 1, D, H, W]
        return out

class ResUNet3DAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, latent_dim=128):
        super(ResUNet3DAutoencoder, self).__init__()
        self.encoder = Encoder3D(in_channels, base_channels)

        # 用一个AdaptiveAvgPool + FC
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1)) 
        self.fc_enc = nn.Linear(base_channels*8, latent_dim)

        # 再用一个FC解码回去
        self.fc_dec = nn.Linear(latent_dim, base_channels*8)

        self.decoder = Decoder3D(base_channels)

    def encode(self, x):
        feats = self.encoder(x)
        x4 = feats[-1]  

        # 同样全局池化 + FC
        pooled = self.global_pool(x4)               
        pooled = pooled.view(pooled.size(0), -1)   
        z = self.fc_enc(pooled)                 

        return feats, z

    def decode(self, feats, z):
        x4 = feats[-1].clone()

        # 把latent向量映射回 x4 的通道数
        z_dec = self.fc_dec(z)      
        z_dec = z_dec.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        x4 = x4 + z_dec
        feats[-1] = x4
        out = self.decoder(feats)
        return out

    def forward(self, x):
        feats, z = self.encode(x)
        x_hat = self.decode(feats, z)
        return x_hat

def train_autoencoder(model, train_loader, val_loader=None, epochs=20, lr=1e-4, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for vol in train_loader:
            vol = vol.to(device) 

            optimizer.zero_grad()
            recon = model(vol)  
            loss = criterion(recon, vol)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * vol.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        msg = f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.6f}"

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vvol in val_loader:
                    vvol = vvol.to(device)
                    vrecon = model(vvol)
                    vloss = criterion(vrecon, vvol)
                    val_loss += vloss.item() * vvol.size(0)
            val_loss /= len(val_loader.dataset)
            msg += f" | Val Loss: {val_loss:.6f}"

        print(msg)


#提取+聚类
def extract_latent_features(model, dataloader, device='cuda'):
    model.eval()
    features = []
    with torch.no_grad():
        for vol in dataloader:
            vol = vol.to(device)
            feats, z = model.encode(vol)  
            z = z.cpu().numpy()
            features.append(z)
    features = np.concatenate(features, axis=0)  
    return features

def cluster_kmeans(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans.labels_

def main():
    set_seed(2023)
    data_dir = "/data"
    pairs = gather_image_mask_pairs(data_dir, image_suffix="_crop.nii.gz")

    # 划分训练验证
    train_list, val_list = split_data(pairs, train_ratio=0.8)
    print(f"Total pairs: {len(pairs)} | Train: {len(train_list)} | Val: {len(val_list)}")

    # 构建一下dataset和loader
    train_ds = Brain3DDataset(train_list)
    val_ds   = Brain3DDataset(val_list)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 模型训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResUNet3DAutoencoder(in_channels=1, base_channels=16, latent_dim=128) 

    train_autoencoder(model, train_loader, val_loader, epochs=20, lr=1e-4, device=device)

    # 提取特征+聚类
    train_feats = extract_latent_features(model, train_loader, device=device)
    print("Train features shape:", train_feats.shape)

    n_clusters = 3
    train_labels = cluster_kmeans(train_feats, n_clusters=n_clusters)
    print("KMeans on Train set -> labels:", train_labels.shape)

    unique_labels, counts = np.unique(train_labels, return_counts=True)
    print("Cluster distribution:", dict(zip(unique_labels, counts)))

    val_feats = extract_latent_features(model, val_loader, device=device)
    val_labels = cluster_kmeans(val_feats, n_clusters=n_clusters)
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    print("[Val] Cluster distribution:", dict(zip(unique_val, counts_val)))

if __name__ == "__main__":
    main()
