import os
import glob
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def compute_binary_metrics(pred, target, smooth=1e-5):
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)

    pred = pred.float().reshape(-1)
    target = target.float().reshape(-1)

    intersection = (pred * target).sum()
    union = (pred.sum() + target.sum())

    dice = (2.0 * intersection + smooth) / (union + smooth)
    iou  = (intersection + smooth) / (pred.sum() + target.sum() - intersection + smooth)

    # TP, FP, TN, FN
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    TN = ((1 - pred) * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    TP, FP, TN, FN = TP.float(), FP.float(), TN.float(), FN.float()

    precision   = (TP + smooth) / (TP + FP + smooth)
    recall      = (TP + smooth) / (TP + FN + smooth)  # sensitivity
    specificity = (TN + smooth) / (TN + FP + smooth)
    accuracy    = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)

    return {
        "dice":        dice.item(),
        "iou":         iou.item(),
        "precision":   precision.item(),
        "recall":      recall.item(),
        "specificity": specificity.item(),
        "accuracy":    accuracy.item()
    }

import glob

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


def generate_sliding_indices(D, H, W, patch_d=64, patch_h=64, patch_w=64):
    coords = []

    d_list = list(range(0, D, patch_d))
    if d_list[-1] + patch_d > D:
        d_list[-1] = max(D - patch_d, 0)

    h_list = list(range(0, H, patch_h))
    if h_list[-1] + patch_h > H:
        h_list[-1] = max(H - patch_h, 0)

    w_list = list(range(0, W, patch_w))
    if w_list[-1] + patch_w > W:
        w_list[-1] = max(W - patch_w, 0)

    for dd in d_list:
        for hh in h_list:
            for ww in w_list:
                d1, d2 = dd, dd + patch_d
                h1, h2 = hh, hh + patch_h
                w1, w2 = ww, ww + patch_w
                coords.append((d1,d2,h1,h2,w1,w2))
    return coords


# 推理函数 
def inference_3d_unet(
    model_path,
    test_list,
    save_result_dir=None,
    device="cuda:0",
    patch_size=(64,64,64)
):
    # 加载权重
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, out_channels=1, base_ch=32, dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if save_result_dir is not None:
        os.makedirs(save_result_dir, exist_ok=True)

    all_metrics = []

    with torch.no_grad():
        for (img_path, mask_path) in test_list:
            img_nib = nib.load(img_path)
            img_data = img_nib.get_fdata().astype(np.float32)

            # 归一化
            mn, mx = img_data.min(), img_data.max()
            if (mx - mn) > 1e-6:
                img_data = (img_data - mn) / (mx - mn)

            D,H,W = img_data.shape
            pred_volume = np.zeros((D,H,W), dtype=np.float32)

            # 滑窗分块，逐块推理
            coords_list = generate_sliding_indices(D, H, W,
                                                   patch_d=patch_size[0],
                                                   patch_h=patch_size[1],
                                                   patch_w=patch_size[2])

            for (d1,d2,h1,h2,w1,w2) in coords_list:
                patch_img = img_data[d1:d2, h1:h2, w1:w2]

                patch_tensor = torch.from_numpy(patch_img).unsqueeze(0).unsqueeze(0).float().to(device)

                logits = model(patch_tensor)  
                probs  = torch.sigmoid(logits)
                preds  = (probs > 0.5).float() 

                pred_np = preds.squeeze(0).squeeze(0).cpu().numpy()
                pred_volume[d1:d2, h1:h2, w1:w2] = pred_np

            if os.path.exists(mask_path):
                lbl_nib = nib.load(mask_path)
                lbl_data = lbl_nib.get_fdata().astype(np.float32)
                lbl_data[lbl_data >= 0.5] = 1
                lbl_data[lbl_data < 0.5]  = 0

                if lbl_data.shape == pred_volume.shape:
                    met = compute_binary_metrics(pred_volume, lbl_data)
                    all_metrics.append(met)
                else:
                    print(f"[Warning] label shape {lbl_data.shape} != pred shape {pred_volume.shape}, skip metrics...")

            if save_result_dir is not None:
                base_name = os.path.basename(img_path).replace(".nii.gz","")
                out_path  = os.path.join(save_result_dir, f"{base_name}_pred.nii.gz")

                pred_nib = nib.Nifti1Image(pred_volume.astype(np.float32), img_nib.affine)
                nib.save(pred_nib, out_path)
                print(f"Saved prediction: {out_path}")

    if len(all_metrics) > 0:
        avg = {}
        keys = all_metrics[0].keys()
        for k in keys:
            avg[k] = np.mean([m[k] for m in all_metrics])
        print("\n========== Final Test Metrics (Average) ==========")
        for k in keys:
            print(f"{k.capitalize()}: {avg[k]:.4f}")
        print("=================================================\n")
    else:
        print("No ground-truth labels found or shape mismatch. Done inference only.")

if __name__ == "__main__":
    data_dir = "./data"

    test_pairs = gather_image_mask_pairs(data_dir, image_suffix="_crop.nii.gz")
    print("Collected test samples:", len(test_pairs))

    inference_3d_unet(
        model_path="vnet_ventricles_best.pth",
        test_list=test_pairs,
        save_result_dir="./inference_outputs",
        device="cuda:0",            
        patch_size=(64,64,64)     
    )
