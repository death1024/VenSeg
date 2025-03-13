import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.enc_level1_pre   = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1, stride=1)
        self.enc_level1_bn    = nn.BatchNorm3d(16)
        self.enc_level1_block = make_n_res_blocks(16, 16, n_block=1, dropout_p=dropout_p)

        self.down_conv1 = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.down_bn1   = nn.BatchNorm3d(32)
        self.enc_level2_block = make_n_res_blocks(32, 32, n_block=2, dropout_p=dropout_p)

        self.down_conv2 = nn.Conv3d(32, 64, kernel_size=2, stride=2)
        self.down_bn2   = nn.BatchNorm3d(64)
        self.enc_level3_block = make_n_res_blocks(64, 64, n_block=3, dropout_p=dropout_p)

        self.down_conv3 = nn.Conv3d(64, 128, kernel_size=2, stride=2)
        self.down_bn3   = nn.BatchNorm3d(128)
        self.enc_level4_block = make_n_res_blocks(128, 128, n_block=3, dropout_p=dropout_p)

        self.down_conv4 = nn.Conv3d(128, 256, kernel_size=2, stride=2)
        self.down_bn4   = nn.BatchNorm3d(256)
        self.enc_level5_block = make_n_res_blocks(256, 256, n_block=3, dropout_p=dropout_p)

        # Decoder
        self.up4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.up4_bn = nn.BatchNorm3d(128)
        self.dec_level4_block = make_n_res_blocks(128+128, 128, n_block=3, dropout_p=dropout_p)

        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up3_bn = nn.BatchNorm3d(64)
        self.dec_level3_block = make_n_res_blocks(64+64, 64, n_block=3, dropout_p=dropout_p)

        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.up2_bn = nn.BatchNorm3d(32)
        self.dec_level2_block = make_n_res_blocks(32+32, 32, n_block=2, dropout_p=dropout_p)

        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.up1_bn = nn.BatchNorm3d(16)
        self.dec_level1_block = make_n_res_blocks(16+16, 16, n_block=1, dropout_p=dropout_p)

        # 输出层
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
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

        # Decoder
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

def compute_binary_metrics(pred, target, smooth=1e-5):
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)
    pred   = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0*intersection + smooth) / (union + smooth)
    iou  = (intersection + smooth) / (pred.sum() + target.sum() - intersection + smooth)

    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    TN = ((1 - pred) * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    TP,FP,TN,FN = TP.float(), FP.float(), TN.float(), FN.float()

    precision   = (TP + smooth)/(TP + FP + smooth)
    recall      = (TP + smooth)/(TP + FN + smooth)  # sensitivity
    specificity = (TN + smooth)/(TN + FP + smooth)
    accuracy    = (TP + TN + smooth)/(TP + TN + FP + FN + smooth)

    return {
        "dice":        dice.item(),
        "iou":         iou.item(),
        "precision":   precision.item(),
        "recall":      recall.item(),
        "specificity": specificity.item(),
        "accuracy":    accuracy.item()
    }

def gather_image_mask_pairs(data_dir, image_suffix="_crop.nii.gz"):
    import glob
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
                coords.append((dd, dd+patch_d, hh, hh+patch_h, ww, ww+patch_w))
    return coords

# 推理函数
def inference_3d_vnet(
    model_path,
    test_list,
    save_result_dir=None,
    device="cuda:0",
    patch_size=(64,64,64)
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = VNet3D(in_channels=1, out_channels=1, dropout_p=0.1).to(device)  
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
            pred_vol = np.zeros((D,H,W), dtype=np.float32)

            # 滑窗分块
            coords_list = generate_sliding_indices(D,H,W,
                                                   patch_d=patch_size[0],
                                                   patch_h=patch_size[1],
                                                   patch_w=patch_size[2])
            for (d1,d2,h1,h2,w1,w2) in coords_list:
                patch_img = img_data[d1:d2, h1:h2, w1:w2]
                patch_tensor = torch.from_numpy(patch_img).unsqueeze(0).unsqueeze(0).float().to(device)

                logits = model(patch_tensor)  
                probs  = torch.sigmoid(logits)
                preds  = (probs > 0.5).float()
                pred_np= preds.squeeze(0).squeeze(0).cpu().numpy()  

                pred_vol[d1:d2, h1:h2, w1:w2] = pred_np
                
            if os.path.exists(mask_path):
                mask_nib = nib.load(mask_path)
                mask_data= mask_nib.get_fdata().astype(np.float32)
                mask_data[mask_data>=0.5] = 1
                mask_data[mask_data<0.5 ] = 0

                if mask_data.shape == pred_vol.shape:
                    met = compute_binary_metrics(pred_vol, mask_data)
                    all_metrics.append(met)
                else:
                    print(f"[Warning] mask.shape={mask_data.shape} != pred.shape={pred_vol.shape}, skip metrics.")

            # 保存
            if save_result_dir is not None:
                base_name = os.path.basename(img_path).replace(".nii.gz","")
                out_path  = os.path.join(save_result_dir, f"{base_name}_vnet_pred.nii.gz")

                pred_nib = nib.Nifti1Image(pred_vol.astype(np.float32), img_nib.affine)
                nib.save(pred_nib, out_path)
                print(f"Saved prediction: {out_path}")

    if len(all_metrics) > 0:
        avg = {}
        keys = all_metrics[0].keys()
        for k in keys:
            avg[k] = np.mean([m[k] for m in all_metrics])

        print("\n====== Final Test Metrics (Average, VNet) ======")
        for k,v in avg.items():
            print(f"{k.capitalize()}: {v:.4f}")
        print("===============================================\n")
    else:
        print("No valid ground-truth labels or shape mismatch. Done inference only.")


if __name__ == "__main__":
    data_dir = "./data"

    test_pairs = gather_image_mask_pairs(data_dir, image_suffix="_crop.nii.gz")
    print("Found test samples:", len(test_pairs))

    # 推理
    inference_3d_vnet(
        model_path="vnet_ventricles_best.pth",
        test_list=test_pairs,
        save_result_dir="./vnet_infer_out",
        device="cuda:0",
        patch_size=(64,64,64)  
    )
