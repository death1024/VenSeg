import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch, embed_dim=512, patch_size=(4,4,4)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  
        B, ED, Dp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1,2)
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
            nn.Dropout(dropout_p)
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
        decoder_channels=(512,256)
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads, mlp_ratio, dropout_p)

        # Decoder (逐级上采样)
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
        # Patch Embedding => tokens
        tokens, (Dp,Hp,Wp) = self.patch_embed(x)  
        # Transformer
        x_trans = self.transformer(tokens)       
        # reshape => (B,ED,Dp,Hp,Wp)
        B, N, ED = x_trans.shape
        x_enc = x_trans.transpose(1,2).view(B, ED, Dp, Hp, Wp)

        # 多级上采样
        dec = x_enc
        for up_block in self.up_blocks:
            dec = up_block(dec)

        out = self.out_conv(dec)
        return out

def compute_binary_metrics(pred, target, smooth=1e-5):
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)

    pred   = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)

    intersection = (pred * target).sum()
    union = (pred.sum() + target.sum())

    dice = (2.*intersection + smooth)/(union + smooth)
    iou  = (intersection + smooth)/(pred.sum() + target.sum() - intersection + smooth)

    TP = (pred * target).sum()
    FP = (pred * (1-target)).sum()
    TN = ((1-pred)*(1-target)).sum()
    FN = ((1-pred)*target).sum()

    TP,FP,TN,FN = TP.float(),FP.float(),TN.float(),FN.float()

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
    pairs = []
    images = sorted(
        f for f in glob.glob(os.path.join(data_dir, f"*{image_suffix}"))
        if "mask" not in os.path.basename(f)
    )
    for img_path in images:
        base = img_path.replace(image_suffix, "")
        corr = base + "_mask_corr.nii.gz"
        crop = base + "_mask_crop.nii.gz"

        if os.path.exists(corr):
            pairs.append((img_path, corr))
        elif os.path.exists(crop):
            pairs.append((img_path, crop))
        else:
            print(f"[Skipping] No mask found for {img_path}")
    return pairs

def generate_sliding_indices(D,H,W, patch_d=64, patch_h=64, patch_w=64):
    coords = []
    d_list = list(range(0,D,patch_d))
    if d_list[-1]+patch_d> D:
        d_list[-1] = max(D-patch_d,0)

    h_list = list(range(0,H,patch_h))
    if h_list[-1]+patch_h> H:
        h_list[-1] = max(H-patch_h,0)

    w_list = list(range(0,W,patch_w))
    if w_list[-1]+patch_w> W:
        w_list[-1] = max(W-patch_w,0)

    for dd in d_list:
        for hh in h_list:
            for ww in w_list:
                coords.append((dd, dd+patch_d, hh, hh+patch_h, ww, ww+patch_w))
    return coords


def inference_3d_transformer(
    model_path,
    test_list,
    save_result_dir=None,
    device="cuda:0",
    patch_size=(64,64,64),
    patch_embed_size=(4,4,4),
    embed_dim=512,
    depth=4,
    num_heads=8,
    mlp_ratio=4.0,
    dropout_p=0.1,
    decoder_channels=(512,256)
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = TransUNet3D(
        in_channels=1,
        out_channels=1,
        patch_size=patch_embed_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout_p=dropout_p,
        decoder_channels=decoder_channels
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if save_result_dir:
        os.makedirs(save_result_dir, exist_ok=True)

    all_metrics = []

    with torch.no_grad():
        for (img_path, mask_path) in test_list:
            # 读取图像
            img_nib = nib.load(img_path)
            img_data= img_nib.get_fdata().astype(np.float32)

            # 归一化
            mn, mx = img_data.min(), img_data.max()
            if (mx - mn)>1e-6:
                img_data= (img_data - mn)/(mx - mn)

            D,H,W = img_data.shape
            pred_vol = np.zeros((D,H,W), dtype=np.float32)

            # 滑窗分块
            coords_list = generate_sliding_indices(D,H,W, patch_size[0], patch_size[1], patch_size[2])
            for (d1,d2,h1,h2,w1,w2) in coords_list:
                patch_img = img_data[d1:d2, h1:h2, w1:w2]
                patch_tensor = torch.from_numpy(patch_img[None,None,...]).float().to(device)

                logits = model(patch_tensor)  
                probs  = torch.sigmoid(logits)
                preds  = (probs>0.5).float()

                pred_np= preds.squeeze(0).squeeze(0).cpu().numpy()
                pred_vol[d1:d2, h1:h2, w1:w2] = pred_np

            if os.path.exists(mask_path):
                mask_nib = nib.load(mask_path)
                mask_data= mask_nib.get_fdata().astype(np.float32)
                mask_data[mask_data>=0.5] = 1
                mask_data[mask_data< 0.5] = 0

                if mask_data.shape == pred_vol.shape:
                    metric_dict = compute_binary_metrics(pred_vol, mask_data)
                    all_metrics.append(metric_dict)
                else:
                    print(f"[Warning] label.shape={mask_data.shape} != pred.shape={pred_vol.shape}, skip metrics...")

            if save_result_dir:
                base_name = os.path.basename(img_path).replace(".nii.gz","")
                out_path = os.path.join(save_result_dir, f"{base_name}_trans_pred.nii.gz")

                out_nib = nib.Nifti1Image(pred_vol.astype(np.float32), img_nib.affine)
                nib.save(out_nib, out_path)
                print(f"Saved prediction: {out_path}")

    if len(all_metrics)>0:
        avg_dict = {}
        keys = all_metrics[0].keys()
        for k in keys:
            avg_dict[k] = np.mean([m[k] for m in all_metrics])
        print("\n==== Final Transformer Test Metrics (Average) ====")
        for k,v in avg_dict.items():
            print(f"{k.capitalize()}: {v:.4f}")
        print("==================================================")
    else:
        print("No valid ground-truth labels or shape mismatch. Done inference.")


if __name__ == "__main__":
    data_dir = "./test_data"
    test_list = gather_image_mask_pairs(data_dir, image_suffix="_crop.nii.gz")
    print("Found test samples:", len(test_list))

    # 推理
    inference_3d_transformer(
        model_path="transformer_ventricles_best.pth",  
        test_list=test_list,
        save_result_dir="./transformer_infer_out",      
        device="cuda:0",
        patch_size=(64,64,64),
        patch_embed_size=(4,4,4),
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout_p=0.1,
        decoder_channels=(512,256)
    )
