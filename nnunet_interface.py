import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_instancenorm_lrelu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_ch),
        nn.LeakyReLU(negative_slope=0.01, inplace=True)
    )

class DoubleConv(nn.Module):
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
        self.down1= nn.Conv3d(base_ch, base_ch*2, kernel_size=2, stride=2)

        self.enc2 = DoubleConv(base_ch*2, base_ch*2)
        self.down2= nn.Conv3d(base_ch*2, base_ch*4, kernel_size=2, stride=2)

        self.enc3 = DoubleConv(base_ch*4, base_ch*4)
        self.down3= nn.Conv3d(base_ch*4, base_ch*8, kernel_size=2, stride=2)

        self.enc4 = DoubleConv(base_ch*8, base_ch*8)
        self.down4= nn.Conv3d(base_ch*8, base_ch*16, kernel_size=2, stride=2)

        # bottleneck
        self.enc5 = DoubleConv(base_ch*16, base_ch*16)

        # Decoder
        self.up4 = nn.ConvTranspose3d(base_ch*16, base_ch*8, kernel_size=2, stride=2)
        self.dec4= DoubleConv(base_ch*16, base_ch*8)

        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3= DoubleConv(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2= DoubleConv(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1= DoubleConv(base_ch*2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, out_channels, kernel_size=1)

        if self.deep_supervision:
            self.ds4 = nn.Conv3d(base_ch*8, out_channels, kernel_size=1)
            self.ds3 = nn.Conv3d(base_ch*4, out_channels, kernel_size=1)
            self.ds2 = nn.Conv3d(base_ch*2, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)          
        p1 = self.down1(e1)       

        e2 = self.enc2(p1)     
        p2 = self.down2(e2)       

        e3 = self.enc3(p2)       
        p3 = self.down3(e3)     

        e4 = self.enc4(p3)     
        p4 = self.down4(e4)     

        b  = self.enc5(p4)       

        up4 = self.up4(b)        
        cat4= torch.cat([up4, e4], 1)  
        d4 = self.dec4(cat4)      

        up3 = self.up3(d4)      
        cat3= torch.cat([up3, e3], 1)
        d3 = self.dec3(cat3)      

        up2 = self.up2(d3)        
        cat2= torch.cat([up2, e2], 1)
        d2 = self.dec2(cat2)     

        up1 = self.up1(d2)       
        cat1= torch.cat([up1, e1], 1)
        d1 = self.dec1(cat1)    

        out_main = self.out_conv(d1)  

        if self.deep_supervision:
            ds4_out = self.ds4(d4)    
            ds3_out = self.ds3(d3)   
            ds2_out = self.ds2(d2)   
            return out_main, ds4_out, ds3_out, ds2_out
        else:
            return out_main

def compute_binary_metrics(pred, target, smooth=1e-5):
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)

    pred   = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    iou  = (intersection + smooth) / (pred.sum() + target.sum() - intersection + smooth)

    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    TN = ((1 - pred) * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

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
    import glob
    images = sorted(
        f for f in glob.glob(os.path.join(data_dir, f"*{image_suffix}"))
        if "mask" not in os.path.basename(f)
    )
    pairs = []
    for img_path in images:
        base = img_path.replace(image_suffix, "")
        corr = base + "_mask_corr.nii.gz"
        crop = base + "_mask_crop.nii.gz"
        if os.path.exists(corr):
            pairs.append((img_path, corr))
        elif os.path.exists(crop):
            pairs.append((img_path, crop))
        else:
            print(f"[Warning] No mask found for {img_path}. Skip or handle differently.")
    return pairs

def generate_sliding_indices(D,H,W, patch_d=64, patch_h=64, patch_w=64):
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


def inference_3d_nnunet(
    model_path,
    test_list,
    save_result_dir=None,
    device="cuda:0",
    patch_size=(64,64,64),
    deep_supervision=True
):

    # 加载
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = nnUNet3D(in_channels=1, out_channels=1, base_ch=32, deep_supervision=deep_supervision).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if save_result_dir is not None:
        os.makedirs(save_result_dir, exist_ok=True)

    all_metrics = []

    with torch.no_grad():
        for (img_path, mask_path) in test_list:
            # 读取图像
            img_nib = nib.load(img_path)
            img_data= img_nib.get_fdata().astype(np.float32)

            # 归一化
            mn, mx = img_data.min(), img_data.max()
            if (mx - mn) > 1e-6:
                img_data = (img_data - mn)/(mx - mn)

            D,H,W = img_data.shape
            pred_vol = np.zeros((D,H,W), dtype=np.float32)

            # 逐块推理
            coords_list = generate_sliding_indices(
                D,H,W,
                patch_d=patch_size[0],
                patch_h=patch_size[1],
                patch_w=patch_size[2]
            )
            for (d1,d2,h1,h2,w1,w2) in coords_list:
                patch_img = img_data[d1:d2, h1:h2, w1:w2]
                patch_tensor = torch.from_numpy(patch_img[None,None,...]).float().to(device)

                net_out = model(patch_tensor)
                if deep_supervision:
                    main_out = net_out[0] 
                else:
                    main_out = net_out  

                probs = torch.sigmoid(main_out)
                preds = (probs>0.5).float()

                pred_np= preds.squeeze(0).squeeze(0).cpu().numpy()
                pred_vol[d1:d2, h1:h2, w1:w2] = pred_np

            if os.path.exists(mask_path):
                mask_nib = nib.load(mask_path)
                mask_data= mask_nib.get_fdata().astype(np.float32)
                mask_data[mask_data>=0.5] = 1
                mask_data[mask_data< 0.5] = 0

                if mask_data.shape == pred_vol.shape:
                    metrics = compute_binary_metrics(pred_vol, mask_data)
                    all_metrics.append(metrics)
                else:
                    print(f"[Warning] shape mismatch: mask={mask_data.shape}, pred={pred_vol.shape}")

            if save_result_dir:
                base_name = os.path.basename(img_path).replace(".nii.gz","")
                out_path = os.path.join(save_result_dir, f"{base_name}_nnunet_pred.nii.gz")
                nib.save(nib.Nifti1Image(pred_vol, affine=img_nib.affine), out_path)
                print(f"Saved prediction to: {out_path}")

    if len(all_metrics) > 0:
        avg = {}
        keys = all_metrics[0].keys()
        for k in keys:
            avg[k] = np.mean([m[k] for m in all_metrics])

        print("======= Final nnUNet Test Metrics (Average) =======")
        for k in keys:
            print(f"{k.capitalize()}: {avg[k]:.4f}")
        print("===================================================")
    else:
        print("No valid ground-truth labels or shape mismatch. Inference done.")


if __name__ == "__main__":
    data_dir = "./data"

    test_pairs = gather_image_mask_pairs(data_dir, image_suffix="_crop.nii.gz")
    print("Found test samples:", len(test_pairs))

    inference_3d_nnunet(
        model_path="nnunet_ventricles_best.pth",
        test_list=test_pairs,
        save_result_dir="./nnunet_infer_out",
        device="cuda:0",            
        patch_size=(64,64,64),      
        deep_supervision=True    
    )
