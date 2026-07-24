"""Stage 6: End-to-end pipeline verification. # Stage 6：验证从信道噪声、MaskDecoder 到 DDIM 修复的完整端到端流程。

Full pipeline:
  z_0 → AWGN(SNR) → z_noisy → MaskDecoder → predicted mask → DDIM inpaint → output

Compares GT mask vs predicted mask inpainting at multiple SNRs.

Usage:
    python scripts/06_e2e_verify.py
"""

import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sc_pri.utils import load_config
from sc_pri.vae import load_vae, decode
from sc_pri.channel import Channel
from sc_pri.models.mask_decoder import MaskDecoder
from sc_pri.diffusion import load_sd_components, ddim_inpaint


def decode_to_numpy(vae, latent, scale_factor): # 定义 latent 解码并转换为 NumPy 图像的函数
    img = decode(vae, latent, scale_factor=scale_factor) # 解码 latent 并转换图像数值范围
    img = ((img.clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0) # 解码 latent 并转换图像数值范围
    return img.cpu().numpy().astype(np.uint8) # 返回函数结果


def mask_to_overlay(img, mask_64, H): # 定义将 mask 叠加到图像上的函数
    """Green overlay from (1,1,64,64) mask onto (H,W,3) image."""
    mask_up = F.interpolate(mask_64, size=H, mode="bilinear", align_corners=False) # 将低分辨率 mask 上采样到图像尺寸
    mask_up = mask_up.squeeze().cpu().numpy() # 将低分辨率 mask 上采样到图像尺寸
    overlay = img.copy().astype(np.float32) # 创建用于 mask 叠加显示的图像副本
    overlay[..., 1] = np.clip(overlay[..., 1] + mask_up * 180, 0, 255)
    return overlay.astype(np.uint8)


def run_e2e(cache_dir, out_dir, vae, sd_components, model, cfg, # 定义端到端验证流程
            snr_list=(20, 10, 5, 0), n_samples=5, num_steps=50,
            prompt="a blurred face", guidance_scale=7.5):
    os.makedirs(out_dir, exist_ok=True) # 创建目录；若目录已存在则不报错
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[:n_samples] # 查找并排序缓存的 .pt 文件

    if not files: # 判断当前条件是否成立
        print(f"No .pt files found in {cache_dir}")
        return # 提前结束当前函数

    device = sd_components["device"] # 读取模型运行设备
    scale_factor = cfg["latent"]["scale_factor"] # 读取 latent 缩放系数
    channel = Channel() # 创建 AWGN 信道对象
    model.eval() # 切换到评估模式

    for i, f in enumerate(files): # 开始遍历当前序列
        data = torch.load(f, map_location="cpu") # 从磁盘加载缓存样本
        latent = data["latent"].float().unsqueeze(0).to(device) # 读取 latent，并增加 batch 维度后移动到设备
        gt_mask = data["mask"].float() # 读取真实多类别 mask

        # GT combined mask # 把多个类别的真实 mask 合并成一个二值 mask
        gt_combined = (gt_mask.sum(dim=0, keepdim=True) > 0.3).float() # 合并真实 mask 得到单通道二值 mask
        gt_combined = gt_combined.unsqueeze(0).to(device)  # (1,1,64,64) # 合并真实 mask 得到单通道二值 mask

        if gt_combined.sum() < 1: # 判断当前条件是否成立
            continue # 跳过当前样本，继续下一次循环

        # Clean decoded # 解码干净 latent 得到参考图像
        img_clean = decode_to_numpy(vae, latent, scale_factor)
        H = img_clean.shape[0] # 设置全局配置参数

        for snr_db in snr_list: # 开始遍历当前序列
            print(f"  [{i}] SNR={snr_db} dB")

            # Channel noise  # 通过 AWGN 信道添加噪声
            z_noisy, _ = channel(latent, snr_db=float(snr_db))
            img_noisy = decode_to_numpy(vae, z_noisy, scale_factor) # 解码带噪 latent

            # MaskDecoder prediction  # 使用 MaskDecoder 预测敏感区域 mask
            with torch.no_grad():
                pred_logits = model(z_noisy) # 使用 MaskDecoder 输出每个类别的 logits
                pred_mask = (torch.sigmoid(pred_logits) > 0.5).float() # 将 MaskDecoder 输出阈值化为多类别二值 mask
                pred_combined = (pred_mask.sum(dim=1, keepdim=True) > 0).float() # 合并预测的多类别 mask

            # Inpaint with GT mask # 使用真实 mask 执行修复
            z_inpaint_gt = ddim_inpaint( # 使用真实 mask 执行 DDIM 修复
                z_0=z_noisy, mask_64=gt_combined,
                sd_components=sd_components,
                num_inference_steps=num_steps, strength=1.0,
                prompt=prompt, guidance_scale=guidance_scale,
                seed=42 + i,
            )
            img_inpaint_gt = decode_to_numpy(vae, z_inpaint_gt, scale_factor) # 解码使用真实 mask 修复后的 latent

            # Inpaint with predicted mask # 使用模型预测的 mask 执行修复
            z_inpaint_pred = ddim_inpaint( # 使用预测 mask 执行 DDIM 修复
                z_0=z_noisy, mask_64=pred_combined,
                sd_components=sd_components,
                num_inference_steps=num_steps, strength=1.0,
                prompt=prompt, guidance_scale=guidance_scale,
                seed=42 + i,
            )
            img_inpaint_pred = decode_to_numpy(vae, z_inpaint_pred, scale_factor) # 解码使用预测 mask 修复后的 latent

            # Overlays # 生成 mask 叠加可视化
            gt_overlay = mask_to_overlay(img_noisy, gt_combined, H) # 生成真实 mask 的叠加图
            pred_overlay = mask_to_overlay(img_noisy, pred_combined, H) # 生成预测 mask 的叠加图

            # 6-panel: clean | noisy | gt_mask | gt_inpaint | pred_mask | pred_inpaint # 生成六栏端到端对比图
            panels = np.concatenate([ # 拼接端到端对比图
                img_clean, img_noisy,
                gt_overlay, img_inpaint_gt,
                pred_overlay, img_inpaint_pred,
            ], axis=1)

            out_path = os.path.join(out_dir, f"e2e_{i:03d}_snr{snr_db}.png") # 生成结果图像保存路径
            Image.fromarray(panels).save(out_path) # 将 NumPy 图像转换为 PIL 图像并保存

    print(f"\nPanels: clean | noisy | GT_mask | GT_inpaint | pred_mask | pred_inpaint")
    print(f"Wrote to {out_dir}")


def main():
    cfg = load_config("configs/data.yaml")
    device = cfg["vae"]["device"]
    num_classes = len(cfg["classes"])

    print("Loading VAE...")
    vae = load_vae(cfg["vae"]["model_id"], device=device)

    print("Loading SD components...")
    sd_components = load_sd_components(device=device)

    print("Loading MaskDecoder...")
    model = MaskDecoder(in_channels=4, out_channels=num_classes, base=64).to(device)
    ckpt_path = "checkpoints/mask_decoder_oi/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt['epoch']}, "
          f"IoU {ckpt['best_mean_iou']:.4f})")

    cache_dir = os.path.join(cfg["cache"]["root"], cfg["cache"]["train_subdir"])
    out_dir = "debug/vis_e2e"

    run_e2e(
        cache_dir, out_dir, vae, sd_components, model, cfg,
        snr_list=[20, 10, 5, 0],
        n_samples=5,
        num_steps=50,
    )


if __name__ == "__main__":
    main()
