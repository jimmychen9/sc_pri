"""Stage 4: Verify latent-space DDIM inpainting. # Stage 4：验证 latent 空间中的 DDIM inpainting 是否正确工作。

For a few cached samples:
  1. Load latent z_0 and mask
  2. Run DDIM repaint inpainting (mask region → generated, outside → original)
  3. Save 4-panel comparison:
     original | decoded_original | mask_overlay | inpainted

Usage:
    python scripts/04_verify_inpaint.py
"""

import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sc_pri.utils import load_config
from sc_pri.vae import load_vae, decode
from sc_pri.diffusion import load_sd_components, ddim_inpaint


def run_verification(cache_dir, out_dir, vae, sd_components, cfg, # 定义 DDIM inpainting 验证流程
                     n_samples=10, num_steps=50, strength=1.0,
                     prompt="a blurred face", guidance_scale=7.5):
    os.makedirs(out_dir, exist_ok=True) # 创建目录；若目录已存在则不报错
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[:n_samples] # 查找并排序缓存的 .pt 文件


    if not files: # 判断当前条件是否成立
        print(f"No .pt files found in {cache_dir}")
        return # 提前结束当前函数

    device = sd_components["device"] # 读取模型运行设备
    scale_factor = cfg["latent"]["scale_factor"] # 读取 latent 缩放系数

    for i, f in enumerate(files):  # 开始遍历当前序列
        data = torch.load(f, map_location="cpu") # 从磁盘加载缓存样本
        latent = data["latent"].float().unsqueeze(0).to(device)  # (1,4,64,64) # 读取 latent，并增加 batch 维度后移动到设备
        mask = data["mask"].float()  # (C, 64, 64) # 读取对应的多类别 mask

        # Combine all class masks into one binary mask (1 = replace)
        # Any class with mask > 0.3 counts as "sensitive"
        combined_mask = (mask.sum(dim=0, keepdim=True) > 0.3).float()  # (1, 64, 64) # 合并多类别 mask 得到单个二值 mask
        combined_mask = combined_mask.unsqueeze(0).to(device)  # (1, 1, 64, 64) # 合并多类别 mask 得到单个二值 mask

        # Skip if no mask content
        if combined_mask.sum() < 1: # 判断当前条件是否成立
            print(f"  [{i}] No mask content, skipping")
            continue # 跳过当前样本，继续下一次循环

        print(f"  [{i}] Mask coverage: {combined_mask.mean().item()*100:.1f}%, running inpaint...")

        # Run DDIM inpainting # 运行 DDIM latent 修复
        z_inpainted = ddim_inpaint( # 调用 DDIM inpainting 得到修复后的 latent
            z_0=latent,
            mask_64=combined_mask,
            sd_components=sd_components,
            num_inference_steps=num_steps, # 设置 DDIM 去噪步数
            strength=strength,
            prompt=prompt,
            guidance_scale=guidance_scale, # 设置 classifier-free guidance 强度
            seed=42 + i,
        )

        # Decode both original and inpainted # 分别解码原始 latent 和修复后的 latent
        img_orig = decode(vae, latent, scale_factor=scale_factor) # 处理原始 latent 解码图像
        img_orig = ((img_orig.clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0) # 处理原始 latent 解码图像
        img_orig = img_orig.cpu().numpy().astype(np.uint8) # 处理原始 latent 解码图像

        img_inpaint = decode(vae, z_inpainted, scale_factor=scale_factor) # 处理修复后 latent 解码图像
        img_inpaint = ((img_inpaint.clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0)
        img_inpaint = img_inpaint.cpu().numpy().astype(np.uint8)

        H = img_orig.shape[0] # 设置全局配置参数

        # Mask overlay on original
        mask_up = F.interpolate( # 将低分辨率 mask 上采样到图像尺寸
            combined_mask, size=H, mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        overlay = img_orig.copy().astype(np.float32)  # 创建用于 mask 叠加显示的图像副本
        overlay[..., 1] = np.clip(overlay[..., 1] + mask_up * 180, 0, 255)
        overlay = overlay.astype(np.uint8) # 创建用于 mask 叠加显示的图像副本


        # Original photo (if available) # 尝试读取原始图片
        try:
            orig_photo = Image.open(data["filepath"]).convert("RGB") # 读取并缩放原始图片
            orig_photo = np.array(orig_photo.resize((H, H))) # 读取并缩放原始图片
        except Exception: # 捕获异常，避免脚本中断
            orig_photo = np.zeros_like(img_orig) # 读取并缩放原始图片

        # 4-panel: photo | decoded_original | mask_overlay | inpainted # 生成四栏对比图
        combined = np.concatenate([orig_photo, img_orig, overlay, img_inpaint], axis=1)  # 横向拼接多张对比图
        out_path = os.path.join(out_dir, f"inpaint_{i:03d}.png") # 生成结果图像保存路径
        Image.fromarray(combined).save(out_path) # 将 NumPy 图像转换为 PIL 图像并保存
        print(f"  [{i}] Saved {out_path}") # 输出运行信息

    print(f"\nWrote visualizations to {out_dir}")
    print("Panels (left→right): photo | decoded | mask | inpainted")
    print("Check:")
    print("  - Mask region should have different/generated content")
    print("  - Non-mask region should be identical to decoded original")


def main(): # 定义脚本主函数
    cfg = load_config("configs/data.yaml")
    device = cfg["vae"]["device"]

    # Load VAE
    print("Loading VAE...") # 加载 VAE 编码器和解码器
    vae = load_vae(cfg["vae"]["model_id"], device=device) # 加载预训练 VAE

    # Load SD components (UNet, scheduler, text encoder)
    print("Loading SD 1.5 components...")
    sd_components = load_sd_components(device=device)

    cache_dir = os.path.join(cfg["cache"]["root"], cfg["cache"]["train_subdir"])
    out_dir = "debug/vis_inpaint" # 设置可视化结果输出目录

    run_verification(
        cache_dir, out_dir, vae, sd_components, cfg,
        n_samples=10,
        num_steps=50,
        strength=1.0,
        prompt="a blurred face",
        guidance_scale=7.5,
    )


if __name__ == "__main__":
    main()
