"""SD-VAE encoder/decoder wrapper."""
import torch
from diffusers import AutoencoderKL


def load_vae(model_id="stabilityai/sd-vae-ft-mse", device="cuda"): # 定义 VAE 加载函数
    vae = AutoencoderKL.from_pretrained(model_id).to(device).eval() # 加载预训练 VAE，移动到设备并设为推理模式
    for p in vae.parameters(): # 遍历 VAE 所有参数
        p.requires_grad_(False) # 冻结参数，不参与梯度更新
    return vae # 返回加载完成的 VAE


@torch.no_grad() # 编码时关闭梯度计算
def encode(vae, img_tensor, scale_factor=0.18215): # 定义图像到 latent 的编码函数
    """Encode HWC uint8 numpy (or preprocessed tensor) to latent.
    
    Args:
        vae: loaded AutoencoderKL
        img_tensor: either (H, W, 3) uint8 numpy, or (B, 3, H, W) float tensor in [-1, 1]
    
    Returns:
        latent (B, 4, H/8, W/8) on VAE device, fp32
    """
    if hasattr(img_tensor, "dtype") and img_tensor.dtype == torch.uint8: # 如果输入是 uint8 PyTorch tensor
        # Assume HWC
        x = img_tensor.float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0 # 转为 BCHW 并归一化到 [-1,1]
    elif img_tensor.ndim == 3:  # numpy HWC # 如果输入是三维 NumPy HWC 图像
        import numpy as np
        assert isinstance(img_tensor, np.ndarray) # 确认输入确实是 NumPy 数组
        x = torch.from_numpy(img_tensor).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0 # 转为 BCHW tensor 并归一化
    else: # 如果输入已经是预处理好的 tensor
        x = img_tensor  # already (B, 3, H, W) float in [-1, 1] # 直接使用输入
    
    device = next(vae.parameters()).device # 获取 VAE 当前所在设备
    x = x.to(device) # 将输入移动到 VAE 设备
    z = vae.encode(x).latent_dist.mean * scale_factor # 使用后验均值作为 latent，并乘缩放系数
    return z # 返回编码后的 latent


@torch.no_grad() # 解码时关闭梯度计算
def decode(vae, latent, scale_factor=0.18215):  # 定义 latent 到图像的解码函数
    """Decode latent to image tensor in [-1, 1].""" # 将 latent 解码为 [-1,1] 范围图像张量
    device = next(vae.parameters()).device # 获取 VAE 当前设备
    latent = latent.to(device) / scale_factor # 将 latent 移到设备并取消缩放
    img = vae.decode(latent).sample # 使用 VAE decoder 生成图像
    return img # 返回解码后的图像张量
