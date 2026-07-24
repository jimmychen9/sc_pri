"""SD 1.5 UNet + DDIM scheduler for latent-space repaint inpainting.

Pipeline:
    1. Forward diffusion: add noise to z_0 up to timestep T
    2. Reverse DDIM: denoise step by step, each step replacing
       mask-outside region with the original noised latent at that timestep.
    Result: mask-inside = generated content, mask-outside = original image.
"""

import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


def load_sd_components(model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", # 定义加载 SD 组件的函数
                       device="cuda"):
    """Load UNet, scheduler, and text encoder from SD 1.5.
    
    Returns:
        dict with keys: unet, scheduler, text_encoder, tokenizer
    """
    print(f"Loading UNet from {model_id}...") # 打印 UNet 加载信息
    unet = UNet2DConditionModel.from_pretrained( # 从预训练模型加载 UNet
        model_id, subfolder="unet" # 指定模型 ID 和 UNet 子目录
    ).to(device).eval() # 移到设备并切换到推理模式
    for p in unet.parameters(): # 遍历 UNet 参数
        p.requires_grad_(False) # 冻结 UNet 参数，不参与训练
    
    print("Loading DDIM scheduler...") # 打印 scheduler 加载信息
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler") # 加载 DDIM scheduler
    
    print("Loading CLIP text encoder...") # 打印 CLIP 加载信息
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer") # 加载 CLIP tokenizer
    text_encoder = CLIPTextModel.from_pretrained( # 加载 CLIP 文本编码器
        model_id, subfolder="text_encoder" # 指定文本编码器子目录
    ).to(device).eval() # 移到设备并切换到推理模式
    for p in text_encoder.parameters(): # 遍历文本编码器参数
        p.requires_grad_(False) # 冻结文本编码器参数
    
    return { # 返回所有 SD 组件
        "unet": unet, 
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "device": device,
    }


def get_text_embedding(tokenizer, text_encoder, prompt, device): # 将文本 prompt 编码为 CLIP embedding
    """Encode a text prompt to CLIP embedding."""
    tokens = tokenizer( # 对 prompt 进行分词
        prompt, padding="max_length", # 补齐到最大长度
        max_length=tokenizer.model_max_length, # 使用 tokenizer 支持的最大长度
        truncation=True, return_tensors="pt" # 超长时截断并返回 PyTorch 张量
    ).input_ids.to(device)
    with torch.no_grad(): # 关闭梯度计算
        emb = text_encoder(tokens).last_hidden_state  # (1, 77, 768) # 获取 CLIP 最后一层隐藏表示
    return emb


@torch.no_grad() # 整个函数不计算梯度
def ddim_inpaint(z_0, mask_64, sd_components, # 定义 latent-space DDIM inpainting
                 num_inference_steps=50,  # 反向去噪步数
                 strength=1.0, # 扩散强度
                 prompt="", # 文本提示词
                 guidance_scale=7.5, # CFG 引导强度
                 seed=42):
    """DDIM repaint-style inpainting in latent space.
    
    Args:
        z_0: (1, 4, 64, 64) original clean latent (already scaled by 0.18215)
        mask_64: (1, 1, 64, 64) float mask, 1 = region to replace, 0 = keep
        sd_components: dict from load_sd_components()
        num_inference_steps: DDIM steps
        strength: how much of the diffusion process to run (1.0 = full)
        prompt: text prompt for generation (empty = unconditional-ish)
        guidance_scale: classifier-free guidance scale
        seed: random seed
    
    Returns:
        z_inpainted: (1, 4, 64, 64) inpainted latent (scaled)
    """
    unet = sd_components["unet"] # 取出 UNet
    scheduler = sd_components["scheduler"]
    tokenizer = sd_components["tokenizer"]
    text_encoder = sd_components["text_encoder"]
    device = sd_components["device"]
    
    # Unscale for diffusion (SD UNet expects unscaled latents)
    scale_factor = 0.18215 # Stable Diffusion 常用 latent 缩放系数
    z_0_unscaled = z_0 / scale_factor # 将输入 latent 反缩放
    
    # Set up scheduler
    scheduler.set_timesteps(num_inference_steps, device=device) # 创建推理时间步序列
    
    # Determine start timestep based on strength
    init_timestep = int(num_inference_steps * strength) # 计算参与扩散的步数
    t_start_idx = max(num_inference_steps - init_timestep, 0) # 计算开始时间步索引
    timesteps = scheduler.timesteps[t_start_idx:] # 截取实际使用的时间步
    
    # Get text embeddings (conditional + unconditional for CFG)
    cond_emb = get_text_embedding(tokenizer, text_encoder, prompt, device)  # 编码用户 prompt
    uncond_emb = get_text_embedding(tokenizer, text_encoder, "", device) # 编码空 prompt
    text_emb = torch.cat([uncond_emb, cond_emb])  # (2, 77, 768) # 拼接 CFG 所需 embedding
    
    # Forward diffusion: add noise to z_0 up to first timestep
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z_0_unscaled.shape, generator=generator, # 生成与 latent 同形状的高斯噪声
                        device=device, dtype=z_0_unscaled.dtype)
    
    # Noised original at the starting timestep
    z_t = scheduler.add_noise(z_0_unscaled, noise, timesteps[:1]) # 按起始时间步向 z_0 添加噪声
    
    # Reverse DDIM loop # 开始反向 DDIM 去噪循环
    for i, t in enumerate(timesteps): # 依次遍历每个时间步
        # Classifier-free guidance: predict noise for uncond and cond # 分别预测无条件和条件噪声
        z_t_input = torch.cat([z_t, z_t])  # (2, 4, 64, 64)  # 复制 latent 供 CFG 两个分支使用
        t_input = torch.cat([t.unsqueeze(0)] * 2) # 将当前时间步复制两份
        
        noise_pred = unet(z_t_input, t_input, encoder_hidden_states=text_emb).sample  # 用 UNet 预测噪声
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2) # 拆分无条件和条件噪声预测
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond) # 应用 CFG
        
        # DDIM step # 执行一次 DDIM 反向更新
        z_t = scheduler.step(noise_pred, t, z_t).prev_sample # 得到前一个时间步的 latent
        
        # Repaint: replace mask-outside with original noised latent at this timestep # 用原图对应噪声替换 mask 外区域
        if i < len(timesteps) - 1: # 如果还没到最后一步
            next_t = timesteps[i + 1] # 取得下一个时间步
            z_orig_noised = scheduler.add_noise(z_0_unscaled, noise, next_t.unsqueeze(0)) # 计算原图在下一时间步的带噪 latent
        else: # 如果已经到最后一步
            # Last step: use clean original # 最后一步直接使用干净原始 latent
            z_orig_noised = z_0_unscaled # mask 外恢复成原始 latent
        
        z_t = mask_64 * z_t + (1 - mask_64) * z_orig_noised # mask 内保留生成结果，mask 外使用原图
    
    
    # Re-scale for VAE decoding # 重新缩放以便 VAE 解码
    z_inpainted = z_t * scale_factor # 恢复 Stable Diffusion latent 缩放
    return z_inpainted # 返回修复后的 latent
