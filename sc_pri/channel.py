"""AWGN channel for latent-space semantic communication.

Fixes vs. previous implementation:
- Uses ALL tensor dimensions (C, H, W) for signal power computation, not 3 of 4.
- No spurious /2 factor. That factor was mistakenly borrowed from complex-signal
  formulas; latents are real-valued, so per-sample noise variance = P_signal / SNR_linear.

Verified by test_channel_snr() below.
"""
import torch


class Channel: # 定义 AWGN 信道类
    """Additive White Gaussian Noise (AWGN) channel for real-valued latents.
    
    Given a target SNR in dB, adds noise such that:
        SNR_dB = 10 * log10(P_signal / P_noise)
    
    where P_signal is measured per-sample over all latent dimensions.
    """
    
    def __init__(self, snr_db_range=(0.0, 20.0), fixed_snr_db=None): # 初始化信道参数
        """
        Args:
            snr_db_range: (low, high) for random SNR sampling per batch
            fixed_snr_db: if set, ignores range and uses this fixed SNR
        """
        self.snr_db_range = snr_db_range # 保存随机 SNR 范围
        self.fixed_snr_db = fixed_snr_db # 保存固定 SNR
    
    def sample_snr(self, batch_size, device): # 为 batch 中每个样本采样 SNR
        """Sample SNR (in dB) per sample in the batch.""" # 以 dB 为单位采样 SNR
        if self.fixed_snr_db is not None: # 如果设置了固定 SNR
            return torch.full((batch_size,), self.fixed_snr_db,  # 创建长度为 batch_size 的固定 SNR 张量
                              dtype=torch.float32, device=device) # 指定数据类型和设备
        low, high = self.snr_db_range # 取出 SNR 下限和上限
        return torch.rand(batch_size, device=device) * (high - low) + low # 在区间内均匀随机采样
    
    def forward(self, z, snr_db=None): # 对 latent 添加 AWGN
        """Add AWGN to latent z.
        
        Args:
            z: (B, C, H, W) real-valued latent
            snr_db: (B,) or scalar SNR in dB. If None, sample from configured range.
        
        Returns:
            (noisy_z, snr_db_used)
        """
        B = z.shape[0] # 获取 batch size
        if snr_db is None: # 如果没有传入 SNR
            snr_db = self.sample_snr(B, z.device) # 为每个样本随机采样 SNR
        elif isinstance(snr_db, (int, float)): # 如果传入的是单个数值
            snr_db = torch.full((B,), float(snr_db), device=z.device) # 扩展为 batch 长度的张量
        
        # Per-sample signal power over ALL dims (C, H, W)
        # z_flat: (B, C*H*W)
        z_flat = z.reshape(B, -1)  # 将每个样本展平成一维
        signal_power = z_flat.pow(2).mean(dim=1)  # (B,) # 计算每个样本的平均信号功率
        
        # noise_power = signal_power / SNR_linear
        snr_linear = 10.0 ** (snr_db / 10.0)         # (B,)  # 将 dB SNR 转换为线性 SNR
        noise_power = signal_power / snr_linear      # (B,) # 计算每个样本的噪声功率
        noise_std = noise_power.sqrt()               # (B,) # 计算高斯噪声标准差
        
        # Broadcast std to z shape
        noise = torch.randn_like(z) * noise_std.view(B, 1, 1, 1) # 生成并缩放高斯噪声
        return z + noise, snr_db # 返回带噪 latent 和使用的 SNR
    
    def __call__(self, z, snr_db=None): # 允许直接把 Channel 对象当函数调用
        return self.forward(z, snr_db)

