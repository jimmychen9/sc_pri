"""Train MaskDecoder on cached OI latents with random SNR channel noise.

Each batch:
  1. Load clean latents + GT masks
  2. Sample random SNR per sample, add AWGN
  3. Feed noisy latents to MaskDecoder
  4. Supervise with BCE + Dice loss against GT masks

Validates at fixed SNR levels [0, 5, 10, 15, 20] dB.

Usage:
    python scripts/05_train_mask_decoder.py
"""

import os
import time

import torch
from torch.utils.data import DataLoader

from sc_pri.utils import load_config, bce_dice_loss, iou
from sc_pri.channel import Channel
from sc_pri.models.mask_decoder import MaskDecoder
from sc_pri.data.latent_dataset import LatentMaskDataset


# ---------- Config ---------- # 配置参数
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 1e-3
SNR_RANGE = (0.0, 20.0)
EVAL_SNRS = [0, 5, 10, 15, 20]
SAVE_DIR = "checkpoints/mask_decoder_oi"
DEVICE = "cuda"


def train_one_epoch(model, loader, optimizer, channel, device): # 定义单个训练 epoch
    model.train() # 切换到训练模式
    total_loss = 0 # 初始化累计训练损失
    n_batches = 0  # 初始化 batch 计数器

    for latent, mask in loader: # 开始遍历当前序列
        latent = latent.to(device)  # (B, 4, 64, 64) # 读取 latent，并增加 batch 维度后移动到设备
        mask = mask.to(device)      # (B, C, 64, 64) # 读取对应的多类别 mask

        # Add random SNR channel noise  # 向 latent 添加随机 SNR 的 AWGN 信道噪声
        noisy_latent, snr_used = channel(latent)

        # Forward
        pred_logits = model(noisy_latent)  # (B, C, 64, 64) # 使用 MaskDecoder 输出每个类别的 logits
        loss = bce_dice_loss(pred_logits, mask) # 计算 BCE 与 Dice 的组合损失

        optimizer.zero_grad()  # 清空上一轮梯度
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 根据梯度更新模型参数

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches # 返回函数结果


@torch.no_grad() # 关闭梯度计算，减少显存占用并加速推理
def validate(model, loader, channel, device, eval_snrs): # 定义多 SNR 条件下的验证流程
    model.eval() # 切换到评估模式
    results = {} # 初始化验证结果字典

    for snr_db in eval_snrs: # 开始遍历当前序列
        total_iou = None # 初始化累计 IoU
        n_batches = 0 # 初始化 batch 计数器

        for latent, mask in loader: # 开始遍历当前序列
            latent = latent.to(device)  读取 latent，并增加 batch 维度后移动到设备
            mask = mask.to(device) # 读取对应的多类别 mask

            noisy_latent, _ = channel(latent, snr_db=float(snr_db))
            pred_logits = model(noisy_latent) # 使用 MaskDecoder 输出每个类别的 logits
            pred_binary = (torch.sigmoid(pred_logits) > 0.5).float() # 将预测概率阈值化为二值 mask

            batch_iou = iou(pred_binary, (mask > 0.5).float())  # (C,) # 计算当前 batch 的逐类别 IoU
            if total_iou is None: # 判断当前条件是否成立
                total_iou = batch_iou  # 初始化累计 IoU
            else: # 处理其他情况
                total_iou = total_iou + batch_iou # 初始化累计 IoU
            n_batches += 1

        mean_iou = total_iou / n_batches  # (C,) # 计算当前 SNR 下的平均 IoU
        results[snr_db] = mean_iou.cpu()

    return results


def main():
    cfg = load_config("configs/data.yaml")
    num_classes = len(cfg["classes"])
    class_names = [c["name"] for c in cfg["classes"]]

    cache_root = cfg["cache"]["root"]
    train_dir = os.path.join(cache_root, cfg["cache"]["train_subdir"])
    val_dir = os.path.join(cache_root, cfg["cache"]["val_subdir"])

    # Datasets # 数据集与 DataLoader
    train_ds = LatentMaskDataset(train_dir) # 创建训练数据集
    val_ds = LatentMaskDataset(val_dir) # 创建验证数据集

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = MaskDecoder(in_channels=4, out_channels=num_classes, base=64).to(DEVICE) # 创建 MaskDecoder 并移动到设备
    total_params = sum(p.numel() for p in model.parameters()) / 1e6 # 统计模型参数量，单位为百万
    print(f"MaskDecoder: {total_params:.1f}M params, {num_classes} classes")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) # 创建 AdamW 优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)  创建余弦退火学习率调度器

    channel = Channel(snr_db_range=SNR_RANGE) # 创建 AWGN 信道对象

    os.makedirs(SAVE_DIR, exist_ok=True) # 创建目录；若目录已存在则不报错
    best_mean_iou = 0.0 # 初始化最佳平均 IoU

    for epoch in range(1, NUM_EPOCHS + 1): # 开始遍历当前序列
        t0 = time.time() # 记录当前 epoch 开始时间

        train_loss = train_one_epoch(model, train_loader, optimizer, channel, DEVICE)  执行一个训练 epoch 并返回平均损失
        val_results = validate(model, val_loader, channel, DEVICE, EVAL_SNRS)  在多个固定 SNR 下验证模型
        scheduler.step() # 更新学习率

        # Print results
        elapsed = time.time() - t0 # 计算当前 epoch 耗时
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} ({elapsed:.1f}s)  loss={train_loss:.4f}") # 输出运行信息
        print(f"  {'SNR':>5s}", end="")
        for name in class_names: # 开始遍历当前序列
            print(f"  {name:>15s}", end="")
        print(f"  {'mean':>8s}")

        epoch_ious = [] # 保存当前 epoch 在不同 SNR 下的平均 IoU
        for snr_db in EVAL_SNRS:  # 开始遍历当前序列
            iou_per_class = val_results[snr_db] # 读取当前 SNR 对应的逐类别 IoU
            print(f"  {snr_db:>3d}dB", end="")
            for c in range(num_classes): # 开始遍历当前序列
                print(f"  {iou_per_class[c].item():>15.4f}", end="")
            mean = iou_per_class.mean().item() # 计算所有类别的平均 IoU
            print(f"  {mean:>8.4f}")
            epoch_ious.append(mean)

        overall_mean = sum(epoch_ious) / len(epoch_ious) # 计算多个 SNR 条件下的整体平均 IoU
        print(f"  Overall mean IoU across SNRs: {overall_mean:.4f}")
 
        # Save best # 保存当前最佳模型
        if overall_mean > best_mean_iou: # 判断当前条件是否成立
            best_mean_iou = overall_mean # 初始化最佳平均 IoU
            save_path = os.path.join(SAVE_DIR, "best.pt") # 生成模型检查点保存路径
            torch.save({ # 保存训练检查点
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mean_iou": best_mean_iou,
            }, save_path)
            print(f"  -> New best! Saved to {save_path}")

        # Save periodic checkpoint # 定期保存训练检查点
        if epoch % 20 == 0:  # 判断当前条件是否成立
            save_path = os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)

    print(f"\nTraining complete. Best mean IoU: {best_mean_iou:.4f}")


if __name__ == "__main__":
    main()
