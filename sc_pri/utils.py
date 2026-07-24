"""Dependency-free utilities. No imports from other sc_pri modules."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def letterbox_image(img_np, new_size=512, pad_value=114): # 定义保持宽高比并补边为正方形的函数
    """Resize keeping aspect ratio, pad to square.
    
    Args:
        img_np: HxWx3 uint8 numpy array
        new_size: target square size
        pad_value: pixel value for padding
    
    Returns:
        (padded_img, scale, pad_x, pad_y)
    """
    h, w = img_np.shape[:2] # 读取原始图像的高度 h 和宽度 w
    scale = new_size / max(h, w) # 让原图较长的一边缩放到 new_size，并保持宽高比
    nh, nw = int(round(h * scale)), int(round(w * scale)) # 计算缩放后的图像高度宽度
    resized = np.array(Image.fromarray(img_np).resize((nw, nh), Image.BILINEAR)) # 将 NumPy 图像转换为 PIL 图像 # PIL 的尺寸顺序是（宽度，高度）
    canvas = np.full((new_size, new_size, 3), pad_value, dtype=np.uint8) # 创建目标尺寸的三通道正方形画布# 使用指定像素值填充整个画布# 图像像素使用 uint8 类型 
    pad_y = (new_size - nh) // 2 # 计算顶部需要填充的像素数
    pad_x = (new_size - nw) // 2  # 计算左侧需要填充的像素数
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized # 将缩放后的图像放到正方形画布中央
    return canvas, scale, pad_x, pad_y


def letterbox_binary_mask(mask_np, new_size=512): # 定义二值 mask 的 letterbox 处理函数
    """Letterbox a HxW binary mask using nearest interpolation with zero padding."""
    h, w = mask_np.shape[:2] # 读取原始掩码的高度和宽度
    scale = new_size / max(h, w) # 计算保持宽高比的缩放比例
    nh, nw = int(round(h * scale)), int(round(w * scale))  # 计算缩放后的掩码高度宽度
    resized = np.array(
        Image.fromarray((mask_np.astype(np.uint8) * 255)) # 将布尔或数值掩码转换为 uint8 类型# 将掩码从 0/1 转换为 PIL 常用的 0/255 格式 
             .resize((nw, nh), Image.NEAREST)  # 使用最近邻插值，避免 mask 出现中间值
    ) > 127 # 使用 127 阈值重新得到布尔二值 mask
    canvas = np.zeros((new_size, new_size), dtype=np.uint8) # 创建全 0 的正方形 mask 画布
    pad_y = (new_size - nh) // 2 # 计算顶部填充高度
    pad_x = (new_size - nw) // 2 # 计算左侧填充宽度
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized.astype(np.uint8) # 将缩放后的 mask 放入画布中央
    return canvas # 返回补边后的二值 mask


def transform_bbox_letterbox(x, y, w, h, orig_w, orig_h, new_size=512): # 将原图相对 bbox 转换到 letterbox 图像坐标
    """Convert relative bbox [0,1] to pixel coords in letterboxed image.
    
    Args:
        x, y, w, h: relative bbox (fiftyone / OI format)
        orig_w, orig_h: original image size
        new_size: letterboxed size
    
    Returns:
        (x1, y1, x2, y2) as clipped ints
    """
    scale = new_size / max(orig_w, orig_h) # 计算原图的 letterbox 缩放比例
    pad_x = (new_size - int(round(orig_w * scale))) // 2 # 计算左侧填充宽度
    pad_y = (new_size - int(round(orig_h * scale))) // 2 # 计算顶部填充高度
    x1 = int(round(x * orig_w * scale + pad_x)) # 计算 bbox 左上角 x 像素坐标
    y1 = int(round(y * orig_h * scale + pad_y)) # 计算 bbox 左上角 y 像素坐标
    x2 = int(round((x + w) * orig_w * scale + pad_x)) # 计算 bbox 右下角 x 像素坐标
    y2 = int(round((y + h) * orig_h * scale + pad_y)) # 计算 bbox 右下角 y 像素坐标
    return (max(0, x1), max(0, y1), min(new_size, x2), min(new_size, y2)) # 将坐标限制在图像范围内并返回


# ---------- Loss / metrics ----------

def dice_loss(pred, target, eps=1e-6):
    """Dice loss for binary segmentation. Both pred/target are (B, C, H, W) in [0,1]."""
    pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)  # 将预测的 H、W 展平成一个像素维度
    target = target.contiguous().view(target.shape[0], target.shape[1], -1) # 将标签的 H、W 展平成一个像素维度
    num = 2 * (pred * target).sum(-1) + eps # 计算 Dice 分子：2 倍交集加 eps
    den = pred.sum(-1) + target.sum(-1) + eps # 计算 Dice 分母：预测面积加标签面积
    return 1 - (num / den).mean() # Dice loss 等于 1 减去平均 Dice 分数


def bce_dice_loss(pred_logits, target, dice_weight=1.0): BCE 主要关心每个像素分类是否正确  # 定义 BCE 与 Dice 的组合损失
    """BCE + Dice loss combo for multi-channel mask prediction."""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target) # 直接用 logits 计算 BCE loss
    dice = dice_loss(torch.sigmoid(pred_logits), target) # 先做 sigmoid，再计算 Dice loss
    return bce + dice_weight * dice # 返回 BCE 与加权 Dice loss 的和


def iou(pred_binary, target_binary, eps=1e-6): 定义 Intersection over Union。输入必须已经是 binary mask。
    """Per-class IoU. Inputs are (B, C, H, W) binary. Returns (C,) tensor. 把 batch 中所有样本合并统计 分别计算每个 channel 的 IoU"""
    dims = (0, 2, 3) # 在 batch、高度和宽度维度上求和，保留类别维度
    inter = (pred_binary * target_binary).sum(dims) # 计算每个类别的交集
    union = pred_binary.sum(dims) + target_binary.sum(dims) - inter # 计算每个类别的并集n 
    return (inter + eps) / (union + eps) # 返回每个类别的 IoU，并用 eps 防止除零


# ---------- Config loading ----------

def load_config(path): 定义配置读取函数
    import yaml 在函数内部导入 PyYAML
    with open(path) as f:
        return yaml.safe_load(f)
