"""Stage 2: Cache latents + multi-class masks.""" # Stage 2：将图像编码为 latent，并缓存对应的多类别掩码。
import os
from pathlib import Path

from sc_pri.utils import load_config
from sc_pri.vae import load_vae
from sc_pri.data.oi_cache import process_split


def main(): # 定义脚本主函数
    cfg = load_config("configs/data.yaml") # 读取 YAML 配置文件
    
    
    # Set fiftyone zoo dir (must match download step) # 设置 FiftyOne 数据集缓存目录
    zoo_dir = os.path.expanduser(cfg["download"]["fiftyone_zoo_dir"]) # 读取并展开 FiftyOne 数据目录路径
    os.environ["FIFTYONE_DATASET_ZOO_DIR"] = zoo_dir # 设置环境变量
    
    class_to_idx = {c["name"]: c["idx"] for c in cfg["classes"]} # 建立类别名称到类别索引的映射
    print(f"class_to_idx = {class_to_idx}")
    
    device = cfg["vae"]["device"] # 读取模型运行设备
    vae = load_vae(cfg["vae"]["model_id"], device=device) # 加载预训练 VAE
    print(f"VAE loaded on {device}")
    
    cache_root = cfg["cache"]["root"] # 读取缓存根目录
    
    process_split( # 处理指定数据划分并缓存 latent 与 mask
        "oi_face_plate_train",
        os.path.join(cache_root, cfg["cache"]["train_subdir"]),
        vae, cfg, class_to_idx,
    )
    process_split( # 处理指定数据划分并缓存 latent 与 mask
        "oi_face_plate_val",
        os.path.join(cache_root, cfg["cache"]["val_subdir"]),
        vae, cfg, class_to_idx,
    )
    
    print("\nCache complete.")


if __name__ == "__main__":
    main()
