"""Stage 3: Visualize cached latents + masks. CRITICAL sanity check.""" # Stage 3：可视化缓存的 latent 和 mask，用于检查数据处理是否正确。
import os

from sc_pri.utils import load_config # 从 sc_pri.utils 导入所需对象
from sc_pri.vae import load_vae # 从 sc_pri.vae 导入所需对象
from sc_pri.viz.sanity import visualize_cached_samples # 从 sc_pri.viz.sanity 导入所需对象


def main(): # 定义脚本主函数
    cfg = load_config("configs/data.yaml") # 读取 YAML 配置文件
    
    vae = load_vae(cfg["vae"]["model_id"], device=cfg["vae"]["device"]) # 加载预训练 VAE
    
    cache_train = os.path.join(cfg["cache"]["root"], cfg["cache"]["train_subdir"]) # 拼接训练集缓存目录
    out_dir = "debug/vis_train" # 设置可视化结果输出目录
    
    visualize_cached_samples(  # 可视化缓存样本以进行人工检查
        cache_train, out_dir, vae,
        n_samples=20, # 选择 20 个样本进行可视化
        scale_factor=cfg["latent"]["scale_factor"], # 读取 latent 缩放系数
    )


if __name__ == "__main__":
    main()
