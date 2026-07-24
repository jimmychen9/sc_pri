"""Stage 1: Download OI-V7 subset for face + plate (bbox only).""" # Stage 1：下载 Open Images V7 中与人脸和车牌相关的检测数据子集。
import os

import fiftyone as fo
import fiftyone.zoo as foz

from sc_pri.utils import load_config # 从 sc_pri.utils 导入所需对象



def download_split(split, max_samples, dataset_name, classes, seed): # 定义下载指定数据划分的函数
    if fo.dataset_exists(dataset_name): # 判断当前条件是否成立
        print(f"[{split}] existing dataset '{dataset_name}' found, deleting...") # 输出运行信息
        fo.delete_dataset(dataset_name) # 删除已有的同名 FiftyOne 数据集

    ds = foz.load_zoo_dataset( # 从 FiftyOne Zoo 下载或加载 Open Images 数据集
        "open-images-v7", # 指定 Open Images V7 数据集
        split=split, # 指定训练或验证数据划分
        label_types=["detections"], # 只下载目标检测标注
        classes=classes, # 从配置中提取类别名称
        max_samples=max_samples, # 限制最多下载的样本数量
        seed=seed,
        shuffle=True, # 下载前随机打乱样本
        dataset_name=dataset_name, # 设置 FiftyOne 数据集名称
        only_matching=True, # 只保留真正包含目标类别的样本
    )
    ds.persistent = True # 将 FiftyOne 数据集设为持久化保存
    return ds


def report(ds, name): # 定义数据集统计信息输出函数
    print(f"\n=== {name} ({len(ds)} samples) ===")
    print("Fields:", list(ds.get_field_schema().keys()))

    try: # 尝试执行可能失败的操作
        counts = ds.count_values("detections.detections.label")
        print("Detection counts:")
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]): # 开始遍历当前序列
            print(f"  {cls}: {n}")
    except Exception as e: # 捕获异常，避免脚本中断
        print(f"(detection counts error: {e})")


def main():  # 定义脚本主函数
    cfg = load_config("configs/data.yaml")  # 读取 YAML 配置文件

    zoo_dir = os.path.expanduser(cfg["download"]["fiftyone_zoo_dir"]) # 读取并展开 FiftyOne 数据目录路径
    os.environ["FIFTYONE_DATASET_ZOO_DIR"] = zoo_dir # 设置环境变量
    os.makedirs(zoo_dir, exist_ok=True) # 创建目录；若目录已存在则不报错
    print(f"fiftyone zoo dir: {zoo_dir}")

    classes = [c["name"] for c in cfg["classes"]] # 从配置中提取类别名称
    seed = cfg["download"]["seed"] # 读取随机种子

    train = download_split( # 下载或加载训练集
        "train", cfg["download"]["n_train"],
        "oi_face_plate_train", classes, seed,
    )
    val = download_split( # 下载或加载验证集
        "validation", cfg["download"]["n_val"],
        "oi_face_plate_val", classes, seed,
    )

    report(train, "TRAIN") # 输出数据集统计信息
    report(val, "VAL") # 输出数据集统计信息

    print("\nDone. Reload later via:")
    print('  fo.load_dataset("oi_face_plate_train")')


if __name__ == "__main__":
    main()
