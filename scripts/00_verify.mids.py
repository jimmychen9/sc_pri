"""Stage 0: Verify class MIDs and whether they have segmentation masks.""" # Stage 0：检查 Open Images 类别 MID，并确认相关类别是否提供分割掩码。
import pandas as pd

CLASS_DESC_URL = ( # 设置全局配置参数
    "https://storage.googleapis.com/openimages/v7/"
    "oidv7-class-descriptions-boxable.csv"
)
SEG_CLASSES_URL = ( # 设置全局配置参数
    "https://storage.googleapis.com/openimages/v7/"
    "oidv7-classes-segmentation.txt"
)


def main(): # 定义脚本主函数
    class_df = pd.read_csv(CLASS_DESC_URL, header=None, # 读取 Open Images 类别描述表
                            names=["MID", "DisplayName"])
    print(f"Total boxable classes: {len(class_df)}") # 输出运行信息
    
    targets = class_df[  # 筛选名称中包含 face 或 plate 的类别
        class_df["DisplayName"].str.contains("face|plate", 
                                             case=False, na=False)
    ]
    print("\n=== Face/plate related classes ===") # 输出运行信息
    print(targets.to_string(index=False)) # 输出运行信息
    
    seg_classes = pd.read_csv(SEG_CLASSES_URL, header=None, names=["MID"]) # 读取支持分割标注的类别 MID
    seg_mids = set(seg_classes["MID"].values) # 将分割类别 MID 转换为集合，便于快速查询
    print(f"\nTotal segmentation classes: {len(seg_mids)}") # 输出运行信息
    
    print("\n=== Mask availability for face/plate ===")
    for _, row in targets.iterrows(): # 开始遍历当前序列
        has_mask = row["MID"] in seg_mids # 判断当前类别是否拥有 segmentation mask
        tag = "HAS MASK" if has_mask else "BBOX only" # 根据是否有 mask 生成显示标签
        print(f"  {row['DisplayName']:35s} ({row['MID']:12s}): {tag}") # 输出运行信息


if __name__ == "__main__":
    main()
