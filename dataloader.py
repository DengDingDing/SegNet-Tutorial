import os
import numpy as np
from PIL import Image
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter  # 导入 Inter 枚举

def get_image_mask_pairs(image_dir, mask_dir):
    """
    获取图像和对应掩码的路径列表

    Args:
        image_dir (str): 图像文件夹路径
        mask_dir (str): 掩码文件夹路径

    Returns:
        list of tuples: [(image_path1, mask_path1), (image_path2, mask_path2), ...]
    """
    image_files = sorted([
        os.path.join(image_dir, file) 
        for file in os.listdir(image_dir) 
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    mask_files = sorted([
        os.path.join(mask_dir, file) 
        for file in os.listdir(mask_dir) 
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    assert len(image_files) == len(mask_files), "图像和掩码的数量不匹配"
    return list(zip(image_files, mask_files))

def generator(image_mask_pairs):
    """
    生成器函数，逐对返回图像和掩码

    Args:
        image_mask_pairs (list of tuples): 图像和掩码路径列表

    Yields:
        tuple: (PIL.Image.Image, PIL.Image.Image)
    """
    for img_path, mask_path in image_mask_pairs:
        try:
            img = Image.open(img_path)  # 假设已是3通道
            mask = Image.open(mask_path)  # 假设已是3通道
            yield (img, mask)
        except Exception as e:
            print(f"Error loading {img_path} or {mask_path}: {e}")
            continue

def mask_to_label(mask):
    """
    将掩码张量转换为标签张量

    Args:
        mask (numpy.ndarray): 掩码Tensor，形状为 (3, H, W)

    Returns:
        numpy.ndarray: 标签Tensor，形状为 (1, H, W)，dtype为 float32
    """
    # 选择第一个通道
    mask = mask[0, :, :]  # 形状: (H, W)
    # 转换为 float32 类型
    mask = mask.astype(np.float32)
    # 如果标签从1开始，转换为从0开始
    mask = mask - 1
    # 扩展维度为 (1, H, W)
    mask = np.expand_dims(mask, axis=0)
    return mask

def create_unet_dataset(image_dir, mask_dir, batch_size=4, shuffle=True):
    image_mask_pairs = get_image_mask_pairs(image_dir, mask_dir)

    # 创建GeneratorDataset
    dataset = ds.GeneratorDataset(
        source=generator(image_mask_pairs),
        column_names=["image", "mask"],
        shuffle=shuffle,
        num_parallel_workers=2,  # 减少并行工作线程数
        python_multiprocessing=False  # 关闭多进程以简化调试
    )

    # 定义图像的转换操作，仅应用ToTensor
    image_transforms = [
        vision.ToTensor()
    ]

    # 定义掩码的转换操作，仅应用ToTensor
    mask_transforms = [
        vision.ToTensor()
    ]

    # 将转换应用到图像列
    dataset = dataset.map(
        operations=image_transforms,
        input_columns="image",
        output_columns="image",
        num_parallel_workers=2
    )

    # 将转换应用到掩码列
    dataset = dataset.map(
        operations=mask_transforms,
        input_columns="mask",
        output_columns="mask",
        num_parallel_workers=2
    )

    # 将掩码转换为整数类标签
    dataset = dataset.map(
        operations=mask_to_label,
        input_columns="mask",
        output_columns="mask",
        num_parallel_workers=2
    )

    # 验证标签的数据类型和形状
    for data in dataset.create_dict_iterator():
        masks = data["mask"]
        print("Mask dtype:", masks.dtype)  # 应为 float32
        print("Mask shape:", masks.shape)  # 应为 (batch_size, 1, H, W)
        break

    # 批处理
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

if __name__ == "__main__":
    # 替换为实际的图像和掩码文件夹路径
    image_dir = "CamVid/train"  # 图像文件夹路径
    mask_dir = "CamVid/trainannot"    # 掩码文件夹路径
    batch_size = 4

    # 创建数据集
    dataset = create_unet_dataset(image_dir, mask_dir, batch_size=batch_size, shuffle=True)

    # 迭代数据集并查看批次形状
    for data in dataset.create_dict_iterator():
        images = data["image"]  # Tensor shape: (batch_size, 3, 360, 480)
        masks = data["mask"]    # Tensor shape: (batch_size, 360, 480)
        print("Batch of images shape:", images.shape)
        print("Batch of masks shape:", masks.shape)
        # 在这里可以进行训练或其他操作
        break  # 仅演示一次迭代