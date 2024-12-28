# train.py

import os
import mindspore
from mindspore import nn, context, Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import dtype as mstype
import numpy as np

# 导入您的模型和数据加载器
from model import Unet
from dataloader import create_unet_dataset

def main():
    # 配置参数
    image_dir = "CamVid/train"         # 替换为您的图像文件夹路径
    mask_dir = "CamVid/trainannot"     # 替换为您的掩码文件夹路径
    batch_size = 4
    epochs = 100
    learning_rate = 1e-4
    checkpoint_dir = "./checkpoints"  # 模型检查点保存目录
    device_target = "CPU"             # 选择设备目标："GPU", "Ascend", "CPU"

    # 设置上下文
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

    # 创建训练数据集
    train_dataset = create_unet_dataset(image_dir, mask_dir, batch_size=batch_size, shuffle=True)

    # 定义模型
    model = Unet()

    # 定义损失函数
    loss_fn = nn.BCEWithLogitsLoss()

    # 定义优化器
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=learning_rate)

    # 定义指标
    metrics = {"accuracy": nn.Accuracy()}

    # 实例化模型
    model = Model(model, loss_fn, optimizer, metrics=metrics)

    # 定义回调函数
    # 创建检查点目录
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 配置检查点
    config_ck = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="unet", directory=checkpoint_dir, config=config_ck)

    # 其他回调
    callbacks = [
        LossMonitor(),
        TimeMonitor(),
        ckpoint_cb
    ]

    # 开始训练
    model.train(epoch=epochs, train_dataset=train_dataset, callbacks=callbacks, dataset_sink_mode=True)

if __name__ == '__main__':
    main()