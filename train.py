# train.py

import os
import mindspore
from mindspore import nn, context, Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import dtype as mstype
import numpy as np
import mindspore.ops as ops

# 导入您的模型和数据加载器
from model import Unet
from dataloader import create_unet_dataset

def main():
    # 配置参数
    train_image_dir = "CamVid/train"         # 替换为您的训练图像文件夹路径
    train_mask_dir = "CamVid/trainannot"     # 替换为您的训练掩码文件夹路径
    val_image_dir = "CamVid/val"             # 替换为您的验证图像文件夹路径
    val_mask_dir = "CamVid/valannot"         # 替换为您的验证掩码文件夹路径
    batch_size = 16
    epochs = 50
    learning_rate = 1e-4
    checkpoint_dir = "./checkpoints"         # 模型检查点保存目录
    device_target = "CPU"                    # 选择设备目标："GPU", "Ascend", "CPU"

    # 设置上下文
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

    # 创建训练和验证数据集
    train_dataset = create_unet_dataset(train_image_dir, train_mask_dir, batch_size=batch_size, shuffle=True)
    val_dataset = create_unet_dataset(val_image_dir, val_mask_dir, batch_size=batch_size, shuffle=False)

    # 定义模型
    model = Unet()

    # 定义损失函数
    loss_fn = nn.MSELoss()

    # 定义优化器
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=learning_rate)

    # 定义自定义准确率指标
    class BinaryAccuracy(nn.Metric):
        def __init__(self):
            super(BinaryAccuracy, self).__init__()
            self.correct = 0
            self.total = 0

        def clear(self):
            self.correct = 0
            self.total = 0

        def update(self, *inputs):
            preds, labels = inputs
            preds = ops.Round()(preds)
            self.correct += (preds == labels).sum().asnumpy()
            self.total += labels.size

        def eval(self):
            return self.correct / self.total

    metrics = {"accuracy": BinaryAccuracy()}

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
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        # 训练
        model.train(epoch=1, train_dataset=train_dataset, callbacks=callbacks, dataset_sink_mode=True)
        
        # 验证
        print("Starting validation...")
        metrics = model.eval(val_dataset, dataset_sink_mode=True)
        val_loss = metrics.get("loss")
        val_accuracy = metrics.get("accuracy")
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

if __name__ == '__main__':
    main()