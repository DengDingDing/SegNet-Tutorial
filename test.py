import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
from mindspore.train import Model
from mindspore.dataset import GeneratorDataset
from dataloader import create_unet_dataset  # 假设 dataloader.py 中的函数
from model import Unet
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 配置测试集路径
test_image_dir = "CamVid/test"              # 测试图像文件夹路径
test_mask_dir = "CamVid/testannot"          # 测试掩码文件夹路径
batch_size = 16
num_classes = 11
checkpoint_path = "./checkpoints/best_model.ckpt"  # 加载训练好的模型检查点

# 设置上下文
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 创建测试数据集
test_dataset = create_unet_dataset(test_image_dir, test_mask_dir, batch_size=batch_size, shuffle=False)
test_loader = GeneratorDataset(test_dataset, ["data", "label"])

# 定义模型
model = Unet(num_classes=num_classes)

# 加载模型参数
param_dict = ms.load_checkpoint(checkpoint_path)
ms.load_param_into_net(model, param_dict)

# 定义评估模型
model = Model(model)

# 测试并收集结果
y_true = []
y_pred = []

print("开始测试...")
for data in test_loader.create_dict_iterator():
    inputs = data["data"]
    labels = data["label"].asnumpy()  # 将标签转为 numpy 格式
    outputs = model.predict(Tensor(inputs)).asnumpy()  # 获取预测结果
    preds = np.argmax(outputs, axis=1)  # 获取每个样本的预测类别

    y_true.extend(labels.flatten())
    y_pred.extend(preds.flatten())

# 计算混淆矩阵和分类报告
conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(num_classes)])

# 打印结果
print("测试结果分析：")
print("混淆矩阵：")
print(conf_matrix)
print("\n分类报告：")
print(report)

# 保存混淆矩阵为图片
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, [f"Class {i}" for i in range(num_classes)], rotation=45)
plt.yticks(tick_marks, [f"Class {i}" for i in range(num_classes)])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png")
plt.show()