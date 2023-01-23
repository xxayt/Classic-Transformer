# Vision Transformer based on Pytorch

具体学习地址见 [Transformer学习总述](../README.md)

## ViT模型简介

- **模型创新点**：使用纯粹的Transformer用于图片token序列，也可以很好的完成图像分类任务
  
- **模型重点**：
  
  - Self-Attention机制
  - Embedding层
  - Transformer Encoder层
  - MLP Head层
  
- **模型结构图**：

  <img src="Structure-image\ViT Structure.png" alt="ViT Structure" style="zoom:50%;" />

- **模型不同变体配置参数**：

  <img src="Structure-image\ViT Structure para.png" alt="ViT Structure" style="zoom:50%;" />

- **Vision Transformer Base (ViT-B/16)详细结构举例**：来自于[Vision Transformer详解](https://blog.csdn.net/qq_37541097/article/details/118242600)

  <img src="Structure-image\ViT-B-16.png" alt="ViT Structure" style="zoom:50%;" />

## 实例

- 数据集
- 配置环境