# Swin Transformer based on Pytorch

具体学习地址见 [Transformer学习总述](../README.md)

## ViT模型简介

- **模型创新点**：
  
  1. 图像分辨率高，SwinT对token下采样分组(window)，可降低ViT的高复杂度计算
  2. 分组使得感受野有限，无法使用全局信息，加入SW-MSA融合交互信息
  
- **模型重点**：

  - Patch Merging层：下采样
  - Windows Multi-head Self-Attention（W-MSA）模块：减少计算复杂度
  - Shifted Windows Multi-Head Self-Attention（SW-MSA）模块：增加不同windows间的信息交互
  - Relative Position Bias模块

- **模型结构图**：

  <img src="Structure-image\SwinT Structure.png" alt="SwinT Structure" style="zoom:50%;" />

- **模型不同变体配置参数**：

  <img src="Structure-image\SwinT Structure para.png" alt="SwinT Structure para" style="zoom:50%;" />

  

## 实例

- 数据集
- 配置环境