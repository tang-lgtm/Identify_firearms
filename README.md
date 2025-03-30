# GitHub 项目文档：枪支图像分类系统

## 📌 项目概述
这是一个基于深度学习（ResNet50）的枪支图像二分类系统，能够识别图像中是否包含枪支。系统采用迁移学习和动态类别权重调整技术，有效处理类别不平衡问题。

## 🛠️ 技术栈
- **框架**: TensorFlow 2.x / Keras
- **核心模型**: ResNet50 (预训练+微调)
- **图像处理**: PIL, OpenCV
- **辅助工具**: scikit-learn (评估指标), matplotlib (可视化)

## 📂 文件结构
```
/gun-classification
│── gun_classifier.py       # 主程序代码
│── gun_classify_model.keras # 训练好的模型文件
│── data/                   # 数据集目录（示例结构）
│   ├── train/              # 训练集
│   │   ├── gun/            # 包含枪支的图像
│   │   └── no_gun/         # 不包含枪支的图像
│   ├── val/                # 验证集
│   └── test/               # 测试集
│── README.md               # 项目说明文档

```

## 🔧 安装与运行

### 1. 环境配置
pychram

### 2. 数据准备
按照以下结构组织数据：
```
data/
  ├── train/
  │   ├── gun/        # 存放训练集枪支图像
  │   └── no_gun/     # 存放训练集非枪支图像
  ├── val/            # 验证集（同train结构）
  └── test/           # 测试集（同train结构）
```

### 3. 运行分类系统
```bash
# 训练模型（会自动跳过无效图像）
python gun_classifier.py --mode train

# 评估现有模型
python gun_classifier.py --mode evaluate
```

## 🧠 算法架构

### 系统流程图
```mermaid
graph LR
    A[输入图像] --> B[图像预处理]
    B --> C[ResNet50特征提取]
    C --> D[全局平均池化]
    D --> E[全连接层]
    E --> F[Sigmoid分类]
```

### 关键技术
1. **动态类别权重调整**：
   - 每4个epoch自动评估各类别准确率
   - 根据表现动态调整损失函数权重

2. **数据增强**：
   - 180度随机旋转
   - 亮度/对比度调整
   - 水平/垂直翻转
   - 随机缩放/剪切

3. **两阶段训练**：
   - 第一阶段：冻结ResNet50基础层
   - 第二阶段：微调最后5层

## 💡 核心功能说明

### 主要组件
| 组件 | 功能 |
|------|------|
| `CustomIterator` | 处理无效图像的自定义数据迭代器 |
| `DynamicClassWeightCallback` | 动态调整类别权重的回调函数 |
| `find_invalid_images()` | 检测并排除损坏的图像文件 |
| `find_best_threshold()` | 寻找最佳分类阈值 |

### 模型参数
- **基础模型**: ResNet50 (ImageNet预训练)
- **分类头**: 
  - GlobalAveragePooling2D
  - Dense(1024, ReLU)
  - Dropout(0.5)
  - Dense(1, Sigmoid)
- **优化器**: Adam (初始lr=1e-3，微调lr=1e-5)

## 📊 性能评估
系统提供完整的评估报告：
```text
Classification Report:
              precision  recall  f1-score  support

         gun       0.92    0.89    0.90       150
      no_gun       0.90    0.93    0.91       150

    accuracy                           0.91       300
   macro avg       0.91    0.91    0.91       300
weighted avg       0.91    0.91    0.91       300
```

## 🚀 后续改进方向
1. 增加多角度检测能力
2. 集成YOLO实现目标检测
3. 开发实时视频流分析功能

## 📜 开源协议
Apache License 2.0

---

> 提示：实际使用时请修改数据路径参数（`E:/机器学习训练数据集`）为您的实际数据路径。系统会自动跳过损坏图像并记录日志。
