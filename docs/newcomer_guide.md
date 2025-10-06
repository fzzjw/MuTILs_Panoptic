# MuTILs 新手指南

本指南介绍了 MuTILs 全景分割（panoptic segmentation）代码库，重点介绍了应首先探索的关键模块，并概述了如何准备与预期的训练和推理格式相匹配的自定义数据。

## 代码库布局

- **顶级使用文档** – `README.md` 文件详细介绍了已发布的模型、Docker 工作流程、预期的主机/容器挂载点以及默认的推理命令。这些是您在阅读代码之前快速启动环境的最快方式。
- **`configs/`** – 集中管理配置文件。除了运行时所需的 YAML 文件 (`MuTILsWSIRunConfigs.yaml`)，这里还有一些 Python 辅助文件，用于枚举区域和细胞核的分类体系 (`panoptic_model_configs.py`)、将原始标签名称链接到标准化的 MuTILs 编码，并提供默认的可视化设置。
- **`mutils_panoptic/`** – 包含核心的模型、训练和推理代码。建议从 `MuTILs.py` (模型定义与评估辅助工具)、`MuTILsTrainer.py` (协调交叉验证、损失计算和日志记录) 以及 `MuTILsWSIRunner.py` (全切片推理流程) 开始阅读。
- **`utils/`** – 包含在训练/推理中通用的共享工具：GPU 分配、torchvision 风格的图像变换、可视化辅助工具以及 numpy/pandas 工具。
- **`tests/`** – 提供了一个自动化的冒烟测试 (`tests/test_inference.py`)，用于验证 Docker 化的推理过程和输出目录的完整性。

## 关键运行时组件

- **MuTILsWSIRunner** (`mutils_panoptic/MuTILsWSIRunner.py`) 将全切片图像（WSI）处理分为专门的预处理、推理和后处理工作单元。该运行器加载一组模型检查点，提取排名最高的感兴趣区域（ROI），将它们分派到不同的 GPU 上进行处理，并在保存掩码（mask）、注释和特征表之前，整合切片级别的各项指标。
- **MuTILs 模型栈** (`mutils_panoptic/MuTILs.py`) 定义了一个多分辨率的 UNet 主干网络，以及用于区域分割、细胞核分类和计算性肿瘤浸润淋巴细胞评估 (CTA) 评分的多个头部网络。该文件还包含了 `MuTILsTransform` (对输入进行批处理/归一化) 和 `MuTILsEvaluator` (聚合 ROI 和高倍镜视野 (HPF) 的指标、CTA 的分子/分母项，以及可接受的错误分类)。
- **训练入口** (`mutils_panoptic/MuTILsTrainer.py`) 管理交叉验证折叠采样、数据加载器、类别平衡、优化器/调度器设置、周期（epoch）循环、可视化检查点以及用于后续分析的评估结果导出。

## 训练数据格式要求

`RegionDatasetLoaders.MuTILsDataset` 是训练期间使用的标准数据集。

- **目录结构** – 该加载器期望根目录下有 `root/tcga` 和 `root/acs` 两个子文件夹，每个子文件夹中都包含平行的 `rgbs/` 和 `masks/` 目录，其中 ROI 切片以 PNG 文件形式保存。文件名遵循 `SLIDENAME_*.png` 的模式，以便数据集能将 ROI 映射回其所属的切片。
- **掩码编码** – 每个掩码都是一个三通道的 PNG 图像。通道 0 存储区域的超类编码，通道 1 存储细胞核类别，通道 2 存储轮廓像素。在加载过程中，位于噪声区域 (`OTHER`, `WHITE`) 的细胞核会被清零，边界像素会被重新分配给细胞核背景类，以在数据增强前强制实现分离。
- **切片缩放** – ROI 会被调整到所要求的高倍镜视野（HPF）放大倍数，在训练期间会应用随机裁剪/尺度抖动，并且高分辨率 (HPF) 和低分辨率 (ROI 尺度) 的 RGB + 掩码张量副本都会被返回给模型。
- **类别平衡** – 当 `training=True` 时，如果 `region_summary.csv` 和 `nuclei_summary.csv` 缓存文件不存在，加载器会创建它们，然后计算 ROI 的样本权重，以同时平衡不同切片和罕见组织区域的出现频率。这些权重会被送入 `MuTILsTrainer` 中的 `WeightedRandomSampler`（加权随机采样器）。
- **训练/测试集划分** – `get_cv_fold_slides` 函数会从 `train_test_splits/` 目录中读取名为 `fold_{k}_train.csv` 和 `fold_{k}_test.csv` 的 CSV 文件。您需要为您的数据集提供这些文件，以控制交叉验证的成员划分。

## 标签分类体系与约束

MuTILs 的标签空间在 `configs/panoptic_model_configs.py` 中定义：

- `RegionCellCombination.REGION_CODES` 列出了用于 ROI 级别分割的区域超类（如 `TUMOR`, `STROMA`, `TILS`, `NORMAL`, `OTHER`, `WHITE`, `BLOOD` 等）。
- `RegionCellCombination.NUCLEUS_CODES` 枚举了九个细胞核类别（包含明确的 `BACKGROUND` 和 `EXCLUDE` 编码），并将外部的 NuCLS 标签映射到这个标准化的集合中。
- `RegionCellCombination.nuclei_regions_codes` 限制了每个区域超类内部允许出现的细胞核类别（例如，上皮细胞核必须出现在 `TUMOR` 中，基质细胞必须出现在 `STROMA`/`TILS` 中，正常上皮细胞必须出现在 `NORMAL` 中）。加载器在构建训练掩码和评估过程中会强制执行这些约束。
- 组合后的“区域+细胞核”掩码会重用这些编码，以便可视化和全景指标在所有输出中保持一致。

在整理您自己的数据时，请将区域和细胞核的注释与这些编码对齐。如果您必须引入新的类别，请相应地扩展字典，并更新掩码、损失函数和可视化颜色图。

## 上手实践与使用自定义数据的步骤

1. **复现推理过程** – 按照 `README.md` 中的 Docker 工作流程，在示例切片上运行 `MuTILsWSIRunner`。确认输出的结构与 `tests/test_inference.py` 所验证的结构相匹配。
2. **检查数据集** – 研究 `RegionDatasetLoaders.MuTILsDataset` 以理解 ROI 采样、数据增强和加权机制。可以构建一个原型 notebook 来打印 ROI 掩码，以确认您的注释符合区域/细胞核的约束。
3. **启动划分与缓存** – 生成训练/测试集的 CSV 文件，将 ROI 组织到 `tcga/` 和 `acs/` 目录下，并在通过 `MuTILsTrainer.py` 启动多 GPU 训练之前，让加载器创建摘要 CSV 文件。
4. **监控训练过程** – 利用每个周期输出的损失曲线、评估导出结果和 CTA 指标，来诊断类别不平衡或掩码对齐问题。
5. **迭代调整标签对齐** – 使用 `panoptic_model_configs.py` 中的分类体系定义和强制映射，来审查您的标签是如何转换为 MuTILs 编码的，并根据需要调整预处理流程或配置文件。

通过这些步骤，新的贡献者可以从熟悉代码库布局开始，逐步过渡到在自定义数据集上运行实验，同时保持与 MuTILs 全景分割技术栈的兼容性。
