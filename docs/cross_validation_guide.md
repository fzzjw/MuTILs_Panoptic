# MuTILs 交叉验证与自定义数据训练指南

本指南汇总 `mutils_panoptic/MuTILsTrainer.py` 与 `mutils_panoptic/RegionDatasetLoaders.py` 的关键实现细节，解释 MuTILs 如何完成交叉验证训练，并提供使用者在导入自定义数据集时需要配置的要点。

## 1. 交叉验证流程概览

MuTILs 的训练入口是 `train_mutils_fold` 函数，其负责按折（fold）循环训练并导出指标。整体流程如下：

1. **准备输出目录**：针对每个折创建 `fold_{k}` 子目录，并在其中建立 `metrics/` 与 `vis/` 文件夹存放日志和可视化结果。
2. **读取折划分**：利用 `get_cv_fold_slides` 从 `root/train_test_splits/` 下的 `fold_{k}_train.csv`、`fold_{k}_test.csv` 读取该折的训练/验证切片列表。
3. **构建数据集与采样策略**：
   - `MuTILsDataset` 会按照配置同时返回 ROI（10×）与 HPF（20×）两个尺度的图像/掩码张量，并应用 stain 增强、随机裁剪等数据增强。
   - 数据集在初始化时统计 ROI 权重，`train_loader` 使用 `WeightedRandomSampler` 以平衡切片与稀有组织类别的出现频率。
4. **搭建模型与损失函数**：`MuTILs` 模型加载多分辨率 UNet 主干，`MuTILsLoss` 聚合区域、细胞核及 CTA 评分等多头任务的损失。
5. **循环训练与评估**：
   - 每个 epoch 执行 `_train_one_mutils_epoch` 完成前向、反向传播与检查点保存。
   - 若开启评估，`evaluate_mutils` 会在验证集上计算并聚合 ROI/HPF 指标，保存每批次可视化以及 slide 级别统计，再输出整体平均值、标准差与 CTA 相关的 RMSE/相关系数。
6. **调度与可视化**：周期性更新学习率、生成损失曲线与评估指标折线，方便监控不同折的表现。

## 2. 关键组件与参数

- **`MuTILsParams` 配置**：在 `configs/panoptic_model_configs.py` 中定义了数据集路径 (`root`)、采样/增强参数、DataLoader 批大小与模型结构。用户可复制该文件并调整以匹配自身资源。
- **批次数与训练总步数**：`n_grupdates` 控制期望的梯度更新次数，结合 DataLoader 的 batch 数量动态计算 epoch 总数，适应不同折 ROI 数量的差异。
- **多 GPU 支持**：若 `CUDA_VISIBLE_DEVICES` 指定多张 GPU，模型会自动切换到 `nn.DataParallel` 进行分布式训练。

## 3. 自定义数据准备步骤

1. **目录结构**：在 `root` 路径下组织 `tcga/` 与 `acs/` 两个子目录，并在各自目录中提供平行的 `rgbs/`、`masks/` 文件夹。PNG 文件名需包含切片 ID（例如 `SLIDEID_001.png`），以便数据集正确回溯来源。
2. **掩码编码**：每个掩码为三通道 PNG，通道 0 写区域超类编码、通道 1 写细胞核类别、通道 2 写轮廓像素。请确保编码与 `RegionCellCombination.REGION_CODES`、`NUCLEUS_CODES` 等映射一致。
3. **折划分 CSV**：为每个折生成 `fold_{k}_train.csv` 与 `fold_{k}_test.csv`，至少包含 `slide_name` 列以声明该折的切片成员。文件需置于 `root/train_test_splits/`。
4. **配置文件更新**：
   - 修改 `MuTILsParams.root` 指向自定义数据根目录。
   - 根据显存与 ROI 数量调整 `train_loader_kwa['batch_size']`、`test_loader_kwa` 等。
   - 若新增组织/细胞类别，请同步更新 `RegionCellCombination` 的编码与颜色映射，并在损失函数/可视化配置中对齐。
5. **运行交叉验证**：使用 `python -m MuTILs_Panoptic.mutils_panoptic.MuTILsTrainer -f {fold_id} --dep {start_epoch}` 分别训练每个折。完成后可以对多个折的指标进行平均，或挑选单折模型执行推理。

通过以上配置，您即可基于 MuTILs 的现有训练管线开展自定义数据的交叉验证实验，并获得一致的指标导出与模型检查点。 
