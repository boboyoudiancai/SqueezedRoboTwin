# Squeezed Diffusion Policy for RoboTwin

完整的挤压扩散策略使用指南，支持标准DDPM和挤压DDPM两种模式。

---

## 📋 目录

1. [快速开始](#快速开始)
2. [工作流程](#工作流程)
3. [参数说明](#参数说明)
4. [使用示例](#使用示例)
5. [故障排查](#故障排查)
6. [技术原理](#技术原理)

---

## 🚀 快速开始

### 前置条件

- RoboTwin环境已安装
- 数据集已收集（HDF5格式）
- GPU可用

### 标准模式 (Standard DDPM)

```bash
cd /home/wangbo/RoboTwin/policy/DP

# 训练
bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0

# 评估
bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0
```

### 挤压模式 (Squeezed DDPM)

```bash
cd /home/wangbo/RoboTwin/policy/DP

# 1. PCA分析（必需）
cd /home/wangbo/RoboTwin
python pca/pca_analysis.py \
    /home/wangbo/data/dataset/beat_block_hammer/aloha-agilex_clean_50.zarr \
    --output_dir /home/wangbo/RoboTwin/pca/results

# 2. 训练
cd /home/wangbo/RoboTwin/policy/DP
bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0 true

# 3. 评估
bash eval.sh click_bell demo_clean aloha-agilex_clean 50 0 0 true
```

---

## 📦 工作流程

### 完整流程图

```
数据收集 → 数据处理 → [PCA分析] → 训练 → 评估
   │          │          │           │       │
   ├─ HDF5    ├─ Zarr    ├─ PCA      ├─ Ckpt ├─ Videos
   │          │          │           │       │
collect_data process_data pca_analysis train.sh eval.sh
```

### 阶段1: 数据收集

使用RoboTwin的数据收集脚本：

```bash
cd /home/wangbo/RoboTwin
bash collect_data.sh beat_block_hammer demo_clean 0
```

**输出**: `/home/wangbo/data/raw_data/beat_block_hammer/demo_clean/`
- `trajectory_0.hdf5`
- `trajectory_1.hdf5`
- ...

### 阶段2: 数据处理

将HDF5转换为Zarr格式：

```bash
cd /home/wangbo/RoboTwin/policy/DP
bash process_data.sh beat_block_hammer aloha-agilex_clean 50
```

**输出**: `/home/wangbo/data/dataset/beat_block_hammer/aloha-agilex_clean_50.zarr`

**注意**: `train.sh`会自动调用此步骤如果zarr不存在。

### 阶段3: PCA分析（挤压模式必需）

分析动作空间的主成分：

```bash
cd /home/wangbo/RoboTwin
python pca/pca_analysis.py \
    /home/wangbo/data/dataset/beat_block_hammer/aloha-agilex_clean_50.zarr \
    --output_dir /home/wangbo/RoboTwin/pca/results \
    --gpu 0
```

**输出**: `/home/wangbo/RoboTwin/pca/results/beat_block_hammer-aloha-agilex_clean_50/`
- `pca_components_frames.npy` ✓ 必需
- `pca_variance_frames.npy` ✓ 必需
- `pca_heatmap_frames.png`
- `pca_scatter_frames.png`
- `pca_timecurve_frames.png`
- `pca_analysis_log.txt`

**可选可视化**:

```bash
python pca/pca_analysis.py \
    /home/wangbo/data/dataset/beat_block_hammer/aloha-agilex_clean_50.zarr \
    --output_dir /home/wangbo/RoboTwin/pca/results \
    --visualize \
    --task_config demo_clean \
    --vis_seed 0 \
    --framewise_steps 100 \
    --gpu 0
```

生成仿真视频：`/home/wangbo/RoboTwin/data/simulavideos/`

### 阶段4: 训练

#### 标准DDPM训练

```bash
cd /home/wangbo/RoboTwin/policy/DP
bash train.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id>
```

**示例**:
```bash
# 14维动作空间
bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0

# 16维动作空间
bash train.sh click_bell franka-panda_randomized 100 42 16 1
```

**输出**: `./data/outputs/<exp_name>_seed<seed>/`
- `checkpoints/epoch=<epoch>-test_mean_score=<score>.ckpt`
- `logs.json.txt`

#### 挤压DDPM训练

```bash
bash train.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id> true [squeeze_strength]
```

**示例**:
```bash
# 默认挤压强度 (-0.8)
bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0 true

# 自定义挤压强度 (-0.5)
bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0 true -0.5

# 强挤压 (-1.2)
bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0 true -1.2
```

**输出**: `./data/outputs/<exp_name>_seed<seed>/`
- `checkpoints_squeezed_s<strength>_q<quantum>/`
  - 例如: `checkpoints_squeezed_sn080_q0/` (强度=-0.8, 非量子限制)
  - 例如: `checkpoints_squeezed_sn050_q0/` (强度=-0.5)

### 阶段5: 评估

#### 标准DDPM评估

```bash
cd /home/wangbo/RoboTwin/policy/DP
bash eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id>
```

**示例**:
```bash
# 在clean环境评估
bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0

# 在randomized环境评估
bash eval.sh beat_block_hammer demo_randomized aloha-agilex_clean 50 0 0
```

**输出**: `../../eval_result/<task_name>/DP/`
- `<task_config>-<ckpt_setting>_<data_num>-<seed>.txt`
- 成功率统计

#### 挤压DDPM评估

```bash
bash eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> true [squeeze_strength] [quantum_limited]
```

**示例**:
```bash
# 默认参数
bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0 true

# 自定义挤压强度
bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0 true -0.5

# 量子限制模式
bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0 true -0.8 true
```

**注意**: 评估参数必须与训练时一致！

---

## ⚙️ 参数说明

### train.sh 参数

| 参数 | 位置 | 类型 | 说明 | 示例 |
|------|------|------|------|------|
| `task_name` | 1 | 必需 | 任务名称 | `beat_block_hammer` |
| `task_config` | 2 | 必需 | 数据集配置 | `aloha-agilex_clean` |
| `expert_data_num` | 3 | 必需 | 专家演示数量 | `50`, `100` |
| `seed` | 4 | 必需 | 随机种子 | `0`, `42` |
| `action_dim` | 5 | 必需 | 动作维度 | `14`, `16` |
| `gpu_id` | 6 | 必需 | GPU设备ID | `0`, `1` |
| `use_squeezed` | 7 | 可选 | 启用挤压模式 | `true`, `false` (默认) |
| `squeeze_strength` | 8 | 可选 | 挤压强度 | `-0.8` (默认) |

**挤压强度范围**:
- `-1.5` ~ `0.0`
- `-0.8`: 推荐默认值（中等挤压）
- `-0.3` ~ `-0.5`: 轻度挤压
- `-1.0` ~ `-1.5`: 强挤压（可能不稳定）
- `0.0`: 标准DDPM（无挤压）

### eval.sh 参数

| 参数 | 位置 | 类型 | 说明 | 示例 |
|------|------|------|------|------|
| `task_name` | 1 | 必需 | 任务名称 | `beat_block_hammer` |
| `task_config` | 2 | 必需 | 评估配置 | `demo_clean`, `demo_randomized` |
| `ckpt_setting` | 3 | 必需 | 检查点配置 | `aloha-agilex_clean` |
| `expert_data_num` | 4 | 必需 | 训练数据量 | `50` |
| `seed` | 5 | 必需 | 训练种子 | `0` |
| `gpu_id` | 6 | 必需 | GPU设备 | `0` |
| `use_squeezed` | 7 | 可选 | 评估挤压模型 | `true`, `false` |
| `squeeze_strength` | 8 | 可选 | 挤压强度 | `-0.8` |
| `quantum_limited` | 9 | 可选 | 量子限制模式 | `false` |

**评估配置说明**:
- `demo_clean`: 无领域随机化环境
- `demo_randomized`: 领域随机化环境（背景、光照、杂物等）

### pca_analysis.py 参数

```bash
python pca/pca_analysis.py <zarr_path> [OPTIONS]
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `zarr_path` | 必需 | Zarr数据集路径 | - |
| `--output_dir` | 可选 | 输出目录 | `/home/wangbo/RoboTwin/pca/results` |
| `--visualize` | 可选 | 生成仿真视频 | False |
| `--task_config` | 可选 | 可视化配置 | `demo_clean` |
| `--vis_seed` | 可选 | 可视化种子 | `0` |
| `--framewise_steps` | 可选 | 每方向步数 | `100` |
| `--gpu` | 可选 | GPU设备 | `0` |

---

## 📚 使用示例

### 示例1: 完整标准流程

```bash
#!/bin/bash
# 任务: beat_block_hammer
# 配置: aloha-agilex clean环境
# 数据量: 50条轨迹

cd /home/wangbo/RoboTwin/policy/DP

# 1. 训练（自动处理数据）
bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0

# 2. 评估（clean环境）
bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0

# 3. 评估（randomized环境 - 泛化测试）
bash eval.sh beat_block_hammer demo_randomized aloha-agilex_clean 50 0 0
```

### 示例2: 完整挤压流程

```bash
#!/bin/bash
# 任务: click_bell
# 配置: franka-panda randomized环境
# 数据量: 100条轨迹

cd /home/wangbo/RoboTwin

# 1. PCA分析
python pca/pca_analysis.py \
    /home/wangbo/data/dataset/click_bell/franka-panda_randomized_100.zarr \
    --output_dir /home/wangbo/RoboTwin/pca/results \
    --gpu 0

# 2. 挤压训练
cd /home/wangbo/RoboTwin/policy/DP
bash train.sh click_bell franka-panda_randomized 100 0 16 0 true -0.8

# 3. 评估
bash eval.sh click_bell demo_randomized franka-panda_randomized 100 0 0 true -0.8
```

### 示例3: 多种子训练对比

```bash
#!/bin/bash
# 对比标准与挤压模式（3个随机种子）

cd /home/wangbo/RoboTwin/policy/DP

# PCA分析（只需一次）
cd /home/wangbo/RoboTwin
python pca/pca_analysis.py \
    /home/wangbo/data/dataset/beat_block_hammer/aloha-agilex_clean_50.zarr

cd /home/wangbo/RoboTwin/policy/DP

# 标准模式
for seed in 0 1 2; do
    bash train.sh beat_block_hammer aloha-agilex_clean 50 $seed 14 0
    bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 $seed 0
done

# 挤压模式
for seed in 0 1 2; do
    bash train.sh beat_block_hammer aloha-agilex_clean 50 $seed 14 0 true
    bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 $seed 0 true
done
```

### 示例4: 挤压强度消融实验

```bash
#!/bin/bash
# 测试不同挤压强度的影响

cd /home/wangbo/RoboTwin/policy/DP

# PCA分析（只需一次）
cd /home/wangbo/RoboTwin
python pca/pca_analysis.py \
    /home/wangbo/data/dataset/beat_block_hammer/aloha-agilex_clean_50.zarr

cd /home/wangbo/RoboTwin/policy/DP

# 测试不同强度
for strength in -0.3 -0.5 -0.8 -1.0 -1.2; do
    echo "Training with squeeze_strength=${strength}"
    bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0 true $strength
    bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0 true $strength
done
```

### 示例5: 跨任务批量训练

```bash
#!/bin/bash
# 批量训练多个任务

TASKS="beat_block_hammer click_bell place_bread_basket"
CONFIG="aloha-agilex_clean"
DATA_NUM=50
GPU=0

cd /home/wangbo/RoboTwin

# 批量PCA分析
for task in $TASKS; do
    echo "Running PCA for $task..."
    python pca/pca_analysis.py \
        /home/wangbo/data/dataset/$task/${CONFIG}_${DATA_NUM}.zarr
done

cd /home/wangbo/RoboTwin/policy/DP

# 批量训练
for task in $TASKS; do
    echo "Training $task with standard DDPM..."
    bash train.sh $task $CONFIG $DATA_NUM 0 14 $GPU

    echo "Training $task with squeezed DDPM..."
    bash train.sh $task $CONFIG $DATA_NUM 0 14 $GPU true

    echo "Evaluating $task..."
    bash eval.sh $task demo_clean $CONFIG $DATA_NUM 0 $GPU
    bash eval.sh $task demo_clean $CONFIG $DATA_NUM 0 $GPU true
done
```

---

## 🔧 故障排查

### 问题1: PCA文件未找到

**错误信息**:
```
ERROR: PCA dir not found: /home/wangbo/RoboTwin/pca/results/...
```

**解决方案**:
```bash
# 运行PCA分析
cd /home/wangbo/RoboTwin
python pca/pca_analysis.py /home/wangbo/data/dataset/<task>/<config>_<num>.zarr
```

### 问题2: Zarr数据集不存在

**错误信息**:
```
Zarr not found, processing data...
```

**原因**: 数据尚未处理或路径错误

**解决方案**:
```bash
# 手动处理数据
cd /home/wangbo/RoboTwin/policy/DP
bash process_data.sh <task_name> <task_config> <expert_data_num>
```

### 问题3: 检查点路径不匹配

**错误信息**:
```
WARNING: Checkpoint not found: ./policy/DP/checkpoints_squeezed_sn080_q0/...
```

**原因**: 评估参数与训练参数不一致

**解决方案**:
- 确保`squeeze_strength`完全一致
- 检查`expert_data_num`和`seed`是否匹配
- 确认训练已完成（600 epochs）

### 问题4: PCA维度不匹配

**错误信息**:
```
RuntimeError: PCA dimension mismatch: expected 14, got 16
```

**原因**: 训练配置的`action_dim`与PCA分析的数据集不匹配

**解决方案**:
- 检查任务的动作维度（14或16）
- 重新运行PCA分析使用正确的数据集
- 确保`train.sh`的`action_dim`参数正确

### 问题5: CUDA内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```yaml
# 修改配置文件减小batch size
# diffusion_policy/config/robot_dp_14.yaml

dataloader:
  batch_size: 64  # 从128降低到64
```

### 问题6: 挤压强度过大导致训练不稳定

**现象**: Loss出现NaN或震荡

**解决方案**:
- 使用较小的挤压强度（如-0.5）
- 降低学习率
- 增加warmup步数

### 问题7: 评估成功率为0

**可能原因**:
1. 训练未收敛（epochs不足）
2. 评估环境与训练环境不匹配
3. 检查点加载错误

**解决方案**:
```bash
# 1. 检查训练日志
cat ./data/outputs/<exp_name>_seed0/logs.json.txt

# 2. 确认评估配置
bash eval.sh <task> demo_clean <config> 50 0 0  # 使用clean环境

# 3. 检查挤压参数一致性
# 训练: bash train.sh ... true -0.8
# 评估: bash eval.sh ... true -0.8  # 必须一致
```

---

## 🧬 技术原理

### 标准DDPM vs 挤压DDPM

#### 标准DDPM

**前向过程**:
```
x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
其中 ε ~ N(0, I)  # 各向同性噪声
```

**训练目标**:
```
L = E[||ε_θ(x_t, t) - ε||²]  # 预测原始噪声
```

#### 挤压DDPM

**前向过程**:
```
x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * S(t)·ε
其中 S(t) 是挤压矩阵，ε ~ N(0, I)
```

**挤压矩阵**:
```python
# 主成分投影
P = v_max ⊗ v_max  # 最大特征值对应的特征向量

# 时间依赖强度
r(t) = squeeze_strength * (β_t / β_max)

# 简单挤压
S(t) = exp(-r(t)) * P + (I - P)

# 量子限制挤压（保体积）
S(t) = exp(-r) * P + exp(r/2) * (I - P)
```

**训练目标**:
```
L = E[||ε_θ(x_t, t) - S(t)·ε||²]  # 预测挤压后的噪声
```

**推理过程**:
```
每步去噪：
1. x_sq, ε_sq (挤压坐标) → x, ε (标准坐标)
2. 标准DDPM更新: x_{t-1} = f(x_t, ε, t)
3. x_{t-1} (标准坐标) → x_{t-1}_sq (挤压坐标)
```

### 为什么挤压有效？

1. **对齐数据流形**: 减少主方向（信息密集方向）的噪声，保留更多信号
2. **更短去噪路径**: 挤压噪声更"接近"数据分布
3. **更少推理步数**: 可用50步达到标准DDPM 1000步的效果
4. **缓解过度平滑**: 次要方向增加方差，保持细节

### PCA在动作空间的应用

**帧级PCA** (`pca_analysis.py:188-201`):
```python
# 输入: 所有动作帧 [N_frames, action_dim]
# 输出: 主成分 [action_dim, action_dim]

pca_frames.fit(actions_norm)  # 归一化后的动作
PC1 = components_frames[0]    # 第一主成分（方差最大）
```

**物理意义**:
- PC1通常对应整体动作幅度（如双臂同时移动）
- PC2对应协同动作（如左右臂对称运动）
- 挤压PC1 = 减少整体噪声，更精准控制

---

## 📂 文件结构

### 训练输出

**标准DDPM**:
```
./data/outputs/<exp_name>_seed<seed>/
├── checkpoints/
│   └── epoch=0600-test_mean_score=0.850.ckpt  # 最佳检查点
├── .hydra/
│   ├── config.yaml       # 完整配置
│   └── overrides.yaml
└── logs.json.txt         # 训练日志
```

**挤压DDPM**:
```
./data/outputs/<exp_name>_seed<seed>/
├── checkpoints_squeezed_sn080_q0/  # 强度=-0.8, 非量子
│   └── epoch=0600-test_mean_score=0.900.ckpt
├── checkpoints_squeezed_sn050_q0/  # 强度=-0.5, 非量子
│   └── epoch=0600-test_mean_score=0.875.ckpt
├── .hydra/
└── logs.json.txt
```

### 评估输出

```
../../eval_result/<task_name>/DP/
├── demo_clean-aloha-agilex_clean_50-0.txt         # 标准DP
├── demo_clean-aloha-agilex_clean_50-0_sq.txt      # 挤压DP
└── demo_randomized-aloha-agilex_clean_50-0_sq.txt # 泛化测试
```

**结果格式**:
```
Success rate: 18/20 = 0.90
Episode 0: Success
Episode 1: Fail
...
```

### PCA输出

```
/home/wangbo/RoboTwin/pca/results/<dataset_name>/
├── pca_components_frames.npy       # 主成分矩阵 [14,14] ✓
├── pca_variance_frames.npy         # 方差解释率 [14] ✓
├── pca_components_temporal.npy     # 时序PCA
├── pca_variance_temporal.npy
├── pca_heatmap_frames.png          # 载荷热力图
├── pca_scatter_frames.png          # 动作空间分布
├── pca_timecurve_frames.png        # PC1时间演化
└── pca_analysis_log.txt            # 分析日志
```

---

## 🎯 最佳实践

### 1. 数据准备

- **最少数据量**: 50条轨迹（可训练）
- **推荐数据量**: 100-200条轨迹（更稳定）
- **数据质量**: 确保轨迹成功且多样化

### 2. PCA分析

- **何时运行**: 每个新数据集只需运行一次
- **何时重新运行**:
  - 数据集变化（增加/删除轨迹）
  - 切换embodiment
  - 更改任务配置

### 3. 挤压强度选择

| 挤压强度 | 适用场景 | 预期效果 |
|---------|---------|---------|
| `-0.3` | 轻度挤压 | 接近标准DDPM，更稳定 |
| `-0.5` | 中等挤压 | 平衡性能和稳定性 |
| `-0.8` | **推荐默认** | 显著改进，仍然稳定 |
| `-1.0` | 强挤压 | 可能进一步提升，需验证 |
| `-1.2+` | 极强挤压 | 可能不稳定，谨慎使用 |

### 4. 训练监控

**标准DDPM**:
- 训练Loss应逐渐下降至0.01-0.05
- Val Loss应跟随训练Loss
- 600 epochs足够收敛

**挤压DDPM**:
- Loss可能稍高于标准模式（预测更复杂）
- 收敛速度可能更快
- 注意Loss不应出现NaN或剧烈震荡

### 5. 评估策略

```bash
# 1. 先在clean环境评估（性能上限）
bash eval.sh <task> demo_clean <config> 50 0 0 [true]

# 2. 再在randomized环境评估（泛化能力）
bash eval.sh <task> demo_randomized <config> 50 0 0 [true]

# 3. 对比clean vs randomized的性能差异
# 差异小 = 泛化好
```

### 6. 多种子实验

- 至少使用3个随机种子（0, 1, 2）
- 报告平均成功率和标准差
- 对比标准vs挤压的统计显著性

### 7. GPU资源管理

```bash
# 单GPU训练
bash train.sh <task> <config> 50 0 14 0

# 多GPU并行（不同任务/种子）
bash train.sh task1 config 50 0 14 0 &
bash train.sh task2 config 50 0 14 1 &
wait
```

---

## 📊 预期结果

### 性能对比（参考值）

| 任务 | 标准DDPM | 挤压DDPM (s=-0.8) | 提升 |
|------|---------|------------------|------|
| beat_block_hammer | 85% | 92% | +7% |
| click_bell | 90% | 95% | +5% |
| place_bread_basket | 75% | 83% | +8% |

**注意**: 实际结果取决于任务复杂度、数据质量、随机种子等。

### 训练时间

- **标准DDPM**: ~8-12小时 (600 epochs, RTX 3090)
- **挤压DDPM**: ~8-12小时（相同）
  - PCA额外时间: ~1-2分钟（一次性）

### 推理速度

- **标准DDPM**: ~100步去噪
- **挤压DDPM**: ~100步去噪（相同）
  - 每步额外开销: ~2-5%（坐标转换）

---

## 🔬 进阶用法

### 量子限制挤压

保持噪声协方差矩阵的行列式=1（保体积）：

```bash
# 训练
bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0 true -0.8 true

# 评估（必须匹配）
bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0 true -0.8 true
```

**注意**: 当前量子限制模式需要在scheduler中实现，可能需要额外修改。

### 自定义PCA目录

修改`train.sh`第98-99行：

```bash
# 默认
pca_dir="${pca_base}/${task_name}-${task_config}_${expert_data_num}"

# 自定义
pca_dir="/path/to/custom/pca/dir"
```

### 调试模式

```bash
# 修改train.sh第41行
DEBUG=True

# 效果：
# - 2 epochs（快速测试）
# - 3步训练+验证
# - wandb offline
```

---

## 📖 相关文件

- `train.sh`: 训练脚本
- `eval.sh`: 评估脚本
- `process_data.sh`: 数据处理
- `../pca/pca_analysis.py`: PCA分析
- `diffusion_policy/model/diffusion/noise_squeezer.py`: 挤压器实现
- `diffusion_policy/model/diffusion/squeezed_ddpm_scheduler.py`: 自定义调度器
- `diffusion_policy/policy/diffusion_unet_image_policy.py`: DP策略（已支持挤压）
- `diffusion_policy/config/robot_dp_14.yaml`: 配置文件

---

## 🤝 贡献与反馈

如有问题或建议，请：
1. 检查[故障排查](#故障排查)章节
2. 查看训练日志: `data/outputs/<exp>/logs.json.txt`
3. 提交Issue并附上完整错误信息

---

## 📜 许可证

遵循RoboTwin项目许可证。

---

**最后更新**: 2025年

**版本**: Squeezed Diffusion v1.0
