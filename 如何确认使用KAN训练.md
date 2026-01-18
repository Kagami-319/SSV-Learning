# 如何确认正在使用 KAN 训练

## ✅ 已修复的问题

您的原始代码存在类定义冲突：
- `FCNet1D` (第56行) - 使用普通 MLP
- `FCNet` (第208行) - 使用 KAN

但 `main` 函数调用的参数与 MLP 版本匹配，导致**实际使用的是 MLP，而不是 KAN**。

现在已修复：
- ✅ 将 KAN 版本重命名为 `FCNetKAN`
- ✅ 修复了 forward 方法签名，使其与训练代码兼容
- ✅ 添加了 `EfficientKAN` 导入
- ✅ 更新了 main 函数以正确实例化 KAN 模型

## 🔍 如何确认正在使用 KAN

### 方法 1: 查看启动信息（最简单）

运行训练命令时，会看到：

```bash
python train_burgers_1d.py --model fcnet --truth truth_burgers_1d.npz ...
```

输出会显示：

```
============================================================
✓ 使用 KAN 版本的 FCNet
  - Basis dimension: 128
  - Branch/Trunk width: 128
  - Branch/Trunk depth: 4
============================================================
Training PHYS ...
✓ KAN is being used!
[PHYS-FCNET] Epoch 50/800  MSE = 1.2345e-03
...
```

**关键标志：**
- ✅ 看到 "使用 KAN 版本的 FCNet"
- ✅ 看到 "KAN is being used!"

### 方法 2: 检查模型参数数量

KAN 模型由于使用样条基函数，参数数量会**显著大于**普通 MLP。

在训练代码中添加：

```python
# 在 main() 函数中，创建模型后添加：
if arch == 'fcnet':
    num_params = sum(p.numel() for p in model_phys.parameters())
    print(f"模型参数总数: {num_params:,}")
    print(f"预期: KAN > 500K, MLP < 200K (取决于配置)")
```

**典型参数数量对比：**
- MLP FCNet (width=128, depth=4): ~100-200K 参数
- KAN FCNet (width=128, depth=4, grid_size=5): ~500K-1M+ 参数

### 方法 3: 检查模型结构

在 Python 中检查模型：

```python
import torch

# 加载检查点
ckpt = torch.load('ckpt_fcnet_physical.pt')
print(ckpt.keys())
print(ckpt['arch'])

# 检查 state_dict 中的键
for key in list(ckpt['state_dict'].keys())[:10]:
    print(key)
```

**KAN 模型会包含这些键：**
```
core.branch.layers.0.base_weight
core.branch.layers.0.spline_weight
core.branch.layers.0.spline_scaler
...
core.trunk.layers.0.base_weight
core.trunk.layers.0.spline_weight
...
```

**MLP 模型只有标准的权重：**
```
branch.net.0.weight
branch.net.0.bias
branch.net.2.weight
...
```

### 方法 4: 查看训练日志

KAN 训练的特征：
- ✅ 训练速度较慢（比 MLP 慢 2-3 倍）
- ✅ 内存占用更大
- ✅ 可能达到更好的收敛精度

## 🚀 运行示例

### 使用 KAN 训练

```bash
python train_burgers_1d.py \
    --model fcnet \
    --truth artifacts/truth_burgers_1d.npz \
    --t_train_max 10 \
    --epochs 800 \
    --batch 2048 \
    --nsamples 200000 \
    --width 128 \
    --depth 4 \
    --latent 128 \
    --phys_ckpt artifacts/ckpt_fcnet_physical_kan.pt \
    --ssv_ckpt artifacts/ckpt_fcnet_ssv_kan.pt
```

### 使用普通 MLP 训练（对比）

```bash
python train_burgers_1d.py \
    --model concat \
    --truth artifacts/truth_burgers_1d.npz \
    --t_train_max 10 \
    --epochs 800 \
    --batch 2048 \
    --nsamples 200000 \
    --width 128 \
    --depth 4 \
    --phys_ckpt artifacts/ckpt_concat_physical_mlp.pt \
    --ssv_ckpt artifacts/ckpt_concat_ssv_mlp.pt
```

## 📊 预期差异

| 特性 | MLP | KAN |
|------|-----|-----|
| 训练速度 | 快 | 慢 2-3x |
| 内存占用 | 小 | 大 3-5x |
| 参数数量 | ~100-200K | ~500K-1M+ |
| 收敛精度 | 标准 | 可能更好 |
| 外推能力 | 一般 | 理论上更强 |

## 🔧 故障排除

### 问题: 没有看到 "KAN is being used!"

**原因**: 可能使用了 `--model concat`，这会使用普通 MLP

**解决**: 确保使用 `--model fcnet` 或 `--model fc`

### 问题: 导入错误 "cannot import name 'EfficientKAN'"

**原因**: `nets.py` 文件不在同一目录或未定义 `EfficientKAN`

**解决**: 
1. 确保 `nets.py` 在同一目录
2. 确保 `nets.py` 中有 `EfficientKAN` 类定义

### 问题: CUDA out of memory

**原因**: KAN 占用更多内存

**解决**: 
- 减小 batch size: `--batch 1024` 或更小
- 减小模型尺寸: `--width 64 --depth 3`
- 使用 CPU: 代码会自动检测

## 📝 验证清单

训练时请确认：

- [ ] 看到 "✓ 使用 KAN 版本的 FCNet"
- [ ] 看到 "✓ KAN is being used!"
- [ ] 训练日志显示 `[PHYS-FCNET]` 或 `[SSV-FCNET]`
- [ ] 训练速度比预期慢（这是正常的，KAN 更复杂）
- [ ] 保存的检查点文件存在

如果以上都确认了，恭喜！您正在使用 KAN 进行训练！🎉
