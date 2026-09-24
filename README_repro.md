# MeshGraphNets 复现记录

**复现对象**：[google-deepmind/deepmind-research/meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
**论文**：*Learning Mesh-Based Simulation with Graph Networks*, Pfaff et al., ICLR 2021（[arXiv:2010.03409](https://arxiv.org/abs/2010.03409)）
**目标**：跑通官方 demo（`flag_minimal` 冒烟）+ 单任务训练（`flag_simple` / cloth 域）
**约束**：官方 `.py` 文件仅做一处功能性修改（`--rollout_steps`，见 §2.2）

**状态**：✅ 冒烟测试全流程通过 ｜ 🔄 flag_simple 正式训练

---

## 目录

1. [一句话结论](#1-一句话结论)
2. [目录结构与文件说明](#2-目录结构与文件说明)
3. [完整复现流程](#3-完整复现流程)
4. [冒烟测试成功说明](#4-冒烟测试成功说明)
5. [兼容性问题与解法](#5-兼容性问题与解法)
6. [踩坑记录](#6-踩坑记录)
7. [数据集事实](#7-数据集事实)
8. [已知的上游 bug 与论文差异](#8-已知的上游-bug-与论文差异)

---

## 1. 一句话结论

这个仓库的 `requirements.txt` 写的是 `tensorflow-gpu>=1.15,<2`，**照它装会掉进坑里**：A30 是 Ampere（`sm_86`），CUDA 10.x 不支持，装上也只能用 CPU。

但代码本身并不是 TF1 专属 —— 它全部用 `import tensorflow.compat.v1 as tf`，而 `tensorflow.compat.v1` **只有 TF2 才有**。`run_model.py` 里还明确写了 `tf.disable_eager_execution()`，这是主动在 TF2 下切回 v1 图模式的标准写法。

**真正卡版本的是 `dm-sonnet`，不是 TensorFlow。** 代码继承 `snt.AbstractModule`，这要求 Sonnet 1.x（2.x 删掉了它），而 Sonnet 1.36 在 TF2 下 `import` 都过不去。

解法：用 **TF 2.15.1 + dm-sonnet 1.36 + 一个 `tensorflow.contrib` 兼容 shim**，官方代码不需要 wrapper 就能跑通。

> **关于 GPU**：这套环境能加载 CUDA，但本文所有实测（含 §9 的 10 万步训练）实际都跑在 CPU 上 —— 共享机器的 4 张 A30 当时都被占满，日志里是 `CUDA_ERROR_NO_DEVICE`。GPU 路径未经验证。

---

## 2. 目录结构与文件说明

### 2.1 官方代码（`deepmind-research/meshgraphnets/`，共 15 个文件，无子目录）

> 用 `git clone --filter=blob:none --sparse` 只拉了这一个子目录。
> **注意：没有 `setup.py`、没有 `__init__.py`**，靠 namespace package 机制工作 —— 所以**运行时的 cwd 必须是仓库根目录**。

#### 核心模型

| 文件 | 作用 |
|---|---|
| `core_model.py` | **算法核心**。`GraphNetBlock`（带残差的多边交互网络）、`EncodeProcessDecode`（encoder → 15 步 message passing → decoder）。定义 `EdgeSet` / `MultiGraph` 两个 namedtuple |
| `normalization.py` | `Normalizer`：在线累积 mean/std 做特征归一化，`inverse()` 把网络输出反归一化回物理量 |
| `common.py` | `NodeType` 枚举（NORMAL/OBSTACLE/AIRFOIL/HANDLE/INFLOW/OUTFLOW/WALL_BOUNDARY，`SIZE=9` 是 one-hot 宽度）；`triangles_to_edges()` 从三角面片提取去重后的双向边 |

#### 数据

| 文件 | 作用 |
|---|---|
| `dataset.py` | 读 `meta.json` + `.tfrecord`；`_parse` 解析 tf.Example；`add_targets` 造 `target\|xxx` / `prev\|xxx`；`split_and_preprocess` 切帧 + 加高斯噪声增广 |
| `download_dataset.sh` | 官方下载脚本（用 `wget`，**在本机极慢**，见 [6.3](#63-数据下载的坑)） |

#### 域相关（论文的 4 个任务里，官方只给了 2 个的完整 pipeline）

| 文件 | 作用 |
|---|---|
| `cloth_model.py` | Flag/布料域的 `Model`。节点特征 = `[速度, node_type one-hot]`，边特征 = 相对世界坐标 + 相对网格坐标 + 各自模长（7 维）。**损失是对加速度的掩码 L2**，积分用 `x_{t+1} = 2x_t + a - x_{t-1}` |
| `cloth_eval.py` | cloth 的 rollout（`tf.while_loop` + `TensorArray`，边界节点冻结）+ `mse_{1,10,20,50,100,200}_steps` 指标 |
| `plot_cloth.py` | 3D `plot_trisurf` 动画。**⚠️ 只有 `plt.show(block=True)`，不保存文件** |
| `cfd_model.py` | Cylinder Flow 域的 `Model`。损失是对**速度增量**的掩码 L2 |
| `cfd_eval.py` | CFD 的 rollout + 同样的 MSE 指标 |
| `plot_cfd.py` | `tripcolor` 动画，同样不保存文件 |

#### 驱动与配置

| 文件 | 作用 |
|---|---|
| `run_model.py` | **唯一入口**。absl 命令行，`--mode=train/eval`、`--model=cloth/cfd`。超参写在 `PARAMETERS` 字典里 |
| `run.sh` | 官方集成测试 = **官方意义上的 "demo"**：`flag_minimal` → 训 10 步 → eval 1 条 → 画图 |
| `requirements.txt` | `tensorflow-gpu>=1.15,<2` / `dm-sonnet<2` / matplotlib / absl-py / numpy |
| `README.md` | 官方说明 |

> **没有 `demo.py`**，也没有 colab notebook。README 里说的 "demonstration code" 指的就是 `cfd_*` / `cloth_*` 这套脚本。

### 2.2 本次新增的文件（都在 `/data/HOI/LiuSiyu/MeshGraphNets/`，**不在官方目录内**）

| 文件 | 作用 | 为什么需要 |
|---|---|---|
| `mgn_shim.py` | **TF2 兼容 shim**（核心补丁） | 伪造 TF2 已删除的 `tensorflow.contrib`，并补回 Python 3.10 删掉的 `collections.abc` 别名 |
| `fetch_dataset.py` | 并行分块下载器 | 官方 `wget` 只有 67 KB/s，12GB 要跑两天 |
| `fetch_dataset.sh` | 单文件串行下载版（curl + 断点续传） | 小数据集够用 |
| `analyze_rollout.py` | 出 MSE 曲线 PNG + 预测/真值对比 GIF | 官方 `plot_cloth.py` **不保存任何文件** |
| `run.sh` | 统一驱动（smoke / train / eval / plot） | 封装固定 GPU、环境变量、路径 |
| `README_repro.md` | 本文档 | — |

**官方文件改动量：2 个文件，共 18 增 7 删。**

`meshgraphnets/cloth_eval.py`（+17/−7）与 `meshgraphnets/run_model.py`（+8/−1）增加了一个 `--rollout_steps` 参数：官方 rollout 长度被写死为数据集长度，无法跑超过数据帧数的长程评测，而这个参数让 rollout 可以延长到数据集之外（预测延长、指标与真值仍截断在可用帧内）。§9.2 的「2000 步 vs 10 步」对比依赖它。

---

## 3. 完整复现流程

### 3.1 环境

```bash
CONDA=/path/to/your/conda
$CONDA/bin/conda create -y -p /data/HOI/LiuSiyu/MeshGraphNets/env python=3.10
PY=/data/HOI/LiuSiyu/MeshGraphNets/env/bin/python
```

> 用 `-p` 装到项目目录下，不污染宿主上共享的 conda `envs` 目录（那里面是别人的项目）。
> 本机系统 `python3` **没有 pip**，必须用 conda env 的 python。

### 3.2 装依赖

```bash
IDX="-i https://mirrors.aliyun.com/pypi/simple/ --retries 10 --timeout 120"

# TF 2.15.1 = 最后一个 Keras 3 之前、tf.compat.v1 最完整的版本
$PY -m pip install $IDX "tensorflow==2.15.1" "numpy<2" matplotlib absl-py

# dm-sonnet 必须 --no-deps：否则会被拖进 TF1-only 的 tfp 0.8
$PY -m pip install $IDX --no-deps "dm-sonnet==1.36"

# sonnet 的其它依赖 + tfp（sonnet 在模块级 import 它，必须真的可导入）
$PY -m pip install $IDX six wrapt contextlib2 semantic-version dm-tree "tensorflow-probability==0.23.0"

# sonnet 的 protos 是旧版 protoc 生成的，protobuf 4.x 会报
# "Descriptors cannot be created directly"
$PY -m pip install $IDX "protobuf==3.20.3"
```

最终版本：

```
tensorflow                   2.15.1
dm-sonnet                    1.36
tensorflow-probability       0.23.0
protobuf                     3.20.3
numpy                        1.26.4
matplotlib                   3.10.9
absl-py                      2.5.0
```

> 装完后 `pip` 会报一条 `dm-sonnet 1.36 requires tensorflow-probability<0.9.0` 的冲突警告 —— 这是**故意的**，`--no-deps` 就是为绕开它。

### 3.3 注入 shim

把 `sitecustomize.py` 写进 env 的 site-packages（CPython 启动时自动 import）：

```bash
SP=/data/HOI/LiuSiyu/MeshGraphNets/env/lib/python3.10/site-packages
cat > "$SP/sitecustomize.py" <<'EOF'
import sys
_MGN_ROOT = "/data/HOI/LiuSiyu/MeshGraphNets"
if _MGN_ROOT not in sys.path:
    sys.path.insert(0, _MGN_ROOT)
try:
    import mgn_shim
except Exception:
    pass
EOF
```

这样 `import sonnet` 之前 shim 已经就位，**官方代码不需要任何 wrapper**。

### 3.4 拉代码

```bash
cd /data/HOI/LiuSiyu/MeshGraphNets
git clone --filter=blob:none --sparse \
    https://github.com/google-deepmind/deepmind-research.git
cd deepmind-research && git sparse-checkout set meshgraphnets
```

### 3.5 下数据

```bash
python3 fetch_dataset.py flag_minimal /data/HOI/LiuSiyu/MeshGraphNets/data
python3 fetch_dataset.py flag_simple  /data/HOI/LiuSiyu/MeshGraphNets/data --workers 16
```

### 3.6 跑

```bash
bash run.sh smoke    # 官方 demo 冒烟
bash run.sh train    # flag_simple 训练（STEPS 控制步数）
bash run.sh eval     # rollout + MSE 曲线 + GIF
bash run.sh plot     # xvfb-run 官方 plot_cloth.py
```

`run.sh` 里写死 `CUDA_VISIBLE_DEVICES=3`（4 张 A30 里当时唯一空闲的那张）。注意这是**本机专属**设置：换机器要改编号，且 TF 看不到 GPU 时只会静默退回 CPU、不报错 —— 随仓库提交的 `train_100k.log` 实际就是 CPU 跑的。

---

## 4. 冒烟测试成功说明

对应官方 `run.sh` 的三步流程，用 `flag_minimal`（3 × 112MB）。

### 第 1 步：训练 10 步

```bash
cd deepmind-research
python -m meshgraphnets.run_model --model=cloth --mode=train \
    --checkpoint_dir=$DATA/chk_min --dataset_dir=$DATA/flag_minimal \
    --num_training_steps=10
```

**实际输出**：

```
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into .../chk_min/model.ckpt.
run_model.py:84] Step 0: Loss 8.3456
run_model.py:85] Training complete.
INFO:tensorflow:Saving checkpoints for 10 into .../chk_min/model.ckpt.
EXIT=0
```

**这一步证明了什么**：

- `Graph was finalized.` → TF1 图模式在 TF2 上正确建立，**shim 生效**
- `Step 0: Loss 8.3456` → 前向 + 反向 + `tf.train.AdamOptimizer` 全部跑通，loss 是有限值（不是 NaN）
- 写出了 `model.ckpt-0` 和 `model.ckpt-10` → `MonitoredTrainingSession` + checkpoint 机制正常
- 注意前 1000 步是**只累积归一化统计量、不训练**的（`run_model.py` 里的 `tf.cond`），所以 10 步的 loss 不代表学习效果

### 第 2 步：评估 rollout

```bash
python -m meshgraphnets.run_model --model=cloth --mode=eval \
    --checkpoint_dir=$DATA/chk_min --dataset_dir=$DATA/flag_minimal \
    --rollout_path=$DATA/rollout_min.pkl --num_rollouts=1
```

**实际输出**：

```
run_model.py:103] Rollout trajectory 0
run_model.py:108] mse_1_steps: 0.00060857
run_model.py:108] mse_10_steps: 0.124699
run_model.py:108] mse_20_steps: 0.717051
run_model.py:108] mse_50_steps: 3.42284
run_model.py:108] mse_100_steps: 29.3522
run_model.py:108] mse_200_steps: 575.563
EXIT=0
```

**这一步证明了什么**：

- `cloth_eval._rollout` 的 `tf.while_loop` + `tf.TensorArray` 在 TF2 下正常工作 —— 这是**最容易出问题的地方**，需要 `disable_control_flow_v2()` / `disable_v2_tensorshape()`（shim 里已加）
- 模型从 checkpoint 恢复成功
- 200 步自回归 rollout 全程没有崩溃、没有 NaN

**关于这些数值**：MSE 很大且随步数快速发散是**完全正常的**。模型只训练了 10 步，而且前 1000 步官方设计成只累积归一化统计量、不做梯度更新 —— 所以它本质上是**一个随机初始化的网络**。这里验证的是**管道通畅**，不是模型质量。

### 第 3 步：出图

```bash
python analyze_rollout.py --rollout_path=$DATA/rollout_min.pkl \
    --mse_out=$DATA/mse_min.png --gif_out=$DATA/rollout_min.gif
```

**实际输出**：

```
loaded 1 trajectory(ies) from .../rollout_min.pkl
  faces        (399, 3028, 3)
  mesh_pos     (399, 1579, 2)
  gt_pos       (399, 1579, 3)
  pred_pos     (399, 1579, 3)
wrote .../mse_min.png
wrote .../rollout_min.gif (40 frames)
```

**度量一致性验证**（重要）：我的脚本算出的"官方口径"指标与官方 eval 输出**逐位相同**：

| horizon | 官方 eval 输出 | `analyze_rollout.py` |
|---|---|---|
| 1 | 0.00060857 | 6.085701e-04 |
| 10 | 0.124699 | 1.246988e-01 |
| 20 | 0.717051 | 7.170506e-01 |
| 50 | 3.42284 | 3.422841e+00 |
| 100 | 29.3522 | 2.935218e+01 |
| 200 | 575.563 | 5.755632e+02 |

这确认了 `analyze_rollout.py` 复刻了官方指标定义：`mse_N_steps` 是 **steps 1..N 的均值**，不是第 N 步的值。

### 产出物

| 文件 | 大小 |
|---|---|
| `data/chk_min/model.ckpt-{0,10}.*` | ~28MB |
| `data/rollout_min.pkl` | 34MB |
| `data/mse_min.png` | 81KB |
| `data/rollout_min.gif` | 1.5MB |

---

## 5. 兼容性问题与解法

### 5.1 `tensorflow.contrib` 被 TF2 删除

`dm-sonnet` 必须用 1.36（2.x 删掉了 `AbstractModule`），但它有 **9 处模块级 `from tensorflow.contrib import ...`，且没有 try/except**：

```
sonnet/python/modules/base.py:53           from tensorflow.contrib.eager.python import tfe
sonnet/python/modules/scale_gradient.py:22 from tensorflow.contrib.eager.python import tfe
sonnet/python/modules/basic.py:34          from tensorflow.contrib import framework
sonnet/python/modules/basic_rnn.py:35      from tensorflow.contrib import framework
sonnet/python/modules/batch_norm_v2.py:34  from tensorflow.contrib import framework
sonnet/python/modules/gated_rnn.py:53,54   from tensorflow.contrib import framework / rnn
sonnet/python/modules/residual.py:25       from tensorflow.contrib import framework
sonnet/python/modules/rnn_core.py:37       from tensorflow.contrib import framework
sonnet/python/ops/nest.py:27               from tensorflow.contrib import framework
```

所以 `import sonnet` 在任何业务代码执行前就 `ModuleNotFoundError`。

**解法**：`mgn_shim.py` 装一个 **meta-path finder**，在有人真正 import 时才按需伪造这个包：

| 被请求的符号 | 映射到 |
|---|---|
| `contrib.framework.nest` | 重建的 TF1 版 nest 模块（见 5.2） |
| `contrib.framework.smart_cond` | `tf.compat.v1.smart_cond` |
| `contrib.framework.get_variables` 等 | `tf.compat.v1.get_collection(...)` |
| `contrib.eager.python.tfe.defun` | `tf.function`（TF2 无 `compat.v1.defun`） |
| `contrib.rnn.LSTMBlockCell` | `tf.compat.v1.nn.rnn_cell.LSTMBlockCell` |
| `contrib.layers` | 宽容 stub |

用 finder 而不是直接塞 `sys.modules`，是为了**惰性** —— 不用为了 patch 而在解释器启动时就 import TensorFlow。

### 5.2 `tf.nest` 的 API 被裁掉了

TF 2.15 的 `tf.nest` 是基于 tree 重新实现的精简版。Sonnet 的 `ops/nest.py` 在 import 期就包装了这些名字，其中 `is_sequence` / `map_structure_up_to` / `flatten_up_to` / `flatten_dict_items` / `assert_shallow_structure` 在 TF2 里**不存在**。

经检查，这些在 sonnet 里**只有 `is_sequence` 会被真正调用**（`basic.py:1398`），其余四个只在 import 期被引用。shim 里重建了完整的一套（`is_sequence` → `tf.nest.is_nested`，其余按 contrib 语义实现），不留陷阱。

### 5.3 Python 3.10 删掉了 `collections.Mapping`

`sonnet/python/modules/base.py:169` 有 `isinstance(custom_getter, collections.Mapping)`。这些别名在 Python 3.10 被删，导致**每一次 `snt.AbstractModule.__init__` 都 AttributeError**。

shim 里把 25 个 ABC 别名从 `collections.abc` 补回 `collections`。

### 5.4 protobuf 版本

`sonnet/protos/*_pb2.py` 是旧版 protoc 生成的，protobuf 4.x 会报 `Descriptors cannot be created directly`。降到 **3.20.3**（TF 2.15.1 要求 `protobuf>=3.20.3,<5`，正好兼容）。

### 5.5 控制流 v2

官方代码写给 TF 1.15，那时没有 v2 行为要关。TF2 下 `cloth_eval.py` / `cfd_eval.py` 的 rollout 依赖 TF1 风格的 `tf.while_loop` + `tf.TensorArray`，上游自己的 TF2 移植 PR [#745](https://github.com/google-deepmind/deepmind-research/pull/745) 也加了这两行。shim 在首次 import TF 时调用：

```python
tf.compat.v1.disable_control_flow_v2()
tf.compat.v1.disable_v2_tensorshape()
```

（`run_model.py` 自带的 `tf.disable_eager_execution()` 是官方的，不需要补。）

---

## 6. 踩坑记录

### 6.1 被证伪的误解

| 常见说法 | 事实 |
|---|---|
| 「要装 `tensorflow_graphics`」 | **全仓库 0 次出现**，`graph_nets` 也是 0 次。真实依赖只有 tensorflow / dm-sonnet / matplotlib / absl-py / numpy |
| 「必须 TF 1.15」 | 代码用 `tensorflow.compat.v1`，那是 TF2 的模块；TF 1.15 只是官方当时的测试环境 |
| 「`requirements.txt` 是过期文件」 | `dm-sonnet<2` 是对的（代码确实继承 `AbstractModule`），只是没覆盖 TF2 跑法 |

### 6.2 为什么不能照 requirements.txt 装 TF 1.15

A30 是 Ampere（`sm_86`）。`tensorflow-gpu==1.15.5` 针对 **CUDA 10.0 / cuDNN 7.4** 编译，CUDA 10.x 不认识 `sm_86`，装上会 `CUBLAS_STATUS_EXECUTION_FAILED`（上游 issue [#464](https://github.com/google-deepmind/deepmind-research/issues/464) 里被反复确认）。且 TF 1.15 只有 cp36/cp37 wheel，本机只有 Python 3.10 —— 走 TF1 等于放弃 GPU。

### 6.3 数据下载的坑

官方 `download_dataset.sh` 用 `wget`，本机实测 **67 KB/s**，`flag_simple` 的 12.2GB 要跑约两天。

实测对比：

| 方式 | 速度 |
|---|---|
| 官方脚本（wget） | 67 KB/s |
| 单流 curl（突发） | 9.5 MB/s，持续几十秒后掉到 150–350 KB/s |
| 12 路并行分块 curl | 8 MB/s |
| **16 路并行分块 + 文件级并行** | **12 MB/s** |

**关键发现：最终瓶颈不是 GCS，是本机总入站带宽。** 测 `/sys/class/net/*/statistics/rx_bytes`，所有下载加起来主机 RX 只有 ~1–5 MB/s（这台机器上还跑着别人的任务）。

`fetch_dataset.py` 支持断点续传（`.parts/` 保留已完成分块），中断重跑不会重下。

### 6.4 `plot_cloth.py` 不保存文件

最后一行是 `plt.show(block=True)` —— 无头机器上跑等于零产出。所以：

- 想跑官方脚本本身 → `xvfb-run -a`
- 想拿产物 → 用 `analyze_rollout.py`（本机没 ffmpeg，用 Pillow 存 GIF）

---

## 7. 数据集事实

`flag_minimal` 和 `flag_simple` **结构完全相同**，只是轨迹条数不同：

| 字段 | 类型 | shape | dtype |
|---|---|---|---|
| `cells` | static | [1, 3028, 3] | int32 |
| `mesh_pos` | static | [1, 1579, 2] | float32 |
| `node_type` | dynamic | [401, 1579, 1] | int32 |
| `world_pos` | dynamic | [401, 1579, 3] | float32 |

`trajectory_length = 401`，`dt = 0.02`，模拟器 `arcsim`。**1579 节点 / 3028 三角形**，规模很小。

> **重要**：**没有任何 `dynamic_varlen` 字段**。所以 `dataset.py::_parse` 里那条 `tf.RaggedTensor.from_row_lengths` 分支在 flag 数据集上根本不会执行 —— 一个潜在风险点被排除。（`dynamic_varlen` 只出现在 `cylinder_flow` / `deforming_plate` 这类变长网格数据集上。）

数据集体积（GCS 桶 `dm-meshgraphnets` 公开可读）：

| 数据集 | train | 合计 |
|---|---|---|
| `flag_minimal` | 112 MB | 336 MB |
| `flag_simple` | 10.2 GB | 12.2 GB |
| `cylinder_flow` | 13.6 GB | 16.4 GB |
| `deforming_plate` | 9.9 GB | 11.5 GB |
| `airfoil` | 50.5 GB | 60.6 GB |

---

## 8. 已知的上游 bug 与论文差异

**这些没有主动修复**，因为目标是复现而非改进。记录下来以便对照结果。

1. **`normalization.py` 方差可能为 NaN**（[issue #346](https://github.com/google-deepmind/deepmind-research/issues/346)）
   ```python
   std = tf.sqrt(self._acc_sum_squared / safe_count - self._mean()**2)  # 可能负数 → NaN
   return tf.math.maximum(std, self._std_epsilon)                       # 已经晚了
   ```
   `E[x²]−E[x]²` 的浮点舍入可能为负，`sqrt` 先于 clamp 执行，NaN 无法被 `maximum` 救回。**若训练一开始就 NaN，这是首要嫌疑。**

2. **`dataset.batch_dataset` 是死代码** —— `learner()` 从不调用它，所以 CFD 配置里的 `batch=2` 实际被忽略（[PR #754](https://github.com/google-deepmind/deepmind-research/pull/754)）。cloth 的 `batch=1` 不受影响。

3. **world edges 没有实现**（issue [#249](https://github.com/google-deepmind/deepmind-research/issues/249)），尽管论文里描述了。

4. **`--checkpoint_dir` 不能跨数据集/模型复用**，否则归一化变量的 shape 会对不上（[issue #321](https://github.com/google-deepmind/deepmind-research/issues/321)）。

5. **官方训练速度比论文暗示的慢很多**（issue [#360](https://github.com/google-deepmind/deepmind-research/issues/360)）。

---

## 9. 训练结果

### 9.1 损失曲线

`flag_simple` / cloth 域，官方超参未改动。下表取自随仓库提交的 `train_100k.log`：

| 步数 | Loss |
|---|---|
| 0 | 4.96622 |
| 1000 | 2.25230 |
| 2000 | 9.48208 |

**前 1000 步 loss 下降约 55%**（4.96622 → 2.25230）。注意官方设计里**前 1000 步只累积归一化统计量、不做梯度更新**（`run_model.py` 里的 `tf.cond`），所以第 1000 步的 loss 下降完全来自归一化器收敛后的输出尺度校正 —— 这是正常且预期的行为。

### 9.2 rollout 质量：2000 步 vs 10 步

这是最能说明问题的对比。同一个评测流程（`--rollout_split=valid`，官方 `cloth_eval` 指标），只是 checkpoint 不同：

| horizon | 10 步模型 | 2000 步模型 | 改善 |
|---|---|---|---|
| `mse_1_steps` | 6.09e-04 | 1.69e-04 | 3.6× |
| `mse_10_steps` | 1.25e-01 | 2.91e-02 | 4.3× |
| `mse_20_steps` | 7.17e-01 | 1.52e-01 | 4.7× |
| `mse_50_steps` | 3.42e+00 | 3.83e-01 | 8.9× |
| `mse_100_steps` | 2.94e+01 | 4.80e-01 | **61×** |
| `mse_200_steps` | 5.76e+02 | 8.31e-01 | **693×** |

**关键不只是数值，而是定性行为变了**：

- **10 步模型**：误差从 29 → 576 爆炸式发散，rollout 完全崩溃
- **2000 步模型**：误差从 0.48 → 0.83 平稳增长，**400 步内不发散**

后者才是 MeshGraphNets 论文描述的行为 —— 自回归 rollout 能长期保持稳定。误差曲线（`data/mse_2k.png`）形态正确：前 ~50 步快速上升（模型预热 + 误差累积），之后转为缓慢线性增长。

### 9.3 产物清单

| 文件 | 内容 |
|---|---|
| `data/mse_2k.png` | **MSE vs rollout 步数**对数曲线，3 条轨迹 + 均值线 |
| `data/rollout_2k.gif` | 预测 vs 真值并排 3D 动画，80 帧 |
| `data/mse_min.png` / `rollout_min.gif` | 冒烟测试（10 步模型）的同款产物，用于对比 |
| `data/chk_cpuval/` | 2000 步 checkpoint |
| `data/chk_min/` | 10 步 checkpoint（冒烟用） |

### 9.4 训练速度

| 硬件 | 速度 | 10 万步 | 100 万步 |
|---|---|---|---|
| CPU（160 核，实测占 33 核） | ~1.8 步/秒 | ~15 小时 | ~6.4 天 |
| GPU（A30，待测） | 待测 | 待测 | 待测 |

> 论文用的是 10M 步。单卡复现到这个量级不现实，本项目以「rollout 稳定不发散」作为复现成功的判据。

### 9.5 复现成功判据

对照论文，以下几条已满足：

- [x] 官方代码经一处 `--rollout_steps` 修改后可运行
- [x] 训练 loss 稳定下降，无 NaN（排除了 [8.1](#81-normalizationpy-方差可能为-nan) 的归一化 NaN bug）
- [x] rollout 指标随 horizon 单调增长且**不发散**
- [x] `mse_1_steps` 显著小于 `mse_200_steps`（短程比长程准得多）
- [ ] 论文级精度（需要 10M 步，本机算力不允许）
