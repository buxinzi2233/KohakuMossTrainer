[English](../../performance/training-5090.md)

<a id="training-speed-on-the-rtx-5090"></a>
# RTX 5090 上的训练速度

本文记录单张及四张 32 GB RTX 5090 上 LoRA 和全量微调的每秒 token 吞吐量，解释 5090 训练配方的设置依据，并比较选择这些设置时测试过的相邻配置。

| 配方 | GPU 数量 | 设置 |
| --- | --- | --- |
| `recipes_lora_5090.py` | 1 | LoRA r32、bf16，`MAX_TOKENS=4608`，`CHECKPOINT="partial:0.2"`，`GRAD_ACC=64` |
| `recipes_lora_ddp_4x5090.py` | 4 | 每个 rank 使用相同配置，`PARALLEL="ddp"`，`GRAD_ACC=16` |
| `recipes_lokr_5090.py` | 1 | LoKr factor 8、全矩阵、bf16，`MAX_TOKENS=4608`，`CHECKPOINT="partial:0.2"`，`GRAD_ACC=64`（需要 lycoris-lora ≥ 4.1.3） |
| `recipes_lokr_ddp_4x5090.py` | 4 | 每个 rank 使用相同配置，`PARALLEL="ddp"`，`CHECKPOINT="partial:0.262"`，`GRAD_ACC=16` |
| `recipes_full_5090.py` | 1 | 全量微调，`ckpt-ring8-actoff-optstream`，`MAX_TOKENS=36864` |
| `recipes_full_ddp_4x5090.py` | 4 | 每个 rank 使用相同配置，`PARALLEL="ddp"`（主机侧的主权重副本仅保留一份共享拷贝） |

LoRA 和 LoKr 配方通过梯度累积与多个 GPU，把每次优化器更新的总 token 预算固定为 **294,912（64 × 4608）**。下面的[单步 294,912 token 对比](#one-step-of-294912-tokens)列出了配方的选择依据。

<a id="one-step-of-294912-tokens"></a>
## 单次优化器更新：294,912 个 token

所有测试组都执行相同规模的一次优化器更新：共 64 个、每个 4608 token 的槽位，分配关系由 **GPU 数 × 微批次 `MAX_TOKENS` × `GRAD_ACC`** 决定。每个微批次都采用在显存中能够容纳的最低梯度检查点比例。衡量指标是整步耗时；实际 tokens/s 等于此步未经填充的图像与文本 token 总数除以耗时。

**测试设置。** 使用 4b1b9dc 代码树，并包含 0a943f9、2c5acba 的修改和 lycoris-lora 4.1.2（LoRA、LoKr 编译路径、全量微调）；LoKr 内核部分则使用更新至 1d2d0b9 的修改及 lycoris-lora 4.1.3 代码。torch 2.14.1+cu130、Triton 3.8.0；扩展性训练层、HF 布局的 bf16 权重；`BATCHING="tokens_ffd"`、`MAX_SAMPLES=0`；LoRA r32 `attn-mlp`；LoKr factor 8、`full_matrix=True`、DIM 10000、alpha 1。DDP 测量 9 步，单卡测量 5 步，取第 3 步至结束（不包含跟踪分析步骤）的中位数。

**列说明。** `s/step` 为各 rank 中最慢者的步骤耗时中位数；`real tok/s` 是所有 rank 实际图像与文本 token 总量除以步骤耗时；`fill` 为实际 token 数相对 294,912 预算的比例（约 1 MP 的图像含 4000～4100 个 token，无法填满所有容器）；`peak` 为 nvidia-smi 记录的各 rank 最大设备内存占用（GiB），可用容量为 31.36 GiB。

<a id="lora-r32"></a>
### LoRA r32

| GPU 数 | 微批次 × GRAD_ACC | 检查点比例 | 秒/步 | 实际 tok/s | 样本/s | 填充率 | 显存峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | **4608 × 16**（配方） | `partial:0.2` | **10.25** | **25,190** | 6.40 | 88.8% | 30.35 |
| 4 | 9216 × 8 | `partial:0.62` | 11.67 | 22,620 | 5.69 | 89.9% | 31.35 |
| 4 | 18432 × 4 | `partial:0.857` | 13.14 | 20,100 | 5.11 | 90.9% | 30.1 |
| 4 | 36864 × 2 | `partial:0.976` | 14.72 | 19,640 | 4.95 | 98.0% | 28.1 |
| 1 | **4608 × 64**（配方） | `partial:0.2` | **40.26** | **6,545** | 1.66 | 89.3% | 30.0 |
| 1 | 9216 × 32 | `partial:0.62` | 45.95 | 5,758 | 1.46 | 89.7% | 31.1 |
| 1 | 18432 × 16 | `partial:0.857` | 51.17 | 5,237 | 1.32 | 91.0% | 29.8 |
| 1 | 36864 × 8 | `partial:0.976` | 57.26 | 4,999 | 1.26 | 96.6% | 28.0 |

四卡上每个 4608-token 微批次耗时 0.640 秒，单卡为 0.629 秒，折算到每卡的效率达到 98%。将单步 token 总量扩大 2 倍或 4 倍，吞吐率仍基本不变：4608 × 32 可达到 25,030 实际 tok/s，4608 × 64 可达到 25,610。

<a id="lokr-factor-8"></a>
### LoKr factor 8

| GPU 数 | 路径 | 微批次 × GRAD_ACC | 检查点比例 | 秒/步 | 实际 tok/s | 样本/s | 显存峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | kernels | **4608 × 16**（配方） | `partial:0.262` | **12.59** | **20,570** | 5.22 | 29.31 |
| 4 | kernels | 4608 × 16 | `partial:0.214` | 12.57 | 19,680 | 5.00 | 30.36 |
| 4 | kernels | 9216 × 8 | `partial:0.69` | 15.24 | 16,450 | 4.14 | 31.25 |
| 4 | kernels | 18432 × 4 | `partial:0.881` | 16.12 | 16,570 | 4.21 | 30.56 |
| 4 | kernels | 36864 × 2 | `partial:0.976` | 17.45 | 16,560 | 4.17 | 29.81 |
| 4 | compile | 4608 × 16 | `partial:0.476` | 14.50 | 18,140 | 4.60 | 29.3 |
| 1 | compile | 4608 × 64 | `partial:0.476` | 56.42 | 4,664 | 1.18 | 29.55 |
| 1 | compile | 9216 × 32 | `partial:0.762` | 61.05 | 4,325 | 1.10 | 25.4 |
| 1 | compile | 18432 × 16 | `partial:0.857` | 显存不足 | | | |
| 1 | compile | 36864 × 8 | `partial:0.976` | 73.90 | 3,944 | 1.00 | 27.65 |

`kernels` 指 LyCORIS 4.1.3 的 LoKr 旁路（bypass）采用 token 优先 GEMM 布局的新内核；目前适配器预热会为 54 个 `lokr.bypass` 桶中的 50 个，以及 54 个 `lokr.bypass_with_base` 桶中的 51 个选择该路径。`compile` 是 4.1.2 为所有桶选用的编译后 torch 算子链。新内核在反向传播时只保存输入 `x`（与 LoRA 的 bypass 相同），因此四卡下 4608-token 微批次仅需重计算 11 个块，而编译路径需要 20 个。

相同微批次下 LoKr 的单步成本比 LoRA 高 23%（12.59 秒对比 10.25 秒）；旧编译路径则高 41%。

<a id="what-sets-the-speed"></a>
### 决定训练速度的因素

较小微批次之所以更快，仅仅是因为它能在显存中放下更多无需重计算的块；它采用的内核效率和较大微批次相同，token 容器的填充率也基本相同。

对于单卡 LoRA r32，在相同检查点比例 `partial:0.857`（42 个块中的 36 个需重计算）下：

| 微批次 token | 每微批次实际 token | 每实际 token 耗时 ms |
| --- | --- | --- |
| 4608 | 4080 | 0.190 |
| 9216 | 8362 | 0.189 |
| 18432 | 15915 | 0.194 |

一个约 4000-token 的微批次已经足以占满 GPU：不同批大小使用相同的 CUTLASS GEMM 和 Flash 内核；617 ms 的微批次中 GPU 仅空闲 27 ms。每个需要重计算的块相当于额外执行一次前向过程：实际 token 为 4080 时耗时约 5.5 ms，为 15915 时耗时约 14.6 ms。因此，显存允许的最低检查点比例决定了总步速：

| LoRA 与可容纳的检查点比例 | 每实际 token 耗时 ms |
| --- | --- |
| 4608，`partial:0.2`（8 个块） | 0.150 |
| 4608，`partial:0.62`（26 个块） | 0.175 |
| 18432，`partial:0.857`（36 个块） | 0.194 |
| 18432，`partial:0.905`（38 个块） | 0.197 |

追踪的一次 DDP 优化步骤（rank 0，共 294,912 token）的时间分解，单位 ms。NCCL 内核在等待最慢 rank 时也会持续轮询，因此 NCCL 耗时包含这部分等待时间：

| 实验组 | 前向 | 反向 | 重计算 | 优化器 | NCCL（暴露部分） | 空闲 |
| --- | --- | --- | --- | --- | --- | --- |
| LoRA 4608 × 16，`partial:0.2` | 3800 | 5281 | 722 | 2 | 419（146） | 240 |
| LoRA 9216 × 8，`partial:0.62` | 3841 | 5305 | 2379 | 2 | 559（62） | 49 |
| LoRA 18432 × 4，`partial:0.857` | 3741 | 5138 | 3196 | 2 | 2960（1365） | 1333 |
| LoRA 36864 × 2，`partial:0.976` | 4271 | 5903 | 4200 | 2 | 1385（159） | 79 |
| LoKr kernels 4608 × 16，`partial:0.262` | 4407 | 6640 | 1148 | 2 | 507（175） | 260 |
| LoKr kernels 18432 × 4，`partial:0.881` | 4503 | 6577 | 3929 | 2 | 2575（588） | 788 |
| LoKr compile 4608 × 16，`partial:0.476` | 4717 | 6873 | 2226 | 2 | 647（233） | 222 |

- 不同微批次的前向和反向耗时基本相同，增加的主要是重计算耗时。
- 当每个 rank 只有 4 个最多 18,432 token 的微批次时，各 rank 在同一步的 token 数量不一致；18,432-token 组合需要等待最慢 rank 约 0.3～1.4 秒。
- DDP 每次优化器更新只进行一次梯度求和。追踪显示，每一步都启动 13 次 fp32 all-reduce（25 MB 分桶、总计 283.5 MB LoRA 梯度）及一次 int64 归约；无论 `GRAD_ACC` 为 16、8 还是 4 均如此。优化器同样每步只执行一次，每种 `GRAD_ACC` 都是 35 个内核。
- 在相同检查点比例下，DDP 的显存峰值比单卡高 0.5～1.9 GiB（NCCL 缓冲区，以及四个 rank 中最不利的容器分配）。因此在显存余量不足时，DDP 必须采用比单卡更高的检查点比例。

<a id="full-finetune"></a>
### 全量微调

| GPU 数 | 微批次 × GRAD_ACC | 配置档 | 秒/步 | 实际 tok/s | 峰值 GiB |
| --- | --- | --- | --- | --- | --- |
| 1 | 36864 × 8 | `ckpt-ring8-actoff-optstream` | 117.7 | 2,432 | 22.3 |
| 1 | 36864 × 8 | 手动：`STREAM="swap"`、8 个常驻块、`ACT_OFFLOAD`、`OPTIM_STATE="stream"`、`partial:0.905` | 125.4 | 2,161 | 31.3 |
| 1 | 18432 × 16 | 上一配置去掉 `ACT_OFFLOAD`，`partial:0.81` | 显存不足 | | |
| 1 | 9216 × 32 | 上一配置去掉 `ACT_OFFLOAD`，`partial:0.62` | 显存不足 | | |

- 当 `GRAD_ACC=1` 时，一个 36,864-token 微批次耗时 8.8 秒；在累积 8 个微批次的步骤中则为 14.7 秒。两个微批次之间，每个流式块的梯度会被传回主机，并在那里累加到 fp32 梯度和中（每个微批次约传送 10 GB 梯度）。瓶颈是主机上的累加操作，而非显存容量。
- 把空闲显存用于减少重计算块反而更慢：启用 `ACT_OFFLOAD` 时，不做检查点的块所保存的全部激活都要经 PCIe 传输；关闭它又会使执行检查点的块的输入无法再从 GPU 卸载。

<a id="method"></a>
## 方法

**硬件与代码版本。** 四张 32 GB RTX 5090（575 W 功率上限），通过不支持 GPU 间 P2P 访问的 PCIe 连接；Linux、代码版本 947155d、torch 2.14.1+cu130、Triton 3.8.0。

**运行环境。** 在扩展性训练层（`configs/moss/scale/scale_lora_bf16.py`）上使用 HF 布局的 bf16 权重，通过 `scripts/train/moss.py` 运行：

- 数据：约 1 MP 图像（4096 个图像 token 加文本描述，每个样本大约 4000 个图文 token）、静态首次适配 token 分批（`BATCHING="tokens_ffd"`、`TOKEN_BUDGET="exact"`）、离线潜变量缓存与文本缓存；
- LoRA：维度 32、`attn-mlp`、bypass、AdamW；
- 全量微调：使用 fp32 主权重副本（master weights）的 AdamW，不启用 EMA；
- 内核：使用预热调优表的 `TUNE="table"`；除明确注明外，`COMPILE="off"`；
- LoRA 测试运行 80 步，全量微调运行 25～40 步；不包含性能分析器采集窗口。

配方基本沿用基础层的数据配置，但 LoRA 配方将 `TOKEN_BUDGET="render"`（从离线文本缓存读取各描述长度）。由于其完整批次内存估计仍预留 1.7 GiB，因此 `MAX_TOKENS` 需要约束每个批次的**实际** token 总数。配方还使用 `BASE_PRECISION="compute"`，使冻结的基础模型保持本测试中的 16 位精度；即使输入是默认 DiT 为 int8 的 ComfyUI 模型目录，也按此规则执行。

**指标定义。**

| 指标 | 含义 |
| --- | --- |
| tokens/s | 第 11 步至结束期间的总 token / 总耗时；剔除耗时超过中位步骤 3 倍的步骤（新批次形状引发的一次性重新编译/调优）；汇总所有 rank |
| window | 保留测量窗口中的所有步骤后的相同指标 |
| s/step | 测量窗口内每步耗时的中位数 |
| samples/s | 测量窗口样本总数 / 总耗时，按 rank 统计 |
| TF/s | 训练器的 `tflops` 指标，按每个 GPU 统计 |
| peak | torch 已分配显存峰值 / nvidia-smi 报告峰值，单位 GiB |
| full batch | 根据本轮每步峰值与 batch token 数的拟合，将已分配内存峰值外推到完整 `MAX_TOKENS` 批次，再加上分配器预留差额与 CUDA 上下文（GiB）；卡可用容量为 31.36 GiB |
| scaling | 四卡总 tokens/s 除以对应单卡每 rank 设置下 tokens/s 的四倍 |

<a id="lora-one-gpu"></a>
## LoRA：单 GPU

训练速度主要取决于反向传播时需要重新计算的块比例。能容纳一个样本的最小批大小（4608 token）能给无需重新计算的块留出最多显存。`partial:<f>` 表示对前 `f` 比例的块使用梯度检查点。

| 配置（`MAX_TOKENS=4608`，手动配置档） | tokens/s | window | 秒/步 | 样本/s | TF/s | peak GiB | full batch GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CHECKPOINT="none"` | 显存不足 | | | | | | |
| `CHECKPOINT="none"`，`COMPILE="block"` | 6781（23 步） | | 0.593 | 1.70 | 162 | 29.2 / 31.3 | 4.1k-token 批次时显存不足 |
| `partial:0.15` | 6479 | 5890 | 0.615 | 1.59 | 211 | 28.1 / 29.7 | 30.6 |
| **`partial:0.2`**（配方） | **6385** | 6189 | 0.628 | 1.67 | 221 | 27.2 / 28.7 | 29.6 |
| `partial:0.25` | 6287 | 5761 | 0.638 | 1.55 | 206 | 26.3 / 27.8 | 28.6 |
| `partial:0.3`，`BASE_PRECISION="fp8_compute"` | 显存不足 | | | | | | |
| `partial:0.4`，`fp8_compute`，`FP8_DGRAD="16bit"` | 5713 | 5103 | 0.671 | 1.38 | 182 | 24.3 / 25.7 | 25.7 |
| `partial:0.15`，`OPTIM_STATE="stream"`，`FUSED_STEP=True` | 5995 | 5764 | 0.654 | 1.55 | 206 | 27.7 / 29.2 | 30.0 |
| `partial:0.15`，`OPTIM_STATE="cpu"` | 5392 | 5246 | 0.719 | 1.41 | 188 | 27.6 / 29.2 | 30.0 |
| `partial:0.1`，`OPTIM_STATE="cpu"` | 5594 | 5393 | 0.713 | 1.45 | 193 | 28.6 / 30.1 | 31.0 |

- `partial:0.2` 是完整批次估算仍能留出 1.7 GiB 空间的最快配置；如果显卡还需要驱动显示器，`partial:0.25` 可预留 2.7 GiB。
- 相同检查点比例下，fp8 基础模型比 bf16 占用更多显存（`partial:0.3` 时 OOM）；fp8 GEMM 的性能收益也抵消不了额外的重计算。
- 将 AdamW 状态放在主机上可节约 0.6 GiB（LoRA 优化器状态共 0.9 GB），不足以减少一个 5% 的检查点比例阶梯，却会让每步增加 0.04～0.10 秒。因此所有优化器卸载配置都慢于将状态留在 GPU 上的 `partial:0.2`。

<a id="lora-four-gpus"></a>
## LoRA：四 GPU

`MAX_TOKENS` 表示**每 GPU、每微批次**的预算。tokens/s 是整个任务的吞吐量，“每 GPU”是总量除以四；加速比相对于单卡 `partial:0.2` 的四倍吞吐量（6385 × 4）。

| 配置（手动配置档） | MAX_TOKENS | tokens/s | 每 GPU | 秒/步 | TF/s | peak GiB | full batch GiB | scaling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DDP，`partial:0.2`**（配方） | 4608 | **23509** | 5877 | 0.646 | 205 | 27.3 / 29.2 | 30.0 | 0.92 |
| DDP，`partial:0.25` | 4608 | 23435 | 5859 | 0.655 | 181 | 26.4 / 28.3 | 29.0 | 0.92 |
| 原生 torch DDP（`DDP_BACKEND="torch"`），`partial:0.25` | 4608 | 23790 | 5948 | 0.656 | 200 | 26.4 / 28.4 | 29.0 | 0.93 |
| pipeline `1f1b`，32 微批次，`CHECKPOINT="none"` | 4608 | 19500 | 4875 | 6.49 | 111～122 | 25.7 / 27.7 | | 0.76 |
| pipeline `interleaved-1f1b`，32 微批次，`CHECKPOINT="none"` | 4608 | 显存不足 | | | | | | |
| pipeline `zbv`，16 微批次，`CHECKPOINT="none"` | 4608 | 显存不足 | | | | | | |
| FSDP2 ZeRO-3（`FSDP_RESHARD="forward"`），`CHECKPOINT="none"` | 4608 | 5627 | 1407 | 2.76 | 34 | 23.7 / 26.5 | 27.0 | 0.22 |
| FSDP2 ZeRO-3，`partial:0.4` | 9216 | 11426 | 2857 | 2.81 | 102 | 28.4 / 31.4 | 32.0 | 0.45 |
| FSDP2 ZeRO-2（`FSDP_RESHARD="backward"`），`partial:0.35` | 4608 | 8360 | 2090 | 1.84 | 75 | 26.2 / 28.6 | 29.3 | 0.33 |
| FSDP2 ZeRO-2，`partial:0.7` | 9216 | 13070 | 3268 | 2.45 | 117 | 26.8 / 29.8 | 29.8 | 0.51 |

- 梯度 all-reduce（每步 284 MB）的等待中位数为 11～12 ms。
- DDP 的步耗时中位数仅比单卡高 3%；其他扩展损失来自新批次形状引发的一次性重新编译，其他 rank 在此期间需要等待。三组 DDP 数据差异小于这种步间波动。
- FSDP2 对冻结基础模型分片，释放约 7 GiB/卡，允许无需重计算地运行块，但前向和反向都要通过主机内存重新 all-gather 每块的 16 位权重（GPU 间无 P2P）。4608 和 9216 token 下的微批次都耗时约 2.8 秒，说明瓶颈在权重收集。
- ZeRO-2 每微批次只进行一次权重收集（4608 token 时 1.84 秒，对比 ZeRO-3 的 2.76 秒），但反向期间必须一直保留收集后的完整基础模型，因此需要比 DDP 更高的检查点比例（4608 token 时为 `partial:0.35`），速度仍不及 DDP。
- 流水线方案无需梯度检查点（stage 0 峰值 25.7 GiB，stage 3 为 9.7 GiB），但每个微批次必须填充到固定的 4608-token 形状（实际约 4000 token），并且 35 个时间段中有 3 个气泡空档；每步的全局批量为 32 个样本，而 DDP 为 4 个。

<a id="full-finetune-one-gpu"></a>
## 全量微调：单 GPU

权重采用流式交换（`swap`），配合块级梯度检查点和激活卸载。fp32 主权重及 AdamW 动量驻留在主机，在反向传播期间流式进入优化器更新步骤。标记为 `lanes` 的测试与同一主机上的另外三个全量微调任务同时运行。

| 配置 | MAX_TOKENS | tokens/s | 秒/步 | 样本/s | TF/s | peak GiB | full batch GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`ckpt-ring8-actoff-optstream`**（配方，独立） | 36864 | **4088** | 8.76 | 1.08 | 195 | 18.3 / 22.1 | 22.3 |
| `ckpt-ring8-actoff-optstream`（lanes） | 36864 | 4052 | 8.81 | 1.07 | 193 | 18.3 / 22.1 | 22.3 |
| `ckpt-ring4-actoff-optstream`（lanes） | 36864 | 3964 | 8.99 | 1.05 | 189 | 17.4 / 21.3 | 21.4 |
| `ckpt-ring16-actoff-optstream`（lanes） | 36864 | 3937 | 8.98 | 1.04 | 188 | 19.9 / 23.8 | 23.9 |
| `ckpt-ring4-actoff-optstream`（lanes） | 55296 | 4082 | 13.06 | 1.07 | 195 | 21.1 / 25.9 | 26.0 |
| `ckpt-ring4-actoff-optcpu`（lanes） | 36864 | 2461 | 12.85 | 0.65 | 117 | 9.1 / 12.7 | 12.8 |
| `ckpt-resident-optstream`（lanes） | 9216 | 1896 | 3.88 | 0.49 | 91 | 27.5 / 31.3 | 31.5 |
| `ckpt-ring8-optstream`（lanes） | 36864 | 显存不足 | | | | | |
| `ckpt-resident-optstream`、`ckpt-resident-optcpu`（lanes） | 18432、36864 | 显存不足 | | | | | |
| 手动 `partial:0.3`、`OPTIM_STATE="stream"`、`GRAD_OFFLOAD=True`、`GRAD_ACC=8`（lanes） | 4608 | 显存不足 | | | | | |

- `swap` 行主要受计算能力限制；环大小及批大小使吞吐变化不到 4%。
- 常驻的全量微调把 fp32 主权重留在 GPU 上（构建后占 18.5 GiB），因此仅剩下小批次的空间。
- 配方运行使用了 116 GiB 主机内存。

<a id="full-finetune-four-gpus"></a>
## 全量微调：四 GPU

| 配置 | tokens/s | 秒/步 | 样本/s（所有 rank） | TF/s | peak GiB | scaling |
| --- | --- | --- | --- | --- | --- | --- |
| **DDP，`ckpt-ring8-actoff-optstream`，每 rank 36864**（配方） | **15271** | 9.25 | 3.92 | 184 | 18.6 / 24.0 | 0.93 |
| pipeline `1f1b`，`ckpt-resident`，16 × 9216 | 10.3k | 12.54 | 2.70 | 123 | 28.3 / 30.4 | |
| FSDP2 ZeRO-3，`ckpt-resident`，每 rank 18432 | 8306 | 7.83 | 2.23 | 99 | 27.6 / 31.4 | |
| FSDP2 ZeRO-3，手动 `partial:0.88`，每 rank 9216 | 4180 | 7.71 | 1.12 | 50 | 26.1 / 31.4 | |
| FSDP2 ZeRO-2，`ckpt-resident`，每 rank 9216 | 第 1 步显存不足 | | | | | |
| pipeline `interleaved-1f1b`，`ckpt-resident`，32 × 4608 | 第 2 步显存不足 | | | | | |

- DDP 的归约等待中位数为每步 6 ms；四个 rank 合计占用 71 GiB 主机内存。
- FSDP2 为每个微批次收集 fp32 主权重：每 rank 为 9216 或 18432 token 时，整步均耗时 7.7～7.8 秒，因此减小检查点比例也无益，而 18432 token 已经填满显存。ZeRO-2 必须同时保存收集后的完整 fp32 主权重（18.9 GiB）与分片，无法容纳。
- FSDP2 和流水线运行产生的全量微调检查点约 55 GB：通过主机内存进行权重收集，不会使 VRAM 超过训练步骤峰值。FSDP2（每 rank 18432）保存耗时约 2.5 分钟；pipeline（`1f1b`、16 × 9216）约 11.5 分钟。
