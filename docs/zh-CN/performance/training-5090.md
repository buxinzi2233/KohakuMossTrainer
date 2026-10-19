[English](../../performance/training-5090.md)

<a id="training-speed-on-the-rtx-5090"></a>
# RTX 5090 上的训练速度

一块和四块 RTX 5090 32 GB 上 LoRA 与全量微调的每秒 token 数，5090 配方背后的各项
设置，以及每个设置所对比的邻近设置。

| 配方 | 显卡数 | 设置 |
| --- | --- | --- |
| `recipes_lora_5090.py` | 1 | LoRA r32 bf16，`MAX_TOKENS=4608`，`CHECKPOINT="partial:0.2"` |
| `recipes_lora_ddp_4x5090.py` | 4 | 每个 rank 相同，`PARALLEL="ddp"` |
| `recipes_full_5090.py` | 1 | 全量微调，`ckpt-ring8-actoff-optstream`，`MAX_TOKENS=36864` |
| `recipes_full_ddp_4x5090.py` | 4 | 每个 rank 相同，`PARALLEL="ddp"`（主副本在主机上共享一份副本） |

<a id="method"></a>
## 方法

**硬件与工作树。** 四块 RTX 5090 32 GB（575 W 功耗上限），PCIe 且无 peer-to-peer，
Linux，工作树 947155d，torch 2.14.1+cu130，Triton 3.8.0。

**运行。** 在扩展性层（`configs/moss/scale/scale_lora_bf16.py`）上用 HF 布局的 bf16
权重运行 `scripts/train/moss.py`：

- 数据：1 MP 图像（4096 个图像 token 加描述文本，每个样本约 4000 个联合 token），
  静态首次适应 token 批次（`BATCHING="tokens_ffd"`，`TOKEN_BUDGET="exact"`），离线
  潜变量与文本缓存；
- LoRA：dim 32，`attn-mlp`，旁路，AdamW；
- 全量微调：使用 fp32 主副本的 AdamW，无 EMA；
- 内核：`TUNE="table"` 配合已预热的表，除某行另有说明外 `COMPILE="off"`；
- LoRA 为 80 步，全量微调为 25 到 40 步；不设 profiler 窗口。

这些配方沿用基座层的数据配置项，只是 LoRA 配方设置了 `TOKEN_BUDGET="render"`
（每个描述文本的长度取自离线文本缓存）：它们对整批的估计会留下 1.7 GiB 空闲，因此
`MAX_TOKENS` 必须约束每个批次的真实 token 数。它们还设置了
`BASE_PRECISION="compute"`，因此这里测得的冻结基座为 16 位，在 ComfyUI 文件夹上也是
如此，后者默认的 DiT 文件是 int8 的那个。

**数值。**

| 列 | 含义 |
| --- | --- |
| tokens/s | 第 11 步到最后一步的 token 数除以其耗时，排除耗时超过中位步骤 3x 的步骤（新批次形状上的一次性重编译和重调优）；跨 rank 求和 |
| window | 同上，但包含该窗口的每一步 |
| s/step | 该窗口的步骤时间中位数 |
| samples/s | 该窗口的样本数除以其耗时，按 rank |
| TF/s | 训练器的 `tflops` 指标，按 GPU |
| peak | torch 已分配峰值 / nvidia-smi 峰值，GiB |
| full batch | 把已分配峰值外推到 `MAX_TOKENS` 批次（每次运行对步骤峰值与批次 token 数做拟合）再加上分配器的预留缺口和 CUDA 上下文，GiB；显卡容量为 31.36 GiB |
| scaling | 四卡 tokens/s 除以同一每 rank 设置下单卡 tokens/s 的 4 倍 |

<a id="lora-one-gpu"></a>
## LoRA，一块显卡

速度取决于反向传播中重计算的块占比：能容纳一个样本的最小批次（4608 个 token）为不做
梯度检查点的块留出最多内存。`partial:<f>` 对前 `f` 比例的块做梯度检查点。

| 设置（`MAX_TOKENS=4608`，手动配置档） | tokens/s | window | s/step | samples/s | TF/s | peak | full batch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CHECKPOINT="none"` | 内存不足 | | | | | | |
| `CHECKPOINT="none"`, `COMPILE="block"` | 6781（23 步） | | 0.593 | 1.70 | 162 | 29.2 / 31.3 | 4.1k token 的批次下内存不足 |
| `partial:0.15` | 6479 | 5890 | 0.615 | 1.59 | 211 | 28.1 / 29.7 | 30.6 |
| **`partial:0.2`**（配方） | **6385** | 6189 | 0.628 | 1.67 | 221 | 27.2 / 28.7 | 29.6 |
| `partial:0.25` | 6287 | 5761 | 0.638 | 1.55 | 206 | 26.3 / 27.8 | 28.6 |
| `partial:0.3`, `BASE_PRECISION="fp8_compute"` | 内存不足 | | | | | | |
| `partial:0.4`, `fp8_compute`, `FP8_DGRAD="16bit"` | 5713 | 5103 | 0.671 | 1.38 | 182 | 24.3 / 25.7 | 25.7 |

- `partial:0.2` 是整批估计仍能留下 1.7 GiB 空闲的最快设置；`partial:0.25` 为还要驱动
  显示器的显卡留下 2.7 GiB。
- 在相同的梯度检查点占比下，fp8 基座比 bf16 基座需要更多内存（`partial:0.3` 内存
  不足），而它的 fp8 GEMM 也弥补不了多出的重计算。

<a id="lora-four-gpus"></a>
## LoRA，四块显卡

| 设置（每个 rank：`MAX_TOKENS=4608`，手动配置档） | tokens/s | window | s/step | samples/s | TF/s | peak | full batch | scaling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DDP，`partial:0.2`**（配方） | **23509** | 22827 | 0.646 | 1.47 | 205 | 27.3 / 29.2 | 30.0 | 0.92 |
| DDP，`partial:0.25` | 23435 | 20239 | 0.655 | 1.31 | 181 | 26.4 / 28.3 | 29.0 | 0.93 |
| 原版 torch DDP（`DDP_BACKEND="torch"`），`partial:0.25` | 23790 | 22361 | 0.656 | 1.44 | 200 | 26.4 / 28.4 | 29.0 | 0.95 |
| pipeline `zbv`，16 个微批次，`CHECKPOINT="none"` | 内存不足 | | | | | | | |

- 梯度 all-reduce（每步 284 MB）每步等待 11-12 ms（中位数）。
- DDP 的步骤时间中位数比单卡步骤长 3 %；其余扩展损失来自新批次形状的一次性重编译，
  期间其他 rank 都在等待。三行 DDP 之间的差异小于这种步骤间波动。

<a id="full-finetune-one-gpu"></a>
## 全量微调，一块显卡

权重以 `swap` 流式传输，配合块级梯度检查点和激活卸载；fp32 主副本和 AdamW 动量驻留在
主机上，在反向传播内部流经优化器步骤。标有 "lanes" 的行是与同一主机上另外三个全量
微调任务同时运行的。

| 设置 | MAX_TOKENS | tokens/s | s/step | samples/s | TF/s | peak | full batch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`ckpt-ring8-actoff-optstream`**（配方，单独运行） | 36864 | **4088** | 8.76 | 1.08 | 195 | 18.3 / 22.1 | 22.3 |
| `ckpt-ring8-actoff-optstream`（lanes） | 36864 | 4052 | 8.81 | 1.07 | 193 | 18.3 / 22.1 | 22.3 |
| `ckpt-ring4-actoff-optstream`（lanes） | 36864 | 3964 | 8.99 | 1.05 | 189 | 17.4 / 21.3 | 21.4 |
| `ckpt-ring16-actoff-optstream`（lanes） | 36864 | 3937 | 8.98 | 1.04 | 188 | 19.9 / 23.8 | 23.9 |
| `ckpt-ring4-actoff-optstream`（lanes） | 55296 | 4082 | 13.06 | 1.07 | 195 | 21.1 / 25.9 | 26.0 |
| `ckpt-ring4-actoff-optcpu`（lanes） | 36864 | 2461 | 12.85 | 0.65 | 117 | 9.1 / 12.7 | 12.8 |
| `ckpt-resident-optstream`（lanes） | 9216 | 1896 | 3.88 | 0.49 | 91 | 27.5 / 31.3 | 31.5 |
| `ckpt-ring8-optstream`（lanes） | 36864 | 内存不足 | | | | | |
| `ckpt-resident-optstream`, `ckpt-resident-optcpu`（lanes） | 18432, 36864 | 内存不足 | | | | | |
| 手动 `partial:0.3`，`OPTIM_STATE="stream"`，`GRAD_OFFLOAD=True`，`GRAD_ACC=8`（lanes） | 4608 | 内存不足 | | | | | |

- swap 各行受计算限制：环形缓冲区大小和批次大小对速度的影响都小于 4 %。
- 常驻的全量微调把 fp32 主副本留在 GPU 上（构建后为 18.5 GiB），只够容纳小批次。
- 配方运行使用了 116 GiB 主机内存。

<a id="full-finetune-four-gpus"></a>
## 全量微调，四块显卡

| 设置 | tokens/s | s/step | samples/s（所有 rank） | TF/s | peak | scaling |
| --- | --- | --- | --- | --- | --- | --- |
| **DDP，`ckpt-ring8-actoff-optstream`，每个 rank 36864**（配方） | **15271** | 9.25 | 3.92 | 184 | 18.6 / 24.0 | 0.93 |
| pipeline `1f1b`，`ckpt-resident`，16 x 9216 | 10.3k | 12.54 | 2.70 | 123 | 28.3 / 30.4 | |
| FSDP2，`ckpt-resident`，每个 rank 18432 | 8306 | 7.83 | 2.23 | 99 | 27.6 / 31.4 | |
| pipeline `interleaved-1f1b`，`ckpt-resident`，32 x 4608 | 第 2 步内存不足 | | | | | |

- DDP 每步等待 6 ms 完成其归约（中位数）；四个 rank 共使用 71 GiB 主机内存。
- pipeline 和 FSDP2 各行都完成了每一步训练，随后在运行结束时的检查点处内存不足，
  该检查点会把全量微调状态收集到 GPU 上。
