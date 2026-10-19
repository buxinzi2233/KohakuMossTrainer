[English](../../concepts/numerics.md)

<a id="numerics"></a>
# 数值计算

各部分使用什么数据类型运行、应用哪些 PyTorch 默认精度设置，以及 fp16 运行时
如何让残差流保持在数值范围内。

| 问题 | 章节 |
| --- | --- |
| 各部分使用什么数据类型运行？ | [数据类型](#dtypes)、[条件计算](#conditioning) |
| fp16 运行应选择哪种 `FP16_FIX`？ | [FP16_FIX 模式](#fp16_fix-modes) |
| 为什么 fp16 需要修复 | [为什么 fp16 需要修复](#why-fp16-needs-a-fix)、[折叠](#the-fold) |

<a id="dtypes"></a>
## 数据类型

- 主干使用计算数据类型（bf16 或 fp16）运行。
- `time_embedding`、`shared_block_modulation` 和 `video_head_modulation` 存储 fp32
  权重与偏置；RoPE 频带始终为 fp32。
- 即时执行算子至少以 fp32 累加，并只舍入一次。

<a id="conditioning"></a>
## 条件计算

- **线性层。** 四个条件线性层（`time_embedding.0`、`time_embedding.2`、
  `shared_block_modulation`、`video_head_modulation`）都是普通的 PyTorch 线性层：
  每层以其权重的数据类型接收输入（`KEEP_FP32` 对应 fp32），并像其他模块一样
  在调用方的 autocast 下运行，因此在训练和采样时，autocast 使用 16 位操作数运行它们，
  并返回 PyTorch 原本会返回的结果（采样器的 DiT 前向在训练的 autocast 下运行，
  `kohakumoss.precision.autocast_for`）。输出转换为 fp32，用于逐样本调制计算。
- **正弦嵌入。** `timestep_embedding` 使用官方 MOSS 公式 `[cos | sin](t f_k)`，
  其中 `t = 1000 sigma`、`f_k = 10000^(-k/128)`，以 fp32 计算。`t f_k` 可达到约 1e3 rad，
  此处 fp32 的 ulp 为 6e-5，因此在较大的 t 处，相位会带有最多约 1e-4 rad 的舍入误差。

<a id="pytorch-defaults"></a>
## PyTorch 默认设置

全程应用 PyTorch 2.14 的默认精度设置：

| 开关 | 默认值 | 效果 |
| --- | --- | --- |
| cuDNN fp32 卷积（`torch.backends.cudnn.allow_tf32`） | True | fp32 卷积在 sm80+ 上使用 TF32 |
| `allow_fp16_reduced_precision_reduction`, `allow_bf16_reduced_precision_reduction` | True | cuBLAS 可以使用 16 位精度归约 16 位 GEMM 的 split-K 部分结果 |

各部分的 autocast 与数据类型：

| 部分 | autocast | 数据类型 |
| --- | --- | --- |
| DiT，训练和采样器 | 内部 | 16 位 GEMM 输入 |
| VAE | 外部 | `VAE_DTYPE`，其 `auto` 在支持 bf16 的显卡（sm80+、Intel Arc）上使用 bf16，其他设备使用 fp16 或 fp32；按照上述默认设置，`VAE_DTYPE="fp32"` 卷积在 sm70 / sm75 上使用 IEEE fp32，在 sm80+ 上使用 TF32 |
| 文本编码器 | 外部 | `TEXT_ENCODE["dtype"]`：bf16 或 fp16 |

<a id="fp16_fix-modes"></a>
## FP16_FIX 模式

训练和采样脚本中的 `FP16_FIX` 接受运行时模式 `none`、`online` 和
`fp32_residual`。`weights` 模式是 `scripts/export/fp16_fix.py` 写入
fp16 修复检查点的内容；训练或采样该文件时使用 `FP16_FIX="none"`，
折叠信息从其元数据中读取。

| 模式 | 变化 | 是否支持逐块因子 |
| --- | --- | --- |
| `none` | 加载或运行时不做任何更改；已加载权重原本携带的折叠（fp16 修复检查点的 `kohakumoss_fp16_fix` 元数据）继续生效 | — |
| `weights`（fp16 修复检查点） | 将折叠写入权重：`video_embedder`、`text_embedder.proj`、文本块的 `proj` / `ffn_down`（权重与偏置）x c；`shared_block_modulation` 的 gate_a / gate_f 行 x c/s；每个块的 `o` / `ffn_down` x s | 否：所有块共享门，因此 c 和 s 必须一致 |
| `online` | 使用原始权重；在运行时应用相同因子：嵌入器和文本细化器的输出 x c_0，每个块的门 x c_l/s_l、写入器输入 x s_l，块之间的流按 c_l / c_{l-1} 重缩放，使用精确 eps | 是 |
| `fp32_residual` | 以 fp32 保存流，将归一化输出转换为计算数据类型 | — |

所有模式都接受 `clamp`（在 fp16 最大值处对联合流进行饱和截断，并使用设备端计数器
`clamp_counts(model)`），作为最后的补救手段。

<a id="the-fp16-fix-checkpoint"></a>
### fp16 修复检查点

`scripts/export/fp16_fix.py` 将 `weights` 模式张量写入普通的单文件检查点
（相同的键与形状，可由 ComfyUI 和 HF 代码加载），并将 c 和 s 记录到
safetensors 元数据（`kohakumoss_fp16_fix`）中。

- **其他运行时**保留原始 eps（1e-6），而不是使用 `1e-6 c^2`；输出只会有轻微差异，
  因为流的 rms 远高于 `sqrt(eps) / c`，而 `EPS_CHECK = True` 会在真实权重上测量该差异。
- **KohakuMoss** 读取元数据：将模型记录为已折叠状态（保存器和适配器导出会撤销折叠），
  并通过 `fp16_fix_eps="exact"` 切换到精确 eps。
- **量化基座**在由该文件构建时，会对它存储的数值做量化
  （[quant-kernels.md](../internals/quant-kernels.md#source)）。以
  `SAVE_DTYPE="fp16"` 写入的文件（随附文件）得到的是折叠后 bf16 权重的 fp16 转换，
  因此由它得到的 `int8` / `fp8` / `fp8_compute` / `mxfp8` 基座保存的是该转换后的编码；
  只有以 `SAVE_DTYPE="bf16"` 写入的文件才保留 bf16 数值（折叠因子是 2 的幂，在 bf16 中精确）。

<a id="adapters-and-gradients"></a>
### 适配器与梯度

在统一折叠下，`q`、`k`、`v`、`ffn_up_gate` 上的适配器（通过 RMSNorm 读取）
以及 `o`、`ffn_down` 上的适配器（通过门写入，s = 1）看到相同的输入并产生相同的输出，
因此 `weights` 和 `online` 给出相同的函数与相同的适配器梯度。

差异只会出现在重缩放写入器内部的可训练参数上（嵌入器、文本细化器输出、调制门的行、
s != 1 时的 `o` / `ffn_down`），以及全量微调中，此时 Adam 的更新相对于一个
缩小了 c（或 s）倍的权重进行。`online` 正是为这些情况而提供的：权重保留原本的尺度。

<a id="why-fp16-needs-a-fix"></a>
## 为什么 fp16 需要修复

fp16 探测（`scripts/probe/activations.py`，fp32 模型，提示词 x sigma x 分辨率）
测得联合残差流最高约为 6.2e4，与 fp16 最大值 65504 的差距不到 6 %，
而其他所有值都很小：

| 值 | 最大值 |
| --- | --- |
| 联合残差流 | ~6.2e4 |
| 注意力 logits | ~600 |
| `o` 输出 | ~1.3e3 |
| `ffn_down` 输出 | ~700 |
| `ffn_up_gate` | ~170 |
| 文本流 | ~1.8e3 |

唯一需要调整的就是这个流。

<a id="the-fold"></a>
## 折叠

残差流（联合流和文本流）的每个读取器都是 RMSNorm，除 eps 外，RMSNorm 具有尺度不变性：
使用 eps `c^2 eps` 的 `rms(c x)` 等于使用 eps 的 `rms(x)`。
因此，将流的每个写入器缩放 `c` 倍，并将每个读取归一化的 eps 缩放 `c^2` 倍，
可以在流缩小为 `c` 倍的同时保持函数不变。门前缩放 `s` 还会将幅度从块的
`o` / `ffn_down` 输出转移到门中（`o * s`、`gate / s`）。

<a id="fold-arithmetic"></a>
### 折叠运算

折叠以 fp32（若张量自身数据类型更宽，则使用其自身类型）对权重做乘法，并只舍入一次。
对于 2 的幂次形式的 c 和 s，乘积在所有数据类型中都是精确的，
唯有折叠后的 fp16 值落入次正规数范围时例外。

<a id="saturation-counters"></a>
## 饱和计数器

`SaturationCounters(model, threshold)` 为每个块安装钩子，在设备上维护运行至今的
最大 |x| 和超过 `threshold` 的值的数量；`read()` 只同步一次。
启用激活梯度检查点的块会在反向中再次运行其前向；钩子会跳过该重计算
（此时正在运行反向图任务），因此每次前向的每个值只计数一次。
