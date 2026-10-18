[English](../../internals/quant-kernels.md)

<a id="quantized-frozen-base-kernels"></a>
# 量化冻结基础权重内核

为以低于 16 位存储的冻结基础权重提供计算的 GEMM。分为两类：

* **仅权重量化**（`base_precision` / TE `precision` 为 `int8`、`fp8`）：权重量化存储，
  在 16 位 GEMM 内反量化（[仅权重量化](#weight-only)）；
* **fp8 计算**（`fp8_compute`、`mxfp8`）：权重量化一次后存储，GEMM 本身使用 e4m3
  操作数运行，激活每次调用时量化（[fp8 计算](#fp8-compute)）。

<a id="choosing-a-base-precision"></a>
## 选择基础精度

`BASE_PRECISION`（训练和采样中的 DiT）接受以下值；`TEXT_ENCODE["precision"]`
（文本编码器）接受 `compute`、`int8`、`fp8`、`fp8_compute` 和 `mxfp8`
（[text-encoder.md](text-encoder.md#quantization)）：

| 值 | 类别 | 架构 | 冻结基础权重每权重字节数 | 可训练基础权重 |
| --- | --- | --- | --- | --- |
| `compute` | 无（`PRECISION` dtype） | 全部 | 2（16 位） | 是 |
| `int8` | 仅权重量化 | 全部（CPU 上使用 eager） | 1 + 每行一个 fp32 缩放值 | 否 |
| `fp8` | 仅权重量化 | sm89+ | 1 + 每行一个 fp32 缩放值 | 否 |
| `quant` | 仅权重量化：架构支持 fp8 GEMM 时为 `fp8`，否则为 `int8` | 全部 | 1 + 每行一个 fp32 缩放值 | 否 |
| `fp8_compute` | fp8 计算，缩放粒度为 `FP8_RECIPE` | sm89+ | 约 1.0；以 `FP8_DGRAD="fp8"` 训练时增加 W^T 副本（[输入梯度](#input-gradient-training)） | 是，通过 fp8 副本（[全量微调](#full-finetune)） |
| `mxfp8` | fp8 计算，1 x 32 e8m0 缩放值 | sm89+（sm100+ 上使用原生块缩放 MMA） | 约 1.03；训练中有相同的 W^T 副本 | 是，通过 fp8 副本 |

在 Ada（sm89）上，采用 `row` 配方的 `fp8_compute` 是速度快的格式；该架构上的
`mxfp8` 保留其权重字节数优势，但消耗更多计算时间（[实测](#measured-rtx-5090-sm120-bf16-activations)）。

<a id="weight-only"></a>
## 仅权重量化

`kohakumoss_kernels.quant` 提供 `quant_linear(x, q, scale, bias, fmt)`，
`kohakumoss.quant.QuantWeight.linear` 为每个权重一次性解析（`bind`）该算子。

<a id="storage"></a>
### 存储

`q` 为 `[N, K]`（`int8`，或在 `fmt="fp8"` 时为 `float8_e4m3fn`），`scale` 是
逐行 fp32 向量 `[N]`，权重为 `q * scale[:, None]`。这是 `QuantWeight.from_weight`
生成的布局（int8：`round(w / s)`，`s = max|w_row| / 127`；
fp8：`s = max|w_row| / 448`）。

<a id="what-runs"></a>
### 运行内容

| 架构 | int8 | fp8 |
| --- | --- | --- |
| sm70 / sm75 | TileLang 仅权重量化 GEMM（fp16） | - |
| sm80 / sm86 | Triton 仅权重量化 GEMM（fp16 / bf16） | - |
| sm89 / sm120 | Triton 仅权重量化 GEMM（fp16 / bf16） | Triton |
| XPU | Triton 仅权重量化 GEMM（fp16 / bf16） | - |
| CPU | eager 参考实现 | eager 参考实现 |

* **前向** `y = (x @ q^T) * scale + bias`：int8 / e4m3 分块在寄存器或共享内存中
  转换为激活 dtype（两种格式的转换均精确），在 MMA 单元上乘法计算并使用 fp32 累加器，
  对累加器应用缩放和偏置，然后在存储时进行一次舍入。
* **输入梯度** `dx = dy @ (q * scale[:, None])`：缩放后的分块舍入一次到激活 dtype
  （与先反量化再矩阵乘法的参考实现使用的舍入相同），并以 fp32 累加。
* **无权重梯度**（基础权重被冻结）；可训练偏置使用 `dy` 的 fp32 列和。
* **为反向保存的内容**：仅有 `q` 和 `scale`。所有注册条目，包括 eager 条目
  （CPU、XPU、没有已构建内核的设备），都是同一个自定义算子（`make_quant_linear`）；
  eager 条目的启动器为参考反量化 + 矩阵乘法，因此其前向和输入梯度等于
  `quant_linear_reference`，而不会从前向到反向持续保留权重的 16 位副本。

<a id="selection"></a>
### 选择

* **候选。** 对于每个（设备、token 桶、N、K、dtypes、偏置），调优器对所有内核分块配置
  **以及**两条 PyTorch 路径（反量化 + `F.linear` eager，以及同一操作的 `torch.compile`
  版本）计时，保留最快者。因此，内核仅在该显卡上胜过两条 PyTorch 路径时运行；否则算子使用
  PyTorch 路径。
* **绑定。** 每个键只解析一次选择并保存在字典中，所以每次调用成本是一次查找。
  仅保留已确定的选择（`tuning.decided`）。
* **不计时的场合。** 不允许调优器计时的调用（追踪内部，或检查点重计算中的前向 GEMM）
  仅在该次调用中使用规划器的首个候选；后续允许计时的调用再测量该键。在调优窗口之外
  （[tuning.md](tuning.md#tuning-windows)），缺失键会不经计时绑定首个候选，并输出一行 WARNING。
* **反向。** dgrad GEMM 在 autograd 反向内部运行，自身不启动 autograd，因此在其中计时
  （`in_backward=True`）。

该算子是 `torch.library` 自定义算子（`kohakumoss::quant_fwd` /
`kohakumoss::quant_dgrad`），提供 fake 实现并注册 autograd，因此会保留在
`torch.compile(fullgraph=True)` 图中。

<a id="checks"></a>
### 检查

* `tests/kernels/quant/test_quant.py`：使用共享精度检查对照 fp64 参考检查前向、输入和偏置梯度
  （统计对比 PyTorch，见 [fp16.md](fp16.md#error-statistics)），非整分块形状、fullgraph 编译，
  以及检查必须拒绝的缩放轴错误变体。
* `scripts/bench/kernel/quant.py`：在 DiT 层形状上对比所有后端与 eager 和已编译的
  反量化 + 矩阵乘法的内核耗时，并给出调优器为每个案例选择的路径。

<a id="fp8-compute"></a>
## fp8 计算

`kohakumoss_kernels.fp8` 提供 `fp8_linear(x, q, scale, aux, q_t, scale_t, aux_t,
bias, recipe)`，`kohakumoss.quant.compute.ComputeWeight.linear` 按（设备、dtype）
一次性绑定该算子。设计遵循 KohakUwULLM 的 MXFP8 路径（其 5090 内核和测量结果是
sm120 参考）以及 OCP MX v1.0 规范。

<a id="knobs"></a>
### 控制项

| 位置 | 控制项 | 值 |
| --- | --- | --- |
| DiT（训练和采样中的 `BASE_PRECISION` / `FP8_RECIPE` / `FP8_DGRAD`、`MEMORY_OVERRIDES`；Python 中的 `memory_overrides`） | `base_precision` | `fp8_compute`、`mxfp8`（以及仅权重量化的 `int8`、`fp8`） |
| | `fp8_recipe` | `row`（默认）、`tensor`、`block`；仅限 `fp8_compute` |
| | `fp8_dgrad` | `fp8`（默认）、`16bit`；仅限训练 |
| 文本编码器（`TextEncoderPlan`，训练和采样中的 `TEXT_ENCODE`） | `precision` | `fp8_compute`、`mxfp8` |
| | `fp8_recipe` | `row`、`tensor`、`block` |

- 二者都需要 CUDA sm89+：否则内存计划和文本计划校验会指出架构
  （拒绝 sm70 / sm75 / sm86、XPU 和主机）。
- 基础权重可训练（全量微调）时，`base_precision` fp8 计算通过 fp8 副本训练 16 位权重
  （[全量微调](#full-finetune)），权重常驻（`stream="off"`）。

<a id="formats-and-recipes"></a>
### 格式与配方

操作数为 `[rows, K]`，其中 K 为收缩轴；值采用 e4m3（`float8_e4m3fn`，最大值 448）。
缩放粒度由配方决定：

| 配方 | 激活缩放 | 权重缩放 | 缩放类型 |
| --- | --- | --- | --- |
| `tensor` | 每张量一个 `[1]` | 每张量一个 `[1]` | fp32 |
| `row` | 每 token 一个 `[M]` | 每输出通道一个 `[N]` | fp32 |
| `block` | 沿 K 的 1 x 128 `[M, K/128]` | 128 x 128 块 `[ceil(N/128), K/128]` | fp32 |
| `mx`（`mxfp8`） | 沿 K 的 1 x 32 `[M, K/32]` | 沿 K 的 1 x 32 `[N, K/32]` | e8m0，带偏置的 uint8 |

* **fp32 缩放值**为 `max(amax, 1e-12) / 448` 和 `q = e4m3(clamp(x / s))`，
  两个除法均正确舍入。
  * PyTorch 通过倒数将张量除以 Python 标量（某些 amax 会偏差一个 ulp）；eager 参考实现
    除以张量，Triton 内核使用 `div_rn`，因此内核与参考实现的字节相等。
  * sm89 上 Triton 的 fp32 -> e4m3 转换经过 fp16（两次舍入，靠近舍入中点的值会产生不同
    字节）；内核先在 fp32 中舍入到 e4m3 网格（通过魔数加法实现最接近偶数舍入），使所有
    架构上的该转换都精确。
* **mx 指数向上舍入**：取满足 `2^e * 448 >= amax` 的最小 `e`，下限为 -127，
  由 amax 的位计算（`amax = m 2^E`：`e = E - 8`，当 `m > 1.75` 时为 `E - 7`），
  因此块的 amax 永不超过 448，也不会截断任何值。OCP 最接近舍入指数会截断块最大值，
  这是 NVIDIA 和 KohakUwULLM 测量为发散的失败情形。`q = e4m3(x * 2^-e)`
  （精确缩放）。
* **`aux`** 是 mx 权重缩放值的厂商布局副本：128 行 x 4 列分块，
  32 x 4 x 4 交错（`SWIZZLE_32_4_4`），补零到完整分块。
* **层尺寸：** fp8 计算层要求 N 和 K 为 128 的倍数；其他层（以及 KEEP_FP32 模块和
  特征数少于 256 的线性层）保持稠密，并计入量化报告（`skipped`）。

 eager 参考实现为 `kohakumoss_kernels.fp8.formats.quantize` / `dequantize`；
`tests/quant/test_fp8_formats.py` 用独立 numpy 公式（由位布局进行 e4m3 舍入，
逐块计算缩放值）验证其字节。

<a id="weights-and-activations"></a>
### 权重与激活

- **权重。** 应用内存计划时，`ComputeWeight.from_weight` 将权重量化一次（在计算设备上，
  随后放置），仅存储 `q`（每个权重 1 字节）、缩放值和 `aux`：每个权重约 1.0
  （tensor / row / block）或 1.03（mx）字节，而 bf16 为 2 字节。内存规划器计入这些字节
  （`PRECISION_BYTES`）、训练中的 W^T 副本，以及每次调用的 fp8 激活和 16 位回退缓冲区。
  使用权重流式传输（`stream="ring"`）时，环形缓冲区承载量化部件（`StreamedComputeWeight`）。
- **激活**在各 GEMM 路径内部即时量化（对 `x` 做一次受带宽限制的遍历，使用固定分块：
  token 数每步变化）；fp8 路径从不构建权重的 16 位副本。
- **量化器的位置。** 与 KohakUwULLM 的稠密路径相同，量化器位于消费端：同一线性层上的
  LoRA 本来就需要 16 位输入，而按形状选择的路径决定布局（自然缩放布局、厂商 swizzle，
  或在 GEMM 分块循环内量化）。遗留的成本是 q / k / v 投影将同一个归一化输入量化三次：
  5090 上 4k-16k 个 token 时，每次额外遍历耗时 0.07-0.14 ms，占 DiT 一步的 2-5%。

<a id="input-gradient-training"></a>
### 输入梯度（训练）

冻结基础权重只需要 `dx = dy W`：

| `fp8_dgrad` | 每权重字节数 | 输入梯度 GEMM |
| --- | --- | --- |
| `"fp8"`（默认） | 2 | 基础权重还存储沿 N 量化的 W^T（该 GEMM 的收缩轴，与 KohakUwULLM 一样：MX 块必须沿收缩轴排列，因此前向副本不能服务于输入梯度）；`dy` 按同一配方量化为 e4m3，GEMM 在 fp8 路径上运行 |
| `"16bit"` | 1 | 16 位的 `dy @ dequant(W)` |

没有权重梯度（适配器运算在 LyCORIS 中保持 16 位）；可训练偏置使用 `dy` 的 fp32
列和。激活梯度为直通式（量化器传递 `dy @ W`）。

<a id="full-finetune"></a>
### 全量微调

KohakUwULLM 以 MXFP8 训练完整模型，这里的全量微调也一样：

- 当基础权重可训练且 `base_precision` 为 fp8_compute / mxfp8 时，保留 16 位权重作为
  被训练参数，并给每个符合 fp8 计算形状的线性层附加 `ComputeCache`
  （前向 fp8 副本，以及 `fp8_dgrad="fp8"` 时的 W^T 副本）（`MossLinear.set_compute`）。
- 前向和输入梯度以 fp8 运行（`fp8_linear_train`）；权重梯度为 16 位操作数、fp32 累加的
  `dy^T x`（x 以自身 dtype 保存），与 KohakUwULLM 保持 WGRAD 为 16 位的做法相同。
- 每个优化器步骤后，从权重重建副本（`MemoryEngines.after_step`）；副本是模块状态之外的
  普通属性，因此检查点仅包含 16 位权重。
- 权重保持常驻（`stream="off"`）。每权重字节数：16 位权重加 1（`fp8_dgrad="16bit"`）
  或 2（`"fp8"`）。

<a id="pre-quantized-checkpoints"></a>
### 预量化检查点

`scripts/export/fp8_quant.py` 写出一个 transformer，其中 fp8 计算线性层存储为
`<n>.weight`（e4m3 q）+ `<n>.weight_scale`（配方的缩放值），其他所有张量为 16 位，
配方和层列表保存在 safetensors 元数据中（`kohakumoss_fp8`）。

- `load_backbone` 直接附加这些基础权重；不构建 16 位副本。在采样脚本中该文件是
  `TRANSFORMER`，在训练脚本中是单文件 `WEIGHTS`。
- 内存计划的量化跳过已经带有基础权重的层。
- 使用 `fp8_dgrad="fp8"` 的冻结基础权重训练，会从存储的前向副本推导 W^T 副本。
- mxfp8 的 MOSS-Image base：文件 5.33 GB；加载到 RTX 5090 用时 1.3 s，而 bf16 加载加
  量化为 7.5 s，常驻 5.34 GiB，前向结果逐位相同。

<a id="what-runs-1"></a>
### 运行内容

GEMM 形式为 `c = Q(a) B^T`，其中 B 为 `[N, K]` 且 K 连续（前向：`a = x`、
`B = W`；fp8 输入梯度：`a = dy`、`B = W^T`）。各配方的 GEMM 路径
（`kohakumoss_kernels.backends.triton.fp8.linear` 中调优器配置的 `gemm` 字段）：

| 配方 | 路径（按问题计时） |
| --- | --- |
| `tensor` | `vendor`（`torch._scaled_mm`，逐张量缩放）、`triton`（`scaled_mm`：每个 K 深度为 BK 的 fp8 MMA 结果加到独立 fp32 累加器，权重缩放在尾处理阶段应用） |
| `row` | `vendor`（`torch._scaled_mm`，逐行缩放）、`triton`、`fused`（一次启动：K 前处理取得每行 amax，主循环量化 A 分块） |
| `block` | `promote`（每个深度为 128 的 K 块的 MMA 结果缩放后加到 fp32 累加器，DeepGEMM 风格）、`fused` |
| `mx` | sm100+：`vendor`（`F.scaled_mm` BlockWise1x32，使用 swizzle 布局 e8m0 缩放）、`dot_scaled`（`tl.dot_scaled`，原生块缩放 MMA）、`fused`（分块循环中按每 32 列量化 A，`tl.dot_scaled`）；所有架构：`promote`（深度为 32 的块；sm100 以下在精确转换操作数的 fp16 MMA 上运行，`acc="upcast"`） |
| 所有配方 | `w16`（反量化权重，16 位矩阵乘法；eager 和 `torch.compile`） |

- **`fused`** 路径在每个程序中量化各自的 A 行（N 倍冗余），在 M 较小时胜出：一次启动
  而非两次，且无激活往返。
- **规划器的首选**（`TUNE="off"` 且无已存储选择时，或追踪时采用；
  `backends/triton/fp8/linear.py` 的 `candidates`）：

  | 配方 | 首选 |
  | --- | --- |
  | `tensor` | `vendor` |
  | `row` | sm90+ 为 `vendor`；sm89 为 `triton`，因为该架构上厂商逐行 GEMM 实测比 Triton 慢 2-3x（L4，见下文） |
  | `block` | `promote` |
  | `mx` | sm100+ 且 K 为 128 的倍数时为 `vendor`，否则为 `dot_scaled`；sm100 以下为 `promote` |
- **mx `vendor` 和 `fused`** 读取完整的 128 列分块（swizzle 布局激活缩放值、融合 K 循环），
  因此规划器仅在收缩长度为 128 的倍数时提供它们；K 仅为 32 倍数的 mx 问题只运行
  `dot_scaled`、`promote` 或 `w16`。
- **`block`** 运行于 Triton 路径：PyTorch 中厂商块（1x128 / 128x128）GEMM 仅在 sm90 存在。
- **sm89** 没有块缩放 MMA：`mx` 在该架构上的 fp16 MMA 上运行 `promote` 路径
  （见[精度](#precision)）。
- **激活量化器：** 自然布局中按行 / 按张量 / 沿 K 每 128 / 每 32 个元素量化，以及每 32
  个元素量化并将缩放值直接写入厂商 swizzle 布局（厂商 mx 路径）。

<a id="selection-1"></a>
### 选择

- 对每个（设备、token 桶、N、K、dtype、配方、偏置、dgrad 副本），调优器在真实调用上
  对每条路径（Triton 路径的每个分块配置）计时，保留最快者；无法在该环境中构建或运行的
  路径（不支持的厂商 GEMM、超出共享内存的分块）会被跳过。
- 与仅权重量化算子完全相同，选择只有确定后（`tuning.decided`）才会绑定，输入梯度 GEMM
  在反向内部计时。
- `w16` 路径在计时候选短名单中，因此 fp8 GEMM 仅在该显卡上胜过 16 位路径时运行。
  选择 `w16` 时，该层计算 `x @ dequant(W)^T`（权重仍量化存储，激活不量化：同一函数的
  更高精度版本）。
- `TUNE="off"` 使用该显卡表中的已存储选择，否则使用规划器首选（上表）。

绑定选择之前，会用该调用自身的输入在 no_grad 下运行一次：

- 无法在该环境运行的配置（构建或资源错误、不支持功能的 RuntimeError，例如某些 sm89
  构建上的逐行 `_scaled_mm`）会从该（设备、dtype、方向、配方）中移除，记录一次日志，
  然后采用下一个选择；
- 失败的 Triton 分块只移除该分块；
- 无剩余路径时，算子抛出 `BackendUnavailable`，注册表将 `fp8_linear` 重新绑定到
  eager 参考实现（`tests/kernels/quant/test_fp8_probe.py`）。

该算子为 `torch.library` 自定义算子 `kohakumoss::fp8_linear`（输入梯度为
`kohakumoss::fp8_dgrad`），提供 fake 实现并注册 autograd，因此会保留在
`torch.compile(fullgraph=True)` 图中。

<a id="measured-rtx-5090-sm120-bf16-activations"></a>
### 实测（RTX 5090、sm120、bf16 激活）

DiT 形状下的一个线性层，每次调用 ms，包含激活量化器（每条路径的最佳分块；
`scripts/probe/fp8_inference.py` 和调优表）：

| M x N x K | bf16 `F.linear` | row vendor | row triton | mx vendor | mx dot_scaled | block promote | tensor vendor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4200 x 3072 x 3072 | 0.347 | 0.293 | 0.328 | 0.243 | 0.281 | 0.535 | 0.342 |
| 4200 x 16384 x 3072 | 1.718 | 0.961 | 1.033 | 0.760 | 0.930 | 2.027 | 0.970 |
| 4200 x 3072 x 8192 | 0.867 | 0.579 | 0.683 | 0.430 | 0.599 | 1.190 | 0.558 |
| 16600 x 3072 x 3072 | 1.334 | 0.789 | 0.845 | 0.592 | 0.737 | 1.543 | 0.784 |
| 16600 x 16384 x 3072 | 7.077 | 3.457 | 3.659 | 2.478 | 3.293 | 7.582 | 2.495 |
| 16600 x 3072 x 8192 | 3.521 | 2.094 | 2.246 | 1.478 | 2.062 | 3.928 | 1.821 |

- 厂商 MX GEMM 在大形状上达到 bf16 的 2.4-2.9x。
- Triton block promote 路径接近 bf16（每步 128 K 的逐块累加器缩放），因此 `block`
  仅在 `fused` 胜出时获益。
- 在这些形状上，激活量化器本身耗时 0.07-0.14 ms（K = 8192、M = 16600 时为
  0.30-0.43 ms）；q / k / v 投影各自量化同一个归一化输入。

NVIDIA L4（sm89、72 W；`scripts/bench/kernel/fp8.py`），bf16，所选路径相对 16 位的
前向 / 输入梯度加速比：

| 案例 | M | row | tensor | block |
| --- | --- | --- | --- | --- |
| 4096 x 3072 x 3072 | 4096 | 2.13 / 1.95 (triton) | 1.91 / 1.65 | 1.79 / 1.63 |
| 4096 x 16384 x 3072 | 4096 | 1.76 / 1.42 | 1.72 / 1.23 | 1.45 / 1.38 |
| 16384 x 3072 x 8192 | 16384 | 1.65 / 1.95 | 1.25 / 1.78 | 1.40 / 1.61 |
| 256 x 3072 x 3072 | 256 | 0.67 / 0.68 | 0.48 / 0.50 | 0.65 / 0.65 |

sm89 没有块缩放 MMA，且其 fp8 MMA 累加过于粗糙，无法达到 MXFP8 参考实现的误差统计，
因此 `mx` 在该架构上的 fp16 MMA 上运行 promote 路径，比 bf16 `F.linear` 更慢。
RTX 4090，bf16，M = 4096 时的前向 ms（最佳 promote 分块）：

| N x K | bf16 | promote | `w16` |
| --- | --- | --- | --- |
| 3072 x 3072 | 0.49 | 0.57 | 0.61 |
| 16384 x 3072 | 2.58 | 2.93 | 3.55 |

在 Ada 上，`mxfp8` 保留其权重字节数优势，但消耗更多计算时间；`row`（Triton）是适合
Ada 的格式。

RTX 5090 上的端到端结果：

| 工作负载 | bf16 | fp8_compute row | tensor | block | mxfp8 |
| --- | --- | --- | --- | --- | --- |
| DiT，1024^2，20 步，2 条提示词，CFG 打包（每步秒数） | 0.875 | 0.567 | 0.570 | 0.638 | 0.483 |
| DiT 权重（GiB） | 9.66 | 5.01 | | | 5.33 |
| 文本编码器（Qwen3.5-9B），每轮 32 条约 150 词的提示词（ms） | 445 | 265 | | | 222 |

5 条短提示词时，文本编码器每次调用都受启动开销限制，fp8 比 bf16 慢
（`scripts/probe/fp8_inference.py` 的编码时间：
[text-encoder.md](text-encoder.md#quantization)）。

LoRA 训练（r16、attn-mlp、1 MP、MAX_TOKENS 8192、块级梯度检查点，文本编码器以同样
精度流式传输并在线编码，30 步，取第 8 步起的中位数；每次运行都将 `gated_residual`
固定为 eager）：

| 基础权重 / TE | 秒 / 步 | 样本 / 秒 | TE 秒 | DiT 秒 | 峰值 GiB（alloc） | 常驻 GiB（alloc） |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 1.478 | 0.68 | 0.542 | 0.938 | 13.60 | 10.23 |
| fp8_compute row | 1.087 | 0.95 | 0.291 | 0.826 | 13.62 | 10.24 |
| mxfp8 | 1.074 | 0.94 | 0.313 | 0.761 | 14.26 | 10.89 |
| mxfp8，`fp8_dgrad="16bit"` | 1.154 | 0.88 | 0.311 | 0.843 | 9.28 | 5.90 |

- 损失与 bf16 一致变化（第 30 步：0.6125 / 0.6196 / 0.6212 / 0.6206）。
- 使用 fp8 输入梯度时，基础权重和 bf16 一样每权重占 2 字节；`16bit` 保持 1 字节，
  节省 4.3 GiB。
- 每步 512^2 / 2048 个 token 时，每个 fp8 调用都受启动开销限制，DiT 部分比 bf16 慢
  （0.37 s -> 0.44-0.46 s），而文本编码器耗时仍减半。

<a id="precision"></a>
### 精度

**参考实现。** fp8 计算层的 fp64 参考是使用其量化权重的 fp64 乘积（激活不量化）；
相同精度的 PyTorch 为量化、反量化、`F.linear`（`fwd_reference` / `dgrad_reference`；
`w16` 路径为 `F.linear(x, dequant(W))`）。

**内核内部**，MMA 接受 e4m3（或精确转换的 fp16）操作数，以 fp32 累加，将缩放和偏置
应用到 fp32 累加器，并在存储时舍入一次到输出张量的 dtype；`vendor` 和 `w16` 路径
使用普通 PyTorch 调用。

**累加。** Triton 和融合的 row / tensor GEMM 将每个 K 深度为 BK（64-256）的 fp8 MMA
结果加到独立 fp32 累加器（`acc="promote"`），而不是沿 K 串接 MMA 自身的累加器：

- sm89 上，串接的 fp8 MMA 累加器精度随 K 增大而下降（K = 3072-8192 时，误差为相同
  操作数下 PyTorch 的 3-6x）；
- sm120 上，两者完全相同且成本相同；
- block 和 mx promote 路径在设计上就会将每个 K 块缩放为 fp32。

**Ada 的 fp8 MMA** 即使加到零累加器，其精度也低于 fp32：一块深度为 32 的 e4m3
乘积相对精确乘积的平均相对误差为 1.7-2.0e-4（相同操作数的 fp16 MMA：约 1e-6，
fp32 舍入；sm120 上的 fp8 MMA 与其一致），无偏，并在行、列和 K 块上均匀分布。

- **sm89 上的 mx。** mx 缩放值是 2 的幂，因此 PyTorch 的反量化 + `F.linear` 使用
  fp32 累加对精确操作数相乘，只对输出舍入。在 fp8 MMA 上，mx promote 路径在该舍入
  之下留下约 1e-4 的绝对误差下限，会在接近零的输出上显现。因此，sm100 以下的 mx
  promote 路径对精确转换的操作数运行 fp16 MMA（`acc="upcast"`；e4m3 -> fp16 精确）。

  | M x N x K = 2048 x 512 x 8192，fp16，对比 fp64 | abs_mean | rel_mean | rel_max |
  | --- | --- | --- | --- |
  | fp8 MMA 上的 mx promote | 2.72e-4 | 5.83e-4 | 27.9 |
  | PyTorch | 2.54e-4 | 1.77e-4 | 0.091 |
  | fp16 MMA 上的 mx promote（`acc="upcast"`） | 2.54e-4 | 1.76e-4 | 0.028 |

- **sm89 上的 tensor / row / block** 保留 fp8 MMA：其 fp32 缩放值使 PyTorch 反量化的
  16 位操作数发生舍入，损失比 fp8 MMA 更大（row、fp16：Triton abs_mean 2.93e-4，
  PyTorch 为 4.96e-4）。

**检查。** `tests/kernels/quant/test_fp8_accumulation.py` 要求每条计时路径在 K = 8192
时，平均绝对 / 相对误差不差于相同量化操作数下的普通 PyTorch（反量化 + F.linear，
以及设备支持时以激活 dtype 输出的 PyTorch fp8 GEMM `torch._scaled_mm` /
`F.scaled_mm`）。每 128 K 舍入一次的累加器无法通过，sm89 上串接的 Triton 路径和
fp8 MMA 上的 mx promote 路径也无法通过；规划器仅向 sm89 提供 fp16-MMA mx promote
分块。每条路径都以 bf16 和 fp16，在非整分块形状上检查前向和输入梯度
（`tests/kernels/quant/test_fp8.py`）。

<a id="checks-1"></a>
### 检查

* `tests/quant/test_fp8_formats.py`：按配方和角色对照独立 numpy 公式检查 eager 参考实现的
  字节和缩放值、缩放布局、对照逐元素乘积检查反量化、对照索引公式检查厂商 swizzle；
  采用最接近舍入的 mx 指数、轴错误的逐行缩放和行主序 swizzle 必须失败。
* `tests/kernels/quant/test_fp8.py`：Triton 量化器与 eager 参考实现逐位相等（包含跨步输入）、
  swizzle 量化器、每条路径的前向 + 输入梯度、已注册算子（fp8 和 16 位 dgrad、偏置梯度）、
  fullgraph 编译、架构检查、在 K 分块不完整时 mx 规划器排除整分块路径；行错误的权重
  缩放必须无法通过误差统计。
* `tests/kernels/quant/test_fp8_probe.py`：无法运行的路径会被探测、移除并记录一次日志；
  无剩余路径时回退到 eager 后端。
* `tests/quant/test_fp8_compute.py`：按架构校验文本编码器和内存计划，以及采样控制项到
  内存计划的映射。
* `scripts/probe/fp8_inference.py`：真实权重：各精度的文本编码器嵌入和 DiT 速度场，
  相对于 bf16 权重的主机 fp64 前向并列比较 bf16，编码 / 前向 / 生成时间、权重和峰值
  显存、解码图像。
