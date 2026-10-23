[English](../../internals/text-encoder.md)

<a id="text-encoder-kernels-batching-placement"></a>
# 文本编码器：内核、批处理与放置

MOSS-Image 以 Qwen3.5-9B 的最后一层隐藏状态作为条件（32 个解码器层：24 个
Gated DeltaNet 层、8 个因果 GQA 全注意力层，隐藏维度 4096）。编码器被冻结；
这里的所有操作都在 `torch.no_grad` 下仅执行前向。

本页介绍编码器放置与批处理的控制项、前向及其布局、提前编码、量化、报告，
然后介绍文本算子及其内核。

<a id="textencoderplan"></a>
## TextEncoderPlan

一套控制项（`kohakumoss.types.plans.TextEncoderPlan`）用于：

- 训练：`TEXT_ENCODE`，另用 `TEXT_DEVICE` 指定设备；
- 流水线：`text_plan=`，以及 `scripts/sample/sample.py` 中的 `TEXT_ENCODE`；
- 离线缓存工厂：`kohakumoss.encoders.text.text_encoder`。

| 字段 | 取值（默认值在前） | 作用 |
| --- | --- | --- |
| `memory` | `resident`、`offload`、`ring`（也接受 `stream`，视为同一值）、`cpu`、`auto` | 放置方式（[offload.md](offload.md#text-encoder)）；数据类默认值是 `resident`（`kohakumoss.types.plans`），训练和采样会将未设置的 `memory` 解析为 `auto` |
| `free_after_use` | `False`、`True` | 首次执行后丢弃模型 |
| `precision` | `auto`、`compute`、`int8`、`fp8`、`fp8_compute`、`mxfp8`、`w4a8` | 权重存储与 GEMM（[量化](#quantization)）；`auto` 表示权重所存储的格式（[ComfyUI 文件](#comfyui-files)），`w4a8` 仅用于 ComfyUI W4A8 文件 |
| `w4a8` | `auto`、`int8`、`packed` | W4A8 文件的解码方式（[ComfyUI 文件](#comfyui-files)） |
| `quant_group` | `0`（每个输出通道），或任意 16 的倍数（例如 `64`、`128`） | 仅权重量化时输入列的缩放分组；仅用于 `int8` / `fp8` |
| `fp8_recipe` | `row`、`tensor`、`block` | `fp8_compute` 缩放方式 |
| `stored` | `""`、文件 | 预量化的文本编码器（[见下文](#pre-quantized-text-encoder)） |
| `dtype` | `auto`（默认 bf16；如果接收嵌入的设备或编码器计算所在显卡不支持 bf16，则使用 fp16。主机执行时遵循接收嵌入的目标设备 dtype，使缓存键不受编码位置影响）、`bf16`、`fp16` | 计算 dtype |
| `stream_resident` | `2` | `ring` 下每种层类型的环形槽位数 |
| `pinned` | `True`、`False` | 用于卸载 / 环形流式传输的主机锁页副本 |
| `host` | `auto`、`copy`、`file` | 卸载 / 环形流式传输保留在主机上的内容的形态：计算 dtype 的副本，或检查点的文件页；`auto` 在放得下主机余量时使用副本（[offload.md](offload.md#host-room)） |
| `batch` | `32` | 每次内核调用的提示词数 |
| `max_tokens` | `16384` | 每次内核调用的 token 数 |
| `layout` | `packed`、`padded`、`single` | [前向与布局](#forward-and-layouts) |
| `forward` | `kernels`、`hf` | 内核前向或 transformers 模块自身的前向（仅限 `single` / `padded`） |
| `ahead` | `1` | 在线文本缓存每轮编码的主机批次数（[提前编码](#encode-ahead)） |
| `ahead_device` | `host`、`device` | 编码结果等待的位置 |
| `kernels` | `{}` | 文本算子 -> 后端，例如 `{"gated_delta_rule_varlen": "triton"}` |

训练时，`memory="cpu"`（或 `TEXT_DEVICE="cpu"`）支持**所有** `TEXT_CACHE` 模式。在 `online` / `off` 模式下，每次提前编码（encode-ahead）均在主机执行；第一轮编码会通过一条 INFO 日志显示耗时，以及摊到每步的成本。

<a id="streamed-load"></a>
## 流式加载

`QwenTextEncoder.from_pretrained`（`kohakumoss.encoders.text.load`）从不在主机内存中
持有模型的稠密副本：

1. 文本模型根据其配置构建，参数位于 meta 设备上（`accelerate.init_empty_weights`；
   旋转缓冲区在主机上构建）。检查点的张量按文本模型的名称读取为内存映射 safetensors
   的视图（`encoders/text/files.py`：模型目录的分片、ComfyUI 文件、`stored` 文件）；
   在张量移动之前不做任何复制。
2. 存储量化的层（ComfyUI、`stored`）在文件页之上附加其基础权重。
3. W4A8 解码和 `memory="auto"` 在读取任何权重之前就确定，依据是 meta 模型的尺寸加上
   本次加载新增的字节数：int8 解码，以及加载时的量化（通过对每种形状的一个 meta 设备
   线性层做量化来测量）。
4. 逐模块（先是每个解码器层，然后是其余部分）每个张量都去往自己的归属位置：`resident`
   下去计算设备（以及环形缓冲区的最终归一化层），否则留在文件页，当文件存储的是另一种
   计算 dtype 时记录该 dtype（从 bf16 检查点得到 fp16 编码器），由移动它的那次复制应用。
   含待量化线性层的模块会经由工作设备以计算 dtype 处理，在那里量化（每个稠密线性层在
   被替换时立即释放）并以量化形式移往归属位置；反量化的 ComfyUI 层在主机上逐层反量化。
5. 主机形态（`host`）根据一次复制会增加的字节数确定；在 `cpu` 下，已经处于计算 dtype 的
   张量直接从其文件页运行。

嵌入与先把整个模型读入主机再加载的结果相同：每个值都是同一个存储张量的同一种转换。

<a id="host-gemm"></a>
## 主机 GEMM（Host GEMM）

不具备原生 16 位 GEMM 指令（AVX-512 BF16 / AMX，由 torch 的 oneDNN 能力检测判断）的主机，执行 bf16 矩阵乘法时性能远低于 fp32。例如 EPYC 7542（Zen2，AVX2 + FMA）对 `[512, 4096] × [4096, 12288]` 线性层，fp32 可达 1060 GFLOP/s，而 bf16 只有 155 GFLOP/s、fp16 只有 219 GFLOP/s。

对于这种主机，在主机上进行计算的所有配置（文本编码器的 `cpu`、VAE 的 `cpu`）都会启用 `HostGemm`（`kohakumoss_kernels.hardware.host_gemm`，根据能力检测只绑定一次）。它将一次前向中对 16 位主机张量执行的矩阵乘法类操作（`mm`、`addmm`、`bmm`、`baddbmm`、`mv`、卷积以及 CPU Flash Attention）统一先把输入提升到 fp32 执行，再将结果仅舍入一次回原有 16 位 dtype。

进入 GEMM 时激活已经是 16 位值，两个 bf16（或 fp16）值的乘积在 fp32 中可以精确表示。这因此符合数值精度规范要求的“16 位输入、fp32 累加”，同时以 fp32 GEMM 的吞吐率执行；权重在存储中仍为 16 位，仅在各次调用时提升精度。支持原生 16 位 GEMM 的主机不会安装此适配。

`tests/kernels/hardware/test_host_gemm.py` 使用可精确验证的整数输入测试这一行为。此模式只拦截当前前向过程中实际派发的操作，不能拦截自定义算子内核内部的运算，因此主机运行时会把每个量化线性层绑定为普通算子。

加载 + 编码 + 释放的私有主机内存峰值（进程的 anon + shmem），Qwen3.5-9B fp16
来自 bf16 检查点，RTX 2080 Ti：

| 计划 | 先把整个模型读入主机 | 流式加载 |
| --- | --- | --- |
| `resident` | 16.4 GiB | 2.7 GiB |
| `ring` | 19.2 GiB | 1.1 GiB（`host` 为 `file`） |
| `offload` | 19.2 GiB | 1.1 GiB（`host` 为 `file`） |

<a id="forward-and-layouts"></a>
## 前向与布局

`kohakumoss.encoders.text.QwenTextEncoder` 移除视觉塔，使用训练聊天模板进行分词，
将提示词分组（最长优先，每次内核调用最多 `batch` 条提示词和 `max_tokens` 个 token），
并执行一轮：

| 布局 | 前向 | 一次调用所见的内容 |
| --- | --- | --- |
| `packed` | `kernels` | 拼接的提示词、`cu_seqlens` 边界、每条提示词重新开始的位置 |
| `padded` | `kernels` / `hf` | 各行向右填充至最长长度（`kernels`：每行为一个填充宽度的片段；`hf`：注意力掩码） |
| `single` | `kernels` / `hf` | 一条提示词 |

内核前向（`encoders/text/forward.py`）使用 transformers 模块自身的线性层、
归一化权重和旋转位置表，并使用[文本算子](#ops)：

- 每个解码器层的块函数由其层类型一次性绑定；
- 按层优先顺序处理一轮中的所有打包组；
- 右侧填充对真实 token 是精确的，因为每一层都是因果的（因果注意力、因果卷积、前向递归）。

<a id="encode-throughput"></a>
### 编码吞吐量

`scripts/bench/textenc/textenc.py`，RTX 5090，bf16，64 条 20-160 词的描述 = 每轮
13.2k 个 token；数值为该轮的 tokens/s / 设备峰值 GiB。`offload` / `ring` 每轮移动
13.8 GB 权重，两轮之间不保留任何权重：

| 前向 | 内存 | 布局 | batch 1 | batch 8 | batch 32 |
| --- | --- | --- | --- | --- | --- |
| kernels | resident | packed | 4671 / 15.1 | 12540 / 15.3 | 14332 / 16.1 |
| kernels | resident | padded | 4713 / 15.0 | 12137 / 15.3 | 10165 / 16.5 |
| kernels | offload | packed | 3890 / 13.1 | 8797 / 13.4 | 9183 / 14.2 |
| kernels | ring | packed | 4652 / 1.8 | 12778 / 2.1 | 12714 / 2.9 |
| kernels | ring | padded | 4554 / 1.8 | 11761 / 2.1 | 9652 / 3.3 |
| transformers | resident | padded | 2383 / 15.0 | 9325 / 15.6 | 6709 / 17.9 |

从 batch 8 开始，`ring` 达到常驻模式的速度：层环形缓冲区将权重复制隐藏在
批处理一轮的计算之后。

<a id="accuracy-against-transformers"></a>
### 与 transformers 的精度比较

检查使用误差统计检查（`tests/_util/error_stats.py`），以相同精度、相同批处理方式的
transformers 编码作为基线（每次调用一条提示词时采用其非批处理编码，批处理布局则采用
其填充批次）：记录相对于 fp64 编码的统计表，只有出现退化时布局才判为失败。
fp64 编码是在主机上执行的内核前向，其中的算子对照独立参考实现检查。

真实权重上的结果（32 条描述，1-300 个 token；`tests/model/test_text_encoder_real.py`，
RTX 5090），绝对误差 / 相对误差：

| dtype | transformers single | transformers padded | kernels single | kernels packed | kernels padded | 打包，合并为一个片段 |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 22.7 / 3.1e5 | 21.1 / 1.2e6 | 10.5 / 6.8e5 | 16.8 / 6.2e5 | 22.5 / 1.0e6 | 76.6 / 4.1e6 |
| fp16 | 1.78 / 1.5e4 | 3.54 / 6.9e4 | 2.07 / 1.1e5 | 7.89 / 1.5e4 | 2.13 / 3.9e4 | |

- 相对误差由接近零的参考元素决定，绝对误差由少数舍入差异经过 32 层不断放大的提示词决定。
- GEMM 形状或归约顺序的任何变化，都会使二者变化约 2 倍（transformers 自身两种批处理方式
  的 fp16 绝对误差相差 2x，相对误差相差 4x）。
- 将所有提示词合并为一个片段的打包执行（提示词跨边界进行注意力和递归）超出两个界限
  （绝对误差为 transformers 非批处理编码的 3.4x，相对误差为 13x）。

<a id="encode-ahead"></a>
## 提前编码

使用在线文本缓存时，`TextAhead` 包装加载器：

- 每 `ahead` 个主机批次，在一轮中编码其中所有未命中缓存的内容；
- 将结果保留在主机或设备（`ahead_device`）上，直到批处理器取用；
- 其检查点状态是尚未发出的第一个批次的位置，因此恢复时既不重复，也不跳过。

在 `ring` / `offload` 下，一轮只需移动一次权重即可处理 `ahead` 个批次。

<a id="quantization"></a>
## 量化

| `precision` | 存储 | GEMM |
| --- | --- | --- |
| `int8` / `fp8` | 仅权重量化，通过 `QuantWeight` 和 `quant_linear` 内核按输出通道量化，或按每 `quant_group` 个输入列量化，每次调用时反量化 | 16 位 |
| `fp8_compute`（`fp8_recipe` row / tensor / block）、`mxfp8` | `ComputeWeight`，权重在计算设备上量化一次，激活每次调用量化 | 调优的 `fp8_linear` 算子，sm89+（[quant-kernels.md](quant-kernels.md#fp8-compute)） |

输入或输出特征数少于 256 的线性层保持稠密，尺寸不是 128 倍数的 fp8-compute 线性层也一样。

每个线性层都从文本编码器检查点存储的数值（按其权重名对应的 safetensors 张量）量化，
而不是从该层加载到工作设备的计划 dtype 副本量化
（[quant-kernels.md](quant-kernels.md#source)）：无论 dtype 如何，编码值都是 bf16
检查点的编码值，因此 fp16 编码器（sm75）的量化结果与 bf16 相同。量化报告在
`from_source` 中统计这些层。

RTX 5090 上的 Qwen3.5-9B（`scripts/probe/fp8_inference.py`，5 条短提示词），
嵌入对比主机 fp64 编码：

| 精度 | 常驻权重 GiB | 余弦相似度 | 绝对误差均值 |
| --- | --- | --- | --- |
| bf16 | 14.8 | 0.99984 | 0.022 |
| fp8_compute row | 8.38 | 0.99635 | 0.118 |
| fp8_compute tensor | 8.38 | 0.99604 | 0.124 |
| fp8_compute block | 8.38 | 0.99654 | 0.115 |
| mxfp8 | 8.79 | 0.99593 | 0.126 |

在该规模下，每个 GEMM 都受启动开销限制，fp8 编码耗时 73-80 ms，而 bf16 为 43 ms。

<a id="pre-quantized-text-encoder"></a>
### 预量化文本编码器

`scripts/export/text_quant.py` 将一种精度的量化线性层（`int8` / `fp8` 按通道仅权重量化、
使用 `FP8_RECIPE` 的 `fp8_compute`、`mxfp8`）写入一个 safetensors 文件：

- `<n>.weight` 保存编码值（从检查点存储的数值量化而来，因此在任何 `SAVE_DTYPE` 和任何
  计算 dtype 下都等于加载时量化），`<n>.weight_scale` 保存配方的缩放值，`<n>.bias`
  保存偏置；
- 其他所有张量均使用 `SAVE_DTYPE`（`bf16` 保留检查点的数值）；
- 元数据键 `kohakumoss_text_quant` 记录精度、配方和层列表（`kohakumoss/quant/stored_text.py`）。

训练和推理都通过计划字段 `stored` 指定文件：

```python
TEXT_ENCODE = {"precision": "fp8_compute", "stored": "<file>"}
```

- 编码器由文本编码器目录的配置构建，不初始化或读取稠密权重。存储的基础权重以
  `QuantLinear` 附加在文件页之上，其余张量从文件流式加载，因此加载时不做任何量化。
- 文件的精度必须与计划一致（或计划的精度为 `auto`），且 `quant_group` 必须为 0。
- 分词器和配置仍来自权重根目录。

<a id="comfyui-files"></a>
### ComfyUI 文件

ComfyUI 文本编码器文件（`text_encoders/qwen3.5_9b_*`，参见[权重指南](../guides/weights.md#text-encoder)）通过 `comfy_checkpoint` 进行流式加载（[流式加载](#streamed-load)）：先根据软件包内置配置构建不初始化权重的文本模型，再将 `model.language_model.*` 权重流式加载到对应模块。分词器也使用软件包内置版本。

- `precision="auto"` 由文件头和量化记录（`text_file_format`）判断存储格式：bf16 文件对应 `compute`，其他可能为 `int8`（int8 ConvRot 线性层）或 `w4a8`。已量化线性层使用附着于原始存储权重的 `QuantLinear`，int8 嵌入层使用 `QuantEmbedding`；**加载时不会重新量化**。即使在主机（`memory="cpu"`）上，权重量化格式仍按其存储形式运行，从而使离线 CPU 文本缓存生成的键能与 GPU 运行读取的键一致；只有存储为 fp8-compute 的格式在主机上会退回 `compute`。主机侧 `QuantWeight` 将调用绑定为普通算子组成的 eager 路径（`_host_linear`：输入先旋转，再经过稠密线性层），使各核心上的 GEMM 都进入[主机 GEMM](#host-gemm)。`HostPlacement` 在每个解码器层执行前，仅为该层临时构建反量化后且保持旋转布局的 16 位稠密权重，执行结束随即释放。在 EPYC 7542（32 线程，bf16）上，ComfyUI int8 ConvRot 文件进行 65 条描述的 `cache_text` 编码，每条耗时 3.83 秒（私有内存 3.3 GiB）；若每轮只有 4 条描述，则每条需要 5.6 秒，因为逐轮反量化的摊销成本更高。
- 选择其他任何精度时，都会先将存储的量化层反量化到稠密模型（会输出一条 INFO），然后根据目标精度再次量化。在 `offload` / `ring` 下，主机保存每层的原始量化代码与缩放系数（文件页），复制到设备时再通过 `Deferred` 在设备端反量化（见[文本编码器的卸载](offload.md#text-encoder)）。以 RTX 3090 上的 fp16、int8 ConvRot 文件为例，加载 + 编码 + 释放的主机私有内存峰值在 `ring` / `offload` 下为 3.6 GiB；若在主机一次反量化全部层则为 19.0 GiB（嵌入表仍在主机上反量化）。两种路径生成的权重按位相同（逐元素缩放，并且 fp32 Hadamard 蝶形变换在所有设备上都按索引顺序累加）。在 `cpu` 下，反量化后的稠密层驻留于主机内存；超过主机余量时会给出 WARNING。
- W4A8 文件的 `w4a8` 解码方式：`int8` 在加载时为每层解码一次（于计算设备上执行，结果放到该层的目标位置）；`packed` 保留 4 位压缩编码、每次调用时解码（CUDA / XPU 使用编译后的解码器）；`auto`（`default_w4a8`）在解码后的编码器能够与每步预留显存一起常驻设备时选择 `int8`，否则选择 `packed`。之后 `memory="auto"` 根据最终选择的存储形态估计放置位置。
- `encoder_spec`（文本缓存键）会附加原始存储布局，例如 `bf16/int8/g0/src:int8-convrot256`；离线缓存工厂同样从文件头计算该规格。
- 用于内核预热的无权重孪生模型，会镜像已加载的量化模块结构（形状、格式、旋转方式），从而记录在线编码时实际出现的旋转 int8 调用。

<a id="reporting"></a>
## 报告

`TextEncodeStats` 记录：

| 指标 | 含义 |
| --- | --- |
| `te_time` | 每步秒数 |
| `te_share` | 墙钟时间占比 |
| `te_tokens_per_s` | 每秒编码的 token 数 |
| `te_h2d_gb` | 主机到设备的 GB |
| `te_tflops`, `te_mfu` | 与 DiT `mfu` 使用同一墙钟时间的 TE FLOPs，因此两者可以相加 |

编码器在每轮执行前（`step`）和执行后（`te`）标记显存时间线。

训练器步骤（`scripts/bench/step/step.py --text-weights`，RTX 5090，`low` 配置档，
1 MP 随机权重 DiT，每步一条描述，在线文本；数值为每步 TE ms、文本编码器执行轮次 /
DiT 步骤的峰值 GiB）：

| TE 内存 | ahead | 每步 TE ms | TE 执行峰值（alloc） | DiT 步骤峰值（alloc） | 每轮 H2D GB |
| --- | --- | --- | --- | --- | --- |
| resident | 1 | 44（中位数） | 20.1 | 20.7 | 0 |
| offload | 1 | 533 | 18.2 | 5.9 | 13.8 |
| ring | 1 | 495 | 6.9 | 6.0 | 13.8 |
| offload | 8 | 68（均值） | 18.2 | 5.9 | 13.8 |
| ring | 8 | 100（均值） | 6.9 | 5.9 | 13.8 |

每轮一条描述时受权重复制限制；`ahead` 将复制成本分摊到该轮的各个批次。

采样（`scripts/sample/sample.py`，2 x 1024^2，8 步）：

| 编码器 | 生成秒数 | 峰值 GiB |
| --- | --- | --- |
| `ring` | 14.9 | 14.3 |
| `offload` | 14.3 | 22.8 |
| `resident` | 13.9 | 29.1 |

<a id="ops"></a>
## 算子

`kohakumoss_kernels.ops.TEXT`（签名位于 `kohakumoss_kernels/ops.py`，eager 参考实现位于
`backends/eager/textenc.py`）：

| 算子 | eager 参考实现 | 内核 |
| --- | --- | --- |
| `causal_conv_varlen` | 宽度为 4 的逐通道卷积，以移位的 fp32 乘加实现，对 token 所在片段起点之前的抽头做掩码，silu，一次舍入 | `compile`（eager 参考实现的 torch.compile） |
| `gated_delta_rule_varlen` | 对逐片段填充张量计算门、l2 范数和分块 delta 规则（块大小 64），使用输入 dtype 操作数的普通 PyTorch 矩阵乘法 | Triton（sm80+、XPU）、TileLang（sm70+），各自对比 eager 参考实现计时 |
| `rms_norm_gated` | `x / rms(x) * w * silu(z)`，fp32，一次舍入 | `compile` |
| `attention_varlen_causal` | 逐片段 SDPA（`is_causal`，重复 kv 头） | Triton（sm80+、XPU）、TileLang（sm70+），当 torch 构建包含 flash attention 且显卡为 sm80+ 时，各自对比 torch `varlen_attn`（flash、因果窗口、GQA）计时，否则对比逐片段 SDPA eager 参考实现（Windows wheel 不含 flash attention） |

块 RMSNorm 使用已有的 `rms_norm` 算子，权重为 `1 + w`（Qwen 的零中心归一化）；
MLP 和投影使用模型自身的线性层。

<a id="gated-delta-rule-designs-triton"></a>
### 门控 delta 规则设计（Triton）

对于每个 value 头 h（key 头为 `h // 2`）：`g = -exp(A_log) softplus(a + dt_bias)`，
`beta = sigmoid(b)`，`q, k` 进行 l2 归一化（q 还乘以 `Dk^-1/2`），每个 token 执行
`S <- e^g S + beta k (v - e^g S^T k)^T`、`o = S^T q`，S 在每个片段起点为零。

- `recurrent`：每个（片段、value 头、value 分块）一个程序，每步一个 token，寄存器中保存 fp32 状态。
- `chunk`：每块 64 个 token。
  - `prep_kernel` 通过 16x16 对角块求逆和块前向代入，构造 UT 变换 `T = (I + A)^-1`
    （A 为严格下三角的 `beta_i k_i.k_j e^(G_i - G_j)`）。
  - `state_kernel` 携带状态遍历每个片段的块，写出各块起始状态 H 和
    `V' = T (beta v - beta e^G k_n H)`；第一个块在循环前运行（无状态乘积、无 H），
    每次循环先将前一个块并入状态，因此永远不会形成最后一个块的结束状态。
  - `out_kernel`（按块并行）写出 `o = (q_n e^G) H + (q_n k_n^T . decay) V'`。
- `fused`：`prep_kernel` 还写出衰减后的 QK 矩阵和逐行因子；每个（片段、头、value 分块）
  一个串行内核，在同一轮中生成每个块的输出（无 H / V' 往返）。片段的第一个块跳过状态乘积。
- `short`：将至多 64 个 token 的片段完整打包，每个 64 行分块放入多个片段；每个片段起点的
  状态为零，因此分块输出只依赖分块内部（`o = (QK . same-segment) T (beta v)`，A 仅保留
  同片段的配对）。同次调用中的较长片段使用 `chunk` 或 `recurrent`。分块计划在主机上为
  每个 `cu_seqlens` 张量构建一次（每轮编码器执行同步一次；所有层复用）。

注册表条目按问题键（dtype、头数、头维度、片段数、最长及平均片段长度桶）对每种设计的
分块 / warp / 精度候选计时，并绑定胜者（`scripts/bench/textenc/ops.py` 写出完整计时表）。

<a id="speed"></a>
### 速度

注册表选择的算子，每次调用 ms，Qwen3.5-9B 头布局（delta：16 个 key / 32 个 value 头，
维度 128；注意力：16 个 query / 4 个 kv 头，维度 256），bf16，形状 = 片段数 x token 数；
`fla` = flash-linear-attention 0.5.2 `chunk_gated_delta_rule`（变长），`hf` = transformers
的 `torch_chunk_gated_delta_rule`（填充批次）：

| 形状 | delta，RTX 3090 | fla | hf | delta，RTX 5090 | fla | hf |
| --- | --- | --- | --- | --- | --- | --- |
| 1x300 | 0.175 | 1.251 | 2.28 | 0.183 | 1.609 | 2.44 |
| 16x300 | 0.893 | 1.669 | 12.45 | 0.333 | 1.624 | 4.34 |
| 64x150 | 1.930 | 3.017 | 28.68 | 0.653 | 1.732 | 11.31 |
| 256x40 | 1.853 | 3.466 | 35.80 | 0.381 | 1.834 | 14.44 |
| 1x2048 | 0.511 | 1.227 | 9.49 | 0.208 | 1.618 | 10.07 |
| 4x1024 | 0.806 | 1.529 | 10.14 | 0.290 | 1.633 | 5.75 |

| 形状 | 注意力，RTX 3090 | 逐片段 SDPA | 注意力，RTX 5090 | 逐片段 SDPA |
| --- | --- | --- | --- | --- |
| 1x300 | 0.051 | 0.194 | 0.068 | 0.178 |
| 16x300 | 0.327 | 2.011 | 0.121 | 2.059 |
| 64x150 | 0.403 | 7.730 | 0.155 | 7.674 |
| 256x40 | 0.348 | 28.48 | 0.142 | 28.24 |
| 1x2048 | 0.635 | 0.794 | 0.232 | 0.359 |
| 4x1024 | 0.647 | 1.102 | 0.208 | 0.605 |

3090 上 fp16 的 delta 规则耗时与 bf16 列相差不超过 3%，注意力相差不超过 0.012 ms。

<a id="precision"></a>
### 精度

- 每次矩阵乘法都使用 16 位（输入 dtype）操作数和 fp32 累加。
- 状态以及所有缩放和衰减值保持 fp32；T、块起始状态 H、V' 和衰减后的 QK 只作为 MMA
  操作数，因此以输入 dtype 存储（该次存储就是操作数舍入）。
- eager 参考实现是普通 PyTorch：同一算法，矩阵乘法使用输入 dtype 操作数，每个输出采用
  PyTorch 给出的 dtype（输入 dtype）。误差统计（`kohakumoss_kernels.error_stats`：
  对照 fp64 的余弦相似度与绝对 / 相对误差，并列展示同 dtype 的 eager 参考实现统计；
  仅退化时判失败）将内核与该 PyTorch 结果比较。
- `recurrent` 仅使用向量运算（无 MMA），检查方式相同。
- fp32 输入只对 `recurrent` 计时（MMA 只接受 16 位操作数）。

两项选择使分块内核的误差保持在 eager 参考实现的水平：

- 衰减值由同号项求和：16 行块内用掩码累积和，跨块用后缀 + 块总和 + 前缀。
  两个很长的前缀和相减得到 `G_i - G_j` 时，大数求和的舍入会丢失其中的小差值。
- softplus 使用 `log1p(exp(x))`（libdevice）；`log(1 + exp(x))` 会丢失小 softplus 值的
  大部分有效数字，误差沿衰减过程累积。
