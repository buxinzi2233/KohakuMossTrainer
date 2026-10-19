[English](../../internals/tuning.md)

<a id="tuning"></a>
# 调优

每个需要调优的内核都会选择自己的 tile 配置，每个块运算都会选择自己的内核或
`torch.compile`：每张卡上测量一次候选，并将胜者存入调优表。之后运行的进程直接读取该表，
不再进行计时。

| 问题 | 答案 |
| --- | --- |
| 使用哪个开关 | `TUNE`（[TUNE](#tune)）：`table`（训练默认）或 `off`（推理默认） |
| 训练运行何时计时 | 第一步之前的预热，之后仅在[调优窗口](#tuning-windows)内计时 |
| 重新调优持续多久 | `RETUNE_STEPS`（默认 3）：前 N 步中每一步之后；随后内核与 compile 的排队被封存（第一个样本网格的窗口仍会为自己的形状计时） |
| 表所在位置 | `<cache>/tuning/<card-sm-cuda-torch>/<tag>/<kernel>.json`（[表](#select)），缓存根目录为 `KOHAKUMOSS_KERNEL_CACHE`（[缓存目录](#cache-directory)） |
| 如何提前填表 | `kohakumoss-tune`（[填表：kohakumoss-tune](#filling-the-table-kohakumoss-tune)） |

<a id="tune"></a>
## TUNE

运行开关 `TUNE`（训练、`sample.py`、`MossPipeline(tune=...)`）每个进程通过
`tuning.set_mode(...)` 应用一次：

| 值 | 计时 | 表 | 行为 |
| --- | --- | --- | --- |
| `table`（训练默认） | 是 | 读、写 | 已存储的决策直接绑定；缺失的键计时一次并保存胜者。训练：在第一步之前的预热中（见下文），之后仅在调优窗口内。推理：在该键第一次 eager 调用时；自己拥有模型的流水线会在每次生成后测量排队的 kernel-vs-compile 形状 |
| `off`（推理默认） | 否 | 读 | 已存储的决策直接绑定；缺失的键采用规划器的第一个候选（内核或 compile 运算：即其内核），从不存储 |

`RETUNE_STEPS`（训练用，int >= 0，默认 3）表示在 `TUNE="table"` 下，进程的前多少步会在
步与步之间执行内核与 compile 的重新调优。

<a id="lycoris-kernels"></a>
### LyCORIS 内核

在两种取值下，LyCORIS 内核都通过 LyCORIS 的进程级开关 `lycoris.kernels.set_tuning`
（LyCORIS >= 4.1.2）跟随 TUNE：

- `kohakumoss.adapters.lycoris.tuning.set_lycoris_tuning` 设置该开关，并返回之前的设置，
  供 `restore_lycoris_tuning` 恢复。
- `table` 下打开，`off` 下关闭（LyCORIS 随后绑定其已存储的 tile 方案，否则使用其第一个）。
- 训练器在设置 `tuning.set_mode` 时一并设置它，并在运行结束时恢复；在 `table` 下，
  第一步之前的适配器预热负责 LyCORIS 的 tiling，与其他调优调用一样处于 `rng_kept`
  之下。

LyCORIS 自身的调优不会改变调用方的随机数生成器状态。它的成本模型每进程测量一次设备，
在第一次对 tile 排序时进行（其 tile 表在磁盘上未命中，发生在预热中，或在实时适配前向中
遇到新形状时），使用来自私有生成器的随机数据，因此预热和实时前向都不会消耗运行的随机流。
`tests/kernels/tuning/test_rng_kept.py` 在 GPU 上检查一次冷启动的 LyCORIS tile 调优
（反例：探针未使用其私有生成器）。

<a id="decision-log"></a>
### 决策日志

决策按问题键通过 `tuning.decisions` 记录（[logging.md](../guides/logging.md)）：

| 来源 | 记录为 |
| --- | --- |
| `select`，仍是候选的已存储行 | `table` |
| `select`，已计时 | `measured`（附带候选列表的计时） |
| `select`，`TUNE="off"` 且无存储行，或只有一个候选 | `default` |
| `select`，临时的首候选返回（此处不允许计时，属于跟踪） | 无 |
| kernel-or-compile 运算，已存储行 | `table` |
| kernel-or-compile 运算，`TUNE="off"` 且无行 | `default` |
| kernel-or-compile 运算，为 `retune_pending()` 排队的问题 | `queued`（DEBUG）；重新调优的测量结果即决策 |

每个 `(kernel, key)` 每进程只记录一次，因此已决策问题的第二种调用模式不会新增记录。
批次会合并为一张 INFO 表输出：预热后、每次实际测量的重新调优后、训练第 1 步结束时、
每次生成后。第 1 步之后（`go_live`），已测量的决策是一条 INFO 行，包含键、选择和计时。

<a id="where-nothing-is-timed"></a>
### 不做计时的位置

- 在 `torch.compile` 跟踪或 CUDA 图捕获内部，缺失的键会不计时地采用第一个候选
  （不存储、不确认），因此编译后的图不包含调优代码。
- `select` 在 autograd 反向传播内部不计时（除非该内核不启动 autograd，
  `in_backward=True`）。
- 对于 autograd 启动（`autograd=True`），在推理模式或保存张量钩子（做过检查点的前向）
  下不计时。

<a id="tuning-windows"></a>
## 调优窗口

训练运行仅在这些窗口内调优内核：

| 窗口 | 计时内容 | 归还驱动的显存 |
| --- | --- | --- |
| 第一步之前的预热（以及 `kohakumoss-tune`） | 每个已规划的问题 | 每个阶段前后，每个受限配置之前 |
| 在线文本编码器的预热，恰在其首次前向之前（`TextPrewarm`） | 其打包桶 | 之后 |
| 进程前 `RETUNE_STEPS` 步中每一步之后（默认 3；`RetuneKernels`） | 排队的 kernel-vs-compile 问题（`torch.compile` 对比内核） | 仅当排队问题的输入放不下时 |
| 运行的第一个样本网格，直到其第一个采样器步骤结束（`PipelineImages`） | 采样形状缺失的配置 | 采样器自身的释放（编码后、采样后、解码后） |

窗口状态位于 `kohakumoss_kernels.tuning.window`：

| 调用 | 时机 | 效果 |
| --- | --- | --- |
| `window.freeze()` | 预热结束时（`end_prewarm`） | 窗口之外 `select` 不计时；首次在此处遇到的键会绑定第一个能构建并运行一次的候选，不计时、不存储，记录在 `tuner.DEFERRED` 中并记录一次 WARNING（`met outside a tuning window`）：预热的计划未覆盖该形状，下次运行的预热在计划覆盖后会处理它 |
| `window.tuning_window()` | 文本编码器预热、第一个采样器步骤 | 在内部解除冻结 |
| `window.timing_here()` | 重新调优 | 在内部解除冻结（以及 go-live 延迟） |
| `window.seal()` | 最后一个重新调优步骤之后（`RetuneKernels`） | kernel-or-compile 运算不再排队新问题（绑定内核）；仍在排队的问题被丢弃并保留在内核上（一条 INFO 行 `kernel-vs-compile retune closed after step N`）；此后该回调为空操作 |
| `window.reopen()` | 运行结束，配合 `tuner.clear_caps()` | 不冻结、不封存、不延迟、无进程调优预算：同一进程之后进行的推理或调优调用会像全新进程一样调优（决策日志的实时模式保留） |

- 重新调优计数按进程计算，因此恢复运行会重新调优自己的前几步。
- 在进程组下，第 1 步之后的窗口只解除冻结，不解除 go-live 延迟，因此其缺失的键在那里
  仍不计时（[多进程](#multi-process)）。
- 第一个样本网格的 VAE 解码在其第一个采样器步骤之后运行，缺失键在此处不计时地绑定。

`tests/kernels/tuning/test_tuning_window.py` 检查：冻结的键不计时地绑定，窗口内会为其
计时（反例：忽略冻结的 select）；重新调优只在前 N 步运行然后封存（反例：每步都重新调优）；
封存后的运算不排队（反例：封存后仍排队）；只有第一个样本网格的第一个采样器步骤开放
（反例：每个网格都开放）。

<a id="filling-the-table-kohakumoss-tune"></a>
## 填表：kohakumoss-tune

`kohakumoss-tune`（也可用 `kogine run scripts/bench/tune/tune.py`）针对一张卡、一种 dtype
和一个图像尺寸列表运行一次训练预热，并存储每个决策；之后的运行直接绑定它们，不再计时。

```
kohakumoss-tune --device cuda:0 --dtype bf16 --sizes 1024x1024 832x1216 --batch 4
kohakumoss-tune --dtype fp16 --sizes 1024x1024 --backward   # training shapes
```

| 标志 | 含义 |
| --- | --- |
| `--device`, `--dtype` | 显卡与 16 位 dtype（`bf16` / `fp16`） |
| `--sizes HxW ...` | 图像尺寸；最长的联合样本（latent 词元 + `--text-tokens`）决定每样本长度 |
| `--batch` | 每个生成批次的图像数；打包长度为 `sample x batch`，除非使用 `--backward`，否则为 CFG 加倍 |
| `--variant` | 架构（`base`、`sft`） |
| `--text-tokens` | 每个样本加到各尺寸 latent 词元上的文本词元数（默认 512） |
| `--backward` | 前向 + 反向（训练形状，不因 CFG 加倍） |
| `--trainable` | 与 `--backward` 合用：`adapter`（默认；冻结基础权重，激活携带梯度）或 `full`（所有可训练权重） |

被记录的 pass 在它所调优运行的 autocast 下执行，因此存储的键正是该运行查找的键
（`tests/cli/test_tune_autocast.py`，反例：无 autocast）：

- **带 `--backward`：** 训练器向所选 dtype 的 autocast
  （`kohakumoss.precision.autocast_factory`），因此条件线性层返回训练步骤传入的 16 位
  shift / scale / gate。
- **不带：** 采样器的 autocast（`cli.tune.record_autocast`：
  `precision.autocast_for` 计算 dtype，与 `inference.pipeline.sampling_autocast` 一致），
  因此存储的键携带采样调用遇到的 dtype。

它会记录测量的问题和表目录，并在之后释放调优工作区。

<a id="select"></a>
## Select

内核通过 `kohakumoss_kernels.tuning.select(kernel, device, key, candidates, launch)`
获取 tile 配置：

1. **规划器。** `candidates()` 枚举合法配置（显卡的共享内存、寄存器和 warp 分区限制），
   并用 `tuning.plan.rank` 排序：一个 roofline 估算（MMA flops 和指数运算对比实测
   MMA 速率，DRAM 字节对比实测带宽）再乘以 SM 上的波量化系数。注意力内核会把该架构实测的
   种子 tile 排在排序结果之前（[kernel-dev.md](kernel-dev.md#measured-design-per-arch)）。
2. **候选列表计时。** 前 `shortlist` 个配置在普通启动周围用设备事件计时
   （`timing="events"`）：先进行繁忙预热以提升频率，然后取十次调用中的最佳值，并将主机
   启动成本计入，因为 eager 训练循环需要承担它。`timing="graph"`（CUDA 图重放，不含启动
   成本）用于基准测试。调优器只用事件计时：在训练进程中进行捕获会让其他线程（DataLoader
   的 pin-memory 线程）的 CUDA 调用失败，全局捕获模式和（在 Windows 上实测）线程局部
   捕获模式都是如此。每个配置都在[工作显存上限](#working-memory-bound)下运行。
3. **表。** 胜者、每次计时、每个配置的工作显存（`working_mib`）以及每次失败都写入
   `<cache>/tuning/<card-sm-cuda-torch>/<tag>/<kernel>.json`（原子替换），其中 `<tag>`
   是 kohakumoss / triton / tilelang 版本的哈希。之后的进程读取该表，不再计时；已存储
   配置若不在当前候选中，则重新调优。

计时期间失败的配置：

| 失败 | 处理 |
| --- | --- |
| 构建 / 启动错误（`errors.BUILD_ERRORS`） | 记录并跳过 |
| 超过工作显存上限 | 记录并跳过 |
| 其他显存不足 | 跳过并释放缓存；该问题仅在本进程绑定运行过的最佳配置（否则第一个），不存储 |
| 所有配置都失败 | `ConfigsFailed` 将该运算交给注册表后备（[kernel-dev.md](kernel-dev.md#fallback)） |

各内核的候选列表大小：

| 内核 | 候选数 |
| --- | --- |
| Triton 注意力（前向、dK/dV、dQ、融合反向） | 10 |
| TileLang 注意力（前向、dK/dV、dQ；每个配置都是一次 nvcc 编译） | 6 |
| Triton 注意力反向策略（融合 vs 拆分） | 两者 |
| 注意力入口（后端内核 vs torch `varlen_attn` / SDPA 循环） | 每种实现 |
| 融合线性层、TileLang 块运算 | 每个合法配置 |
| 量化 matmul（Triton、TileLang） | 每个 tile 加上 PyTorch 路径 |
| 块运算的内核 vs compile | 两者 |
| 其他（`select` 默认） | 4 |

表的写入：

- 每次写入都会先合并文件中已有的行，因此同时调优同一模型不同显卡的进程（基准矩阵池、
  DDP rank）会保留彼此的行。
- 写入通过一个按进程、线程和写入唯一的临时文件完成，然后替换到表中（若另一进程持有该文件
  则重试，Windows 上会抛 `PermissionError`）；共享同一张表的线程一次只写一个。
- 仍然失败的写入（磁盘满、缓存目录只读）会为每个文件记录一条 WARNING，并将行保留在内存中：
  选择在本进程生效，且永远不会被记录为后端失败（`hardware.device.save_json`；设备探针、
  工具链缓存和失败记录也以同样方式写入）。

<a id="problem-keys"></a>
### 问题键

问题键对可变维度进行分桶，因此分辨率变化的训练运行会归结为少量键：

- 词元数、段数和最长段通过 2 的幂桶进入每个存储键。固定模型维度、dtype 和标志按原样进入。
- 存储键中不包含任何按进程变化的值（设备索引、id、指针），因此相同配置重启后会命中存储行。
- 注意力键为 `<dtype>-h<heads>-d<head dim>-L<最长段向上取整到 2 的幂，至少 64>-S<段数桶：1、2、4、8 及以上>`；
  词元数不属于键（网格覆盖它）。
- 注意力入口的实现选择会增加 `-g1`（有反向运行，计前向 + 反向）或 `-g0`（仅前向，
  只计前向）。

<a id="working-memory-bound"></a>
### 工作显存上限

调优时并不知道运行所需的确切 VRAM，因此调优会将自身峰值限制在运行自身调用所需的范围内。

- **工作显存。** 配置的工作显存是计时开始时已分配字节之上的峰值分配字节
  （`hardware.memory.working`）。
- **上限。** 同一问题的配置按候选列表顺序计时；每个配置运行时的上限为
  `WORKING_FACTOR`（8）乘以其之前计时配置中最精简的工作显存，至少
  `WORKING_FLOOR`（256 MiB）；第一个配置没有上限。
- **强制执行。** 分配器执行该上限：对计时调用，将每进程显存比例
  （`set_per_process_memory_fraction`，CUDA 和 XPU）设为开始时保留的字节加上限，随后恢复，
  因此该配置最多将保留显存增加该上限；超过上限的分配会在实际占用显存前抛出内存不足。
- **清理。** 预热中每个受限配置都从空的分配器缓存开始（之前调用
  `hardware.memory.trim`），因此配置能否容纳从不取决于先前配置留下的空闲块缓存，同一张卡上
  两次运行会存储相同的判定（`tests/kernels/tuning/test_tuner_working.py`，反例：不清理）。
- **判定。** 超过上限的配置记录在 `failures` 下（"over the working-memory bound"），
  上限内最快的配置照常存储（同一问题在之后的任何进程中都会再次排除它）。若上限不低于设备
  空闲显存，或分配器不接受比例，则不执行上限，真实的内存不足按上面的按进程跳过处理。

该规则与具体算子无关。主要情形是需要填充到稠密布局的 PyTorch 路径候选。eager 的
`gated_delta_rule_varlen` 在 fp32 下将每一段填充到最长段，因此在一包包含一行 4110 词元和
31 行 1 词元的数据上，它会处理 32 x 4160 个填充行：

| 情形 | 工作显存 |
| --- | --- |
| 填充路径，2 个 128 维 value head（主机峰值，bf16 输入） | 1040 MiB |
| 填充路径，文本编码器的 32 个 value head | 约 16 GiB |
| 仅同样的最长行，2 个 head | 51 MiB |
| 32 行 128 词元，2 个 head | 83 MiB |

Triton 设计直接处理 4141 个打包行：在 32 个 value head 下，chunk 设计分配输出
（32 MiB）、chunk 状态（96 MiB）、T 矩阵（24 MiB）和归一化值（32 MiB）。有了该上限
（chunk 领先配置工作显存的 8 倍），填充路径在偏斜包上被跳过，在其他包上仍会计时，调优峰值
就是运行所用内核的峰值。

候选列表把领先设计排在最前（`triton/textenc/delta.candidates`：chunk 领先配置，然后是
PyTorch 路径），因此第一个无上限配置是精简的。当默认值不是精简配置时，
`select(..., lean=i)` 会先计时候选列表第 `i` 项，并保留第一个候选作为默认值：适配器运算
先计时其主体候选，再计时 LyCORIS 内核（其首次调用还会运行 LyCORIS 自身的 tile 调优），
见[适配器自动调优运算预热](#adapter-autotuned-op-prewarm)。

<a id="device-description"></a>
## 设备描述

`hardware.device.measure_rates(device)` 返回显卡的驱动限制，以及三个实测速率（带 fp32
累加的 16 位 GEMM TFLOP/s、DRAM 和 L2 拷贝 GB/s），每张卡 / CUDA / torch 组合测量一次，
并缓存在 `<cache>/devices/`。在跟踪或图捕获内部不进行测量：规划器使用未测量的 `query`
描述及默认速率。

<a id="kernel-vs-compile"></a>
## 内核 vs compile

每个块运算和融合线性内核都注册为 `KernelOrCompile`
（`backends/compile/choice.py`，表 `choice_<backend>_<op>`）：按问题（分桶形状加上张量的
梯度标志，`shape_key`）绑定其后端的内核或 `compile` 后端的可调用对象。只有最高优先级后端
的内核和 `compile` 参与计时；其他后端只能通过注册表后备到达，否则绝不构建或计时。

一次调用按以下顺序查找其问题：

1. 已绑定的可调用对象，按调用键：设备、梯度模式、算子内线程数、autocast dtype 以及每个
   参数的布局（其在 `shape_key` 桶中的主导词元维度）或值，因此无论词元数如何变化，绑定表
   每个词元桶只保留一个条目（`tests/kernels/tuning/test_choice_binding.py`，反例：精确形状）；
2. 已确定的选项或先前测量的存储行；
3. 否则使用内核，并将问题排队（直到 `window.seal()` 停止 kernel-vs-compile 排队）。

由 torch.compile 跟踪的调用会运行内核，但不绑定任何内容。调用路径中不做计时，因此前向
和它的检查点重算总是绑定同一实现。

- **`retune_pending()`** 在任何 autograd 图之外运行：训练中在前 `RETUNE_STEPS` 步的每一步
  之后（[调优窗口](#tuning-windows)），自有流水线中在每次生成之后。它用按自身 dtype 构建的
  合成输入测量排队的问题（对需要梯度的参数执行前向 + 反向），并存储胜者；每个问题下一次
  调用会绑定其胜者。若排队问题的合成输入无法放入设备显存，则记录 WARNING 后丢弃，并保留
  在内核上。
- **`decide(*args)`** 直接测量一个问题（基准测试、预热）。
- 在 `TUNE="off"` 下两者都不测量。
- 每个设备首次使用时，内核会在无 autograd 的情况下运行一次（`_probe`），因此无法构建的
  内核会在记录任何图之前将运算交给注册表后备。

<a id="pipelines"></a>
### 流水线

拥有自身模型的 `MossPipeline`（`from_pretrained`，或构造函数 `owned=True`）会将其
`tune`（默认 `off`）应用为进程的 TUNE 和 LyCORIS 的 TUNE：

- `tune="off"`：其调用绑定该卡已存储的决策，否则使用规划器的第一个候选。
- `tune="table"`：缺失的键在其首次 eager 调用时计时，排队的 kernel-vs-compile 问题在每次
  生成后测量（`retune_pending`）；胜者会被存储。基于共享训练模型的流水线
  （`from_components`）将 TUNE 留给训练器。

<a id="prewarm"></a>
## 预热

模型预热（`kohakumoss.tuning.prewarm.ModelPrewarm`）从模型结构中发现需要调优的运算。
训练器（`training.build.kernels`）和 `kohakumoss-tune`（`kohakumoss.cli.tune`）使用同一个类。

| 部分 | 模块 |
| --- | --- |
| 预热 | `kohakumoss.tuning.prewarm.ModelPrewarm` |
| 计划长度 | `kohakumoss.tuning.lengths` |
| 训练器构建 | `training.build.kernels` |
| 步骤回调 | `training.callbacks.kernels` |

内核调优需要调用的张量形状和 dtype，而不需要权重，因此它从无权重孪生模型记录，并将自身
VRAM 保持在运行最大计划尺寸下单个算子所需的范围。它分五个阶段运行。

<a id="1-twin"></a>
### 1. 孪生模型

`kohakumoss.training.build.shapes.run_shapes` 在 meta 设备上构建运行所用的主干
（`model.build.build_shapes`）：

- 它用显存计划所用的相同函数在那里应用运行的基础存储
  （`kohakumoss.memory.apply.quantize_base`：int8 / fp8 / fp8_compute / mxfp8 基础及其
  fp8 输入梯度副本；`trainable_compute`：可训练基础的 fp8 计算缓存）。
- 它将每个张量物化为设备上一个零元素的 stride-0 视图（`model.build.materialize`）：
  每个张量只占一个元素的内存，因此孪生模型不持有权重显存，也从不读取真实权重。
- 每个参数采用所构建主干同名同形状参数的 dtype 和梯度标志（master、KEEP_FP32 模块、
  训练中的 norm）；norm eps、残差缩放和 fp16 修复流配置从它复制；所构建主干上挂载适配器
  的线性层返回需要梯度的输出，与适配器的输出一致。
- 量化基础和 fp8 缓存绑定到设备内核。

`kohakumoss-tune` 使用相同的 `build_shapes`，但不构建主干。

<a id="2-record"></a>
### 2. 记录

在一个小型合成批次上运行一次孪生模型前向 + loss，位于
`kohakumoss_kernels.recording.recording()` 内，使用运行的 autocast，并带有不保留任何内容的
保存张量钩子（`kohakumoss.tuning.prewarm.keep_nothing`：记录的 pass 从不执行反向，因此
不持有激活或反量化权重）。

- 每次 `Bound` 调用都连同其运行时的 autocast 状态被记录，并由 eager 参考实现应答，因此
  不会构建或调优任何内容。
- 每个张量参数在调用时被快照（`Snap`）：形状、dtype、设备、梯度标志，以及当元素数不超过
  65536 时其值的设备副本（id、边界、位置、逐样本向量）。记录不持有活动张量。

<a id="3-replay"></a>
### 3. 重放

`kohakumoss_kernels.tuning.replay.replay` 在每个计划的联合长度桶上运行每个唯一的已记录
调用，一次一个算子，在参数需要梯度处执行前向 + 反向，并处于记录的 autocast 状态，因此重放
的问题携带实时步骤传入的 dtype。

- **长度。** 桶在其顶部或计划最大值（`planned_lengths`：数据计划构建的最大联合批次）处重放，
  取较小者（`block_lengths`）；顶部桶的键不变。
- **参数。** 从记录的设备上的快照物化：保留的值沿词元维度重采样（类似 `cu_seqlens` 的边界
  以及最长段参数按比例缩放），更大的张量用相同布局的新值（普通浮点数；整数在记录范围内均匀
  分布）。计时不依赖具体值。
- **显存。** 每个问题的张量在其结束时丢弃，每个算子的空闲块在其问题之后归还驱动。
- **后备。** 每个问题像构建时一样通过该算子的 `Bound` 运行（`registry.exercise`）：若某个
  后端在前向或反向中无法构建或启动，孪生模型的算子和模型中同一查询的算子都会在第 1 步之前
  回退（一条 WARNING），构建 / 工具链原因会记录到磁盘
  （[kernel-dev.md](kernel-dev.md#fallback)）。
- **计时。** 内核调优和 kernel-vs-compile 测量在此处运行，每个计时配置都在工作显存上限下；
  排队的 kernel-vs-compile 问题在最后决定。

**注意力段桶。** 注意力会在每个段长桶上预热，直到计划的最长样本；并且对每个长度，在计划
批次能达到的每个段数桶（1、2、4，然后 8 及以上）上预热，不覆盖其他情况：

- `planned_segments` 给出最短联合样本以及一个批次容纳的最多样本数。
- 最短联合样本是 `Geometry.fit` 在 TARGET_AREA 下、ASPECT_RANGE 范围内产生的最小网格，
  加上数据集最短标题的估算（`MossDataset.text_tokens_min`：词元预算打包时依据的估算列，
  覆盖实时（未墓碑化）行；当启用标题丢弃时，还包括被丢弃标题的估算（空提示词），因为被
  丢弃的样本携带该文本）。训练器在预热前为其构建数据加载器。
- 批次容纳的最多样本数在按数量分批时为 BATCH_SIZE，否则为 MAX_TOKENS 除以最短样本，并受
  MAX_SAMPLES 限制。最长样本长度为 L 的批次最多容纳
  `1 + (MAX_TOKENS - L) // shortest` 个样本（`reachable_segments`）。
- 每次调用在 MAX_TOKENS 内打包一个长度为 L 的段和其余最短长度的段。
- 若存在数据集，最小图像网格取数据集自身的值（`MossDataset.image_tokens_min`：分辨率
  规划器对实时行的逐行目标中的最小值；`tests/training/prewarm/test_prewarm_buckets.py`，
  反例：所有行、不丢弃标题），因此在 UPSCALE="never" 下，小于 TARGET_AREA 的图像保持原尺寸，
  最短样本和段数会跟随数据。
- 若没有数据集，文本最小值是无标题模板（`text_overhead_tokens`）：`kohakumoss-tune`
  从 `--batch` 取数量（生成形状因 CFG 加倍），最短样本取最小 `--sizes`。不带
  `segments` / `shortest` 调用 `ModelPrewarm` 时，取 `batch_max // sample_max` 个最长
  长度的样本。

示例：1 MP 下 MAX_TOKENS 20000（最小网格 3942 词元），标题至少 600 词元时最多达到 4 个
样本，因此 8 段桶永远不会预热。

**注意力流。** 注意力预热将记录 pass 中的每个注意力调用点作为一个流
（`AttentionStream`）：联合注意力（一个段达到计划最长样本，其余为最短样本）以及文本细化器
（标题从数据集最短到最长估算，否则 MAX_TEXT_TOKENS）。每个流在记录的调用需要梯度处执行
前向 + 反向，否则仅前向（`-g0` 键），如适配器运行的冻结细化器。

预热之后在实时调用中首次遇到的键会在调优窗口之外不计时地绑定
（[调优窗口](#tuning-windows)）。适配器词元桶也以同样方式设上限（`token_buckets`）。
VAE 没有需要调优的算子。

<a id="4-decided-problems-are-not-run"></a>
### 4. 已决策的问题不再运行

每个重放的问题（一个算子在一个计划桶上的调用；注意力：其桶扫描）都有一个清单键：算子、
其绑定的后端、autocast 状态、词元维度映射和参数布局（形状、dtype、梯度标志、保留的整数
内容）的哈希。

- 运行期间，`tuner.watching()` 收集它查找的每个调优选择的 `(kernel, key)`
  （候选数大于 1 的 `select`、kernel-vs-compile 表查找）。
- 在 kernel-vs-compile 决策之后，所有选择均已存储的问题会写入 `prewarm_problems` 表。
- 同一张卡和包版本上的后续预热不会运行清单行只列出已存储选择的问题；其实时调用会从表中
  绑定。
- 清单行缺失，或某个选择未存储的问题会再次运行。
- 步骤之间的 `retune_pending()` 只测量实时调用因其表行缺失而排队的问题，因此已决策的键
  绝不会再次计时。

<a id="5-report"></a>
### 5. 报告

每个设备一张表：`op | from table | tuned | skipped | timed | secs | peak GiB |
reserved GiB | bound`，带总计行，然后一行
`prewarm: N problems from the tuning table, M tuned in X s`。

| 列 | 含义 |
| --- | --- |
| `from table` | 已决策的问题（未运行） |
| `timed` | 计时的候选列表 |
| `peak GiB`、`reserved GiB` | 每个算子在其问题期间的设备峰值 |
| `bound` | kernel-or-compile 运算所选的实现（`kernel` / `compile`），或其他运算绑定的后端 |
| `skipped` | 重放问题在后备之后仍然报错（无后端可用、非后端失败的错误、显存不足）：记录为警告，留给实时步骤 |

第 1 步结束时（`PrewarmVram`），一行输出将运行的预热峰值与第 1 步峰值对比。峰值在每个运行的
`build_prewarm` 时从空开始，因此训练多个运行的进程会分别报告每个运行自己的峰值。第 1 步
峰值是整个步骤的一个窗口（其中若发生预热则会重新打开），因此步骤内部的文本编码器和 VAE
阶段标记无法截断它。

<a id="text-encoder-prewarm"></a>
### 文本编码器预热

仅当运行在循环中编码文本时（`TEXT_CACHE="online"` 或 `"off"`）才会调优文本编码器算子，
且只在确实编码时进行。`attach_text_prewarm` 向编码器传入一个 `TextPrewarm`，它恰在编码器
首次前向之前运行一次（`QwenTextEncoder.before_first`），因此所有标题都命中在线缓存的运行
永远不会调优它们。它运行文本预热（[文本编码器桶](#text-encoder-buckets)），记录自己的预热
表，将峰值保留给第 1 步输出行，并将工作区归还驱动；第 1 步的 VRAM 窗口和速度不包含它
（`late_prewarm`）。

<a id="prewarm-tests"></a>
### 预热测试

| 测试 | 检查内容 | 反例 |
| --- | --- | --- |
| `tests/training/prewarm/test_replay_stream.py` | 快照布局 | |
| `tests/kernels/tuning/test_replay.py` | 记录的 autocast 状态；失败的重放会重新绑定记录的算子和模型算子 | 在实现上重放 |
| `tests/training/prewarm/test_prewarm_shapes.py` | int8 基础运行的孪生模型每个张量一个元素，每个量化线性层带有 int8 基础，dtype 和梯度标志与构建的主干一致；记录的 pass 不保留保存张量 | 权重规模的模块、fp32 基础、普通 autograd |
| `tests/training/prewarm/test_prewarm_buckets.py` | 重放点和注意力调用保持在计划最大值内，同时覆盖每个桶 | 桶顶部 |
| `tests/kernels/tuning/test_tuner_working.py` | 工作显存上限 | 无上限 |

<a id="adapter-autotuned-op-prewarm"></a>
### 适配器自动调优运算预热

适配器预热（`AdapterSet.prewarm`）会为每个不同的 `AutotunedOp` 确定一次
（[adapter-binding.md](adapter-binding.md#autotuned-ops)）。

- `signature` 匹配的算子（调优名称，即算子与后端，以及所有无词元轴操作数的布局）共享问题；
  问题键恰好是这些内容、词元桶和梯度模式的函数。
- 在一个算子内，已在本进程确定或已存储在该卡表中的桶与梯度模式
  （`lycoris_<op>_<backend>.json`）会直接绑定而不计时，因此针对同一内核缓存重启不会计时。
- 预热会记录自动调优算子、其中不同的数量、测量的问题及其设备峰值。

**显存。** 它遵循内核预热的规则：

- 暂存输入按词元桶构建，并限制在计划最大值（`token_buckets`）。
- 问题逐个运行：每个运行时输入在其问题到达时构建（`runtime.problems(rows)`：每个桶和
  梯度模式一个构建器），随后丢弃；已测量问题的空闲块在下一个问题前归还驱动（`release`），
  因此一个词元数的缓存块不会堆叠到下一个之下。
- 每个问题的主体候选首先计时（`select(lean=BODY)`）。
- 预热从空分配器缓存开始并在其结束，其设备峰值进入 `note_peak`，因此第 1 步的
  `VRAM peak: prewarm` 行和 `prewarm_peak_*` 指标覆盖它。

**预算。** 每个候选的计时（包括内核首次调用中 LyCORIS 自身的冷 tile 调优）都在一个绝对预算
下运行，即 `hardware.memory.tuning_budget`：内核预热的设备峰值（运行在计划最大值下的块运算，
低于训练步骤峰值）减去该点已分配的字节，最多为运行常驻状态之后设备空闲内存的
`BUDGET_FRACTION`（0.5）。

- **缓存运行。** 带缓存的内核预热几乎不重放或完全不重放，因此其算子峰值接近常驻基线，
  会给出接近零的预算。因此内核预热将其工作字节（设备峰值减去启动时已分配字节）存入预热
  清单，每次预热一个键
  （`prewarm_working|<重放问题键和注意力键的哈希>`）。`note_prewarm` 保留一起运行的算子峰值
  加上当前基线及存储的工作字节，并将较大的工作字节写回；缓存算子的接近基线峰值从不计入。
  因此完全缓存的运行会获得测量它的运行的预算。若无重放且无存储（清单早于存储值），则不记录
  内核预热峰值，只应用空闲内存项。
- **多进程。** 只有 rank 0 运行内核预热，而每个 rank 都运行适配器预热并计时自己的候选
  （`training/entry.py` 的 `_prewarm`）。Rank 0 以工作字节作为门控负载释放预热门控
  （`share_working` -> `release_ranks(gate, payload=...)`，其他 rank 已在等待的存储键，
  无需额外的集合通信），其他每个 rank 将当前已分配字节加上这些工作字节作为其预热峰值
  （`take_working`），因此每个 rank 的适配器预算都是 rank 0 的。
- **超出预算。** 超出预算的词元桶不计时，而是继承（`tuner.inherit`）：同一算子和梯度模式
  下最大已测量桶的决策，存储时带 `"inherited": <源键>` 并记录在 `inherited` 下，因此该尺寸
  的实时调用会绑定它且绝不再调优。当其源桶的最大工作显存按词元行缩放后超过预算，或候选计时
  达到上限（`select(cap=...)` 在分配前抛出 `OverBound`，不存储）时，该桶超出预算。每个模式
  的最小桶总是计时。未计时时内核保持默认，相对上限（最精简者的 8 倍）同样适用。

**预热之后的上限。** 适配器预热结束时（`prewarm.end_prewarm`）将该预算设为进程预算：

- `tuner.set_cap(device, budget)`，因此之后所有未传上限的 `select`（在调优窗口内：步间
  kernel-vs-compile 重新调优、第 1 步的文本编码器预热、第一个采样器步骤）都在其内计时；
  计时会超过上限的键在本进程绑定其第一个候选，不计时、不存储（`tuner.DEFERRED`，一条 INFO
  行 `over the tuning budget`）。
- 预热后的调优将其工作显存留在分配器缓存中，供训练步骤复用；内存不足的配置会归还空闲块
  （`release`）。
- 它关闭 LyCORIS 的 tile 调优（`set_lycoris_tuning(False)`，运行结束时恢复）：预热未调优的
  LyCORIS 键（继承桶的尺寸）会绑定 LyCORIS 已存储或按模型排序的 tile，绝不在步骤内调优。
- 适配器预热将自己的决策作为独立表输出（`after the adapter prewarm`），因此第一次重新调优
  的表只列出重新调优的决策。

`tests/kernels/tuning/test_tuner_working.py::test_live_tunes_stay_within_the_budget`
（反例：无上限）和
`tests/training/prewarm/test_adapter_prewarm.py::test_prewarm_end_caps_every_later_tune`
（反例：LyCORIS 仍在调优）对此进行检查。

**实测案例。** 当输入需要梯度时，内核候选是较重的一个。在 RTX 5090 上
（LoKr full_matrix factor 16、fp16、3072 -> 16384 的 bypass 加 base）：

| 行数 | 内核工作 MiB | 编译主体工作 MiB |
| --- | --- | --- |
| 20000 | 6680 | 2500 |
| 16384 | 5472 | 2048 |
| 8192 | 3040 | 1024 |

从 4096 行起内核更快（20000 行时 5.6 ms 对 9.1 ms）。这些块从一个问题到下一个问题一直留在
分配器缓存中，在 nvidia-smi 中堆叠到 12.8 GB（LyCORIS 4.1.2 上为 11.5 GB），而训练步骤为
6.6 GB，内核预热峰值为 4.0 GiB；以该峰值为预算，16384 和 20000 行的桶会继承 8192 行的决策
（内核）。

`tests/training/prewarm/test_adapter_prewarm.py` 检查：

- 计时时只有一个问题输入存活（反例：每个桶的输入都预先构建）；
- 超预算的桶通过预测和上限继承最大可容纳桶的决策而不计时（反例：无预算，该桶被计时）；
- 完全缓存的内核预热给出未缓存运行的预算，从未测量的键给出空闲内存项（反例：忽略存储的
  工作字节）；
- 报告的峰值包含适配器预热（反例：未记录峰值）。

<a id="text-encoder-buckets"></a>
### 文本编码器桶

打包的文本编码器算子按打包形状的桶为其问题设键，这些桶定义在
`kohakumoss_kernels.tuning.buckets` 中一次，由内核和预热共享：

| 算子 | 键维度 |
| --- | --- |
| `gated_delta_rule_varlen`（triton、tilelang） | 段数：1、2、4，然后 8 及以上合为一个；最长段：pow2 >= 64；平均段：pow2 >= 64 |
| `attention_varlen_causal`（triton、tilelang） | 最长段：pow2 >= 64（段数不是键维度） |
| kernel-or-compile `rms_norm`（q / k / 层 norm） | 词元数：pow2 >= 256（`shape_key`） |

运行时整数（词元数、段数、最长段）是内核参数，不是编译期常量。TileLang 仅按 head 维度、
tile 配置和 dtype 编译；Triton 仅按是否可被 16 整除以及是否等于 1 特化整数参数。因此桶内的
新值会复用已编译内核。键带有版本标签（`KEY_VERSION`），因此旧键格式存储的行绝不会被匹配。

`kohakumoss.encoders.text.prewarm.prewarm_text` 通过编码器文本模型的无权重孪生模型记录一次
合成两行包的核调用（`text_shapes`：根据编码器配置在 meta 设备上构建的模型类、计划中的量化
线性层、每个张量采用编码器同名张量的 dtype、物化为 stride-0 视图、量化 GEMM 绑定到设备），
在 no_grad 下运行于编码器自己的算子上；已放置的权重从不读取。

它在计划可产生的每个桶的一个目标包上重放调用：

- 最多 `plan.batch` 行、`plan.max_tokens` 词元的包；
- 行数最多为 `min(MAX_TEXT_TOKENS + chat template, encoder max length)`；当训练器知道数据时，
  最多为最长标题估算加模板的长度桶（`longest_row`；`MossDataset.text_tokens_max`，规划器的
  估算列；桶舍入可覆盖略低于分词长度的估算，更长的标题会在其首次编码时调优）；
- `pack_lengths` 构建行长度：对每个（段数、最长、平均）桶，构建一个长行加上若干等长较短行，
  使平均值落入该桶，然后为尚未达到的每个词元桶构建包。

偏斜包（一个长行和许多一词元行）正是填充 PyTorch delta-rule 路径开销最大的场景；工作显存
上限可防止它抬高调优峰值。`tests/inference/test_text_prewarm.py` 在 CPU 上检查规划器覆盖
随机 `plan_groups` 包的桶（反例：仅等长行），以及数据上限约束行数（反例：无上限）。

<a id="multi-process"></a>
## 多进程

**广播。** `tuning.broadcast(src=0)` 在已初始化的进程组上将每个 rank 的表（内存和磁盘）
替换为 rank `src` 的表，然后运行 `tuner.reset()`：已确定的选项和所有通过 `tuner.on_reset`
注册的绑定缓存（kernel-or-compile 块运算、融合线性运行器、注意力、文本编码器、优化器和
量化绑定）都被丢弃，因此实时调用会从共享表绑定。

- 训练器只在 rank 0 上预热，广播后启动；显卡相同时每个 rank 运行相同配置（使用不同显卡的
  rank 没有其卡的行，会自行调优）。
- 文本编码器预热在每个 rank 的首次编码前于该 rank 上运行，不做广播（所有标题都已缓存的
  rank 从不编码）；每个 rank 的文本编码器只编码自己的提示词。
- 预热之后计时的问题（在调优窗口内；适配器和文本编码器预热在每个 rank 上运行）由遇到它们的
  rank 计时。

**达成一致。** 每次重新调优步骤后，每个 rank 上的 `RetuneKernels` 都会运行
`tuner.agree()`：

- 每个 rank 自上一轮以来的测量用一个单元素 all-reduce 计数
  （`positive_anywhere`，在后端设备上），仅当某个 rank 进行了测量时才收集
  （`all_gather_object`），因此预热之后的步骤只需一次小 all-reduce；
- 每个（卡、内核、键）以最低 rank 的测量为准（`agreed`），使用该卡设备的每个 rank 都会确定
  它、存储它，并在胜者替换了不同的本地选择时丢弃其绑定（`adopt`），因此从下一步起每个 rank
  绑定相同配置；
- 单进程（或单 rank 组）跳过交换。

**延迟。** 在两个或更多 rank 的进程组下，步骤内不对任何键计时（否则其同伴会在步骤的集合
通信中等待，直到 NCCL 超时）：

- 从 `decisions.go_live`（第 1 步结束）起，`tuner.defer_if_grouped` 使 `select` 对缺失键
  绑定第一个能构建并运行一次的候选，不计时（构建报错的配置像计时时一样跳过；全部失败则抛出
  `ConfigsFailed`），仅在本进程生效、不存储，记录在 `tuner.DEFERRED` 中并记录一次 INFO
  （`deferred under DDP: first candidate, untimed`）。
- 步间重新调优在 `window.timing_here()` 内计时。
- 样本网格的第一个采样器步骤（仅 rank 0）在进程组下仍保持延迟：其窗口只解除冻结，不解除
  延迟。
- 重新调优只计时 kernel-vs-compile 问题，其调用布局由它排队；被延迟的通用 `select` 键
  （注意力、量化 GEMM、适配器配置）没有可重放的调用，因此在 DDP 下整个运行都停留在其第一个
  候选上，下次运行的预热在其计划覆盖该形状时处理它。

`tests/kernels/tuning/test_tuner_agree.py` 检查延迟、单进程路径和重新调优
（反例：无延迟、在步骤内计时）、合并（反例：最高 rank 获胜）、重新绑定
（反例：保留绑定）以及仅在测量后收集对象（反例：每步都收集）。

<a id="cache-directory"></a>
## 缓存目录

`KOHAKUMOSS_KERNEL_CACHE` 可移动缓存根目录（默认
`~/.cache/kohakumoss_kernels`）。测试会话在任何表读取之前将其指向空临时目录
（`tests/conftest.py`），因此每个测试为每个算子运行一个固定配置，从不读取主机的表；需要
调优的测试会设置 `table` 及其自己的缓存目录。
