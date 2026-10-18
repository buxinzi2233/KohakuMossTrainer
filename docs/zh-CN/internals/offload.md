[English](../../internals/offload.md)

<a id="memory-profiles-and-offload"></a>
# 内存配置档与卸载

`kohakumoss.memory` 将配置档名称一次性转换为 `MemoryPlan`，
`apply_to_backbone`（训练）/ `apply_inference_plan`（`MossPipeline.from_pretrained
(memory=...)`、`sample.py` 中的 `MEMORY_PROFILE`）安装计划指定的引擎。训练循环随后调用
一个在构建时绑定的组合 `BlockRunner`。文本编码器有自己的 `TextEncoderPlan`，由
`apply_to_text_encoder` 放置（见下方“文本编码器”一节及 [text-encoder.md](text-encoder.md)）。

<a id="profiles"></a>
## 配置档

| 配置档 | 检查点 | 激活卸载 | 权重流式传输 | 优化器状态 | 融合步骤 | 梯度卸载 |
| --- | --- | --- | --- | --- | --- | --- |
| `speed` | 无 | 否 | 关闭 | GPU | 否 | 否 |
| `balanced` | 块 | 否 | 环形（16 个常驻） | GPU | 否 | 否 |
| `low` | 块 | 是 | 环形（4 个常驻） | 主机，流式传输至 GPU | 是 | 否 |
| `minimal` | 块 | 是 | 环形（2 个常驻） | 主机，流式传输至 GPU | 是 | 是 |
| `auto` | 上述配置中估算可装入设备空闲内存的最快者（见[内存模型](#memory-model)） | | | | | |

每个配置档都保留计算 dtype；`BASE_PRECISION`（`compute`、`int8`、`fp8`、`quant`、
`fp8_compute`、`mxfp8`）设置任意配置档下的基础精度。目录条目
（`ckpt-ring4-actoff-optstream` 等）覆盖放置轴的所有组合；
[../concepts/profiles.md](../concepts/profiles.md) 列出它们及其估算值。

`optim_state`：`gpu`（fp32 主副本和矩常驻）、`stream`（保存在主机锁页内存中，按参数 /
按块流式传输经过 GPU；更新在 GPU 上运行）、`cpu`（`CPUAdamW`，在主机上更新）。
它仅决定放置方式：8 位状态属于采用 GPU 放置的 `OPTIMIZER`
（`adamw8bit`、`adamw8bit_kernel`）。

`quant` 在支持 fp8 GEMM 的架构（sm89+）上解析为 fp8，其他架构上为 int8。
`MemoryPlan` 的每个字段都可以覆盖（`MEMORY_OVERRIDES`）；`resolve` 在一个错误中列出
所有无效组合。全量微调（`trainable_base`）时，基础权重保留计算 dtype，`ring` 变为
`swap`。推理（`inference=True`）时丢弃激活控制项，流式传输也不需要梯度检查点。

<a id="block-runners"></a>
## 块运行器

| 运行器 | 功能 |
| --- | --- |
| `PlainRunner` | `block(x, ctx)` |
| `CheckpointRunner("block")` | 对整个块使用非重入检查点 |
| `CheckpointRunner("sub-block")` | 为 `block.sub_blocks()` 的每个可调用对象设置检查点（注意力半块、FFN 半块） |
| `CheckpointRunner("selective")` | 保留矩阵乘法 / 注意力输出，重新计算逐元素算子（`register_saved_op` 将自定义内核添加到保留集合） |
| `partial:<f>` | 仅为前 `round(depth * f)` 个块设置检查点 |
| `OffloadRunner` | 块保存的张量进入主机锁页内存 |
| `StreamRunner` | 对块权重进行流式传输（`WeightStream`） |
| `StageRunner` | 流水线阶段的全局块索引（内部运行器在全局索引处绑定） |

<a id="activation-offload"></a>
## 激活卸载

`ActivationOffload` 为每个块安装 `saved_tensors_hooks`。前向期间，超过 `min_bytes`
的保存张量在旁路流上从设备复制到主机（其存储会为该流保持预留，直到复制结束）。反向时，
块 `i` 的首次解包会取回块 `i` 的全部数据包，并在第二条流上预取块 `i-1`。取回操作的
设备缓冲区在消费端（计算）流上分配，复制流先等待计算流。释放后的缓冲区便回到计算流的
分配器池，也就是块 `i-1` 自身工作分配内存的池。在复制流下分配的缓冲区会留在该流的池中，
计算流从不从中取用，因此每步都会再次预留已卸载字节所需的空间。
参数、缓冲区和流式权重槽位通过存储指针排除。构建完成后，从活跃骨干张量、环形槽位和可训练
参数重建排除集合（`MemoryEngines.exclude_params`），因为应用计划之后，构建会替换存储
（全量 FT 将可训练参数转换为 fp32）：否则，已释放 bf16 权重的指针会继续被排除，分配器
日后放到该地址的激活就永远不会卸载。结合块级梯度检查点时，卸载的是检查点保存的块输入。

每次解包都会预取前一个块（数据包已取回后为空操作），因此反向中每个块的数据包会在使用前
提前一个块的计算入队，只有反向的第一个块按需取回。RTX 5090 上，一个异步复制引擎承载
所有主机到设备复制：激活数据包和 `ring` 权重加载在其上依次运行，从不并行（nsys GPU
指标追踪显示只有引擎 0 活跃）。若仅在解包时取回数据包，它会排在下一个块的权重加载之后；
预取则能在所属块运行前完成复制。使用 `ring` 时，一步的复制下限是权重字节加激活字节除以
H2D 带宽（下方 1 MP LoRA 步骤中为 19.18 GB / 28.3 GB/s）。

打包和解包钩子在 `torch.compiler.disable` 下运行：它们进行主机记账（旁路流、事件、
锁页池），无论 autograd 从哪里调用，都作为普通 Python 运行，包括已编译帧后端放弃处理的
eager 算子内部（在无原生 bf16 的 GPU sm75 上，inductor 跳过 bf16 图），否则 Dynamo
会使用符号形状追踪它们。

两类保存张量不复制。叶参数保存的 dtype 转换结果或其视图（无梯度检查点的全量 FT 前向中，
autocast 的权重转换，由线性层以 `w.t()` 保存：参数 `AccumulateGrad` 上的
`ToCopyBackward0` 上的视图节点）会记录为参数、dtype 和视图的形状 / 步幅 / 偏移，
解包时重新转换并构造视图；复制它会在每步将每个权重的一份 16 位副本 D2H 再传回。
解包会检查参数版本计数器，如果自前向以来发生原地修改则抛出错误。同一块中保存两次的张量
（流式线性层和 LyCORIS 旁路模块都保存线性层输入）作为带两个使用者的一个数据包，按存储、
偏移、形状、步幅、dtype 和版本建立键，并且仅在第一次保存的张量仍存活时（弱引用）有效，
因此已释放地址上的新张量永远不会误匹配。

交给复制操作的设备张量会一直保持预留，直到复制在设备上运行（`record_stream`）。没有其他
机制阻止主机前进（环形权重流每块不进行主机同步），因此不受限的主机会在设备执行前就排入
整个前向，使所有块的卸载输入同时预留。`act_offload_inflight`（默认 2）对此施加上限：
每个方向都保留最新几个块的最近复制事件，超出该数量的块在复制之前，先在主机上等待
（`Event.synchronize`）最老块的最后一次复制。于是主机最多领先设备这么多个块，同时仍为
设备留下这么多个块的排队工作。内存估算计入 `max(act_offload_inflight, 2)` 个块的保存
激活（一个反向块及其预取的前驱块）。

RTX 5090、LoRA、在线 VAE 和文本编码、1 MP、6 步
（`tests/memory/real/test_profile_memory.py`），每步 torch 分配器计数器：

| 配置档 | 已分配峰值 GiB | 已预留峰值 GiB，受节奏限制 | 已预留峰值 GiB，不受节奏限制 | tokens / s，受节奏限制 | tokens / s，不受节奏限制 |
| --- | --- | --- | --- | --- | --- |
| `minimal` | 3.96 | 4.45 | 7.08 | 3034 | 3015 |
| `low` | 4.38 | 4.95 | 7.71 | 2981 | 2958 |

不受节奏限制时，`minimal` 预留峰值中的 2.79 GiB 是等待设备到主机复制完成才能释放的块输入。

主机缓冲区来自按 2 的幂划分大小类别的锁页字节缓冲区池（至少 1 MiB）。数据包使用能容纳
它的最小空闲类别，因此较小桶形状会复用较大的缓冲区。H2D 复制入队时，缓冲区返回池中，
但只有该复制完成后才能复用。取回的设备副本与数据包一起保留，直到数据包释放，因此即使反向
读取保存张量的次数多于保存次数，也不会重新读取已回收的主机缓冲区。构建之后，从活跃张量
重建保留（从不卸载）的存储（`exclude_params` -> `rebuild`），因此参数 dtype 改变所
释放的地址不会一直保留。只有数据包比所有空闲缓冲区都大时，才分配新的锁页内存。V100 主机上，
每个 8-32 MiB 缓冲区的锁页分配（`cudaHostAlloc`）会阻塞主机约 10-30 ms，每步 42 次
分配的成本超过该步计算。使用池后，V100 上带卸载的 LoRA `balanced` 步骤耗时 1207 ms，
而无卸载为 1199 ms。

<a id="weight-streaming"></a>
## 权重流式传输

`WeightStream` 将所有流式 `MossLinear` 权重保存在主机（锁页）内存，并维护一个含
`slots` 个设备槽位的环（`STREAM_RESIDENT`）。`enter(i)` 让块 `i` 常驻并预取 `i+1`；
反向让 `i` 再次常驻并预取 `i-1`。只有计算流记录槽位上一个块已使用完毕后，槽位才会被覆盖。
流式线性层（`_StreamedLinear`）只为反向保存其输入，从不保存槽位。其反向再次从槽位读取
权重：块的反向让块常驻（`_Exit`），如果槽位被后续块占用，则重新流式加载它。因此流式块
不需要梯度检查点。不启用梯度检查点时，其激活持续保存并跳过重计算。反向重新流式加载一次
块权重，与启用检查点的重计算相同。可训练块的权重梯度为从保存输入计算的 `g^T x`。
fp8 计算基础权重（`fp8_compute`、`mxfp8`）通过 `_StreamedFp8Linear` 流式传输，
它不保存任何内容：前向是无梯度的 `fp8_linear_infer`，输入梯度（通过流式 `q_t` 副本进行
fp8 计算，或通过 `q` 进行 16 位计算）通过 `fp8_linear` 的后端启动器
（`kohakumoss_kernels.fp8.op.dgrad`）再次从槽位读取各部分。基础权重被冻结，因此不保留输入。

`WeightStream`（`offload/weights/stream.py`）拥有主机副本（`host`、可训练 fp32
主副本 `params`、槽位 `layout`），并安装流式基础权重；具体工作由
`offload/weights/streaming/` 中五个部分完成，针对每条流的设备构建一次：

| 部分 | 职责 |
| --- | --- |
| `SlotRing`（`ring`） | 设备槽位、每个槽位承载的块、跳过活跃块与保留块的 LRU 牺牲者；`ensure` / `prefetch` / `release` |
| `H2DCopier`（`copier`） | H2D 旁路流上的主机 -> 槽位复制（锁页，或经由两组暂存区）、每槽位的 `ready` / `free` 和每块的 `host_ready` 事件；主机上使用普通复制 |
| `SpanTimer`（`timer`） | 启用计时时的复制与复制等待时间段，按步切分（`set_timing`、`cut_timing`、`read_timing`） |
| `GradSink`（`grads`） | fp32 主机梯度和（`host_grad`、`filled`）、通过 `HostAccumulator` 执行 D2H、为 `expose_grads` 暴露 `.grad` |
| `BlockReducer`（`reducer`） | 最后一个微批次的跨 rank 块求和：一个 fp32 扁平缓冲区、旁路流、`reduce` |

设备相关调用（`copier.copy` / `wait` / `mark_free`、`grads.accumulate`）在该部分
构建时绑定，流式权重调用环自身已绑定的方法 `slot` / `ensure` / `prefetch` / `release`。
各部分共享一个 `stats` 字典（加载次数、H2D / D2H 字节、主机同步）。

- **冻结基础权重（`ring`）**：仅主机 -> 设备；量化基础权重流式传输 `q` 和 `scale`，
  使用时反量化。
- **可训练块（`swap`）**：主机上的 fp32 主副本就是参数。最后一个微批次之前，每个权重
  梯度都从设备 -> 主机锁页内存，并在主机工作线程（`HostAccumulator`）上以 fp32 求和，
  不占用 autograd 线程。最后一个微批次中，`StreamedAdamW` 在反向内部逐块执行更新：
  块 `i` 的反向开始时，其状态（主副本、矩、主机梯度和；`optim_state="gpu"` 时只有
  梯度和，状态保持常驻）会预取到一个 3 槽位设备环中；最后一个权重梯度到达后，AdamW 在
  旁路流上运行，将新的主副本 / 矩 / 计算 dtype 权重写回主机。下一次前向中块 `i` 的 H2D
  等待该写回。槽位为每个块参数保存 fp32 梯度和流式 dtype 权重，状态流式传输时还保存
  fp32 主副本和矩行（`offload/optim/stream_optim.py` 中的 `slot_bytes`，内存估算
  使用它）：bf16 常驻状态时每参数 3 x 18 B，流式状态时 3 x 54 B。若块的最后一个梯度尚未
  到达就再次需要其槽位，或在反向中从未对该块执行更新，会抛出错误，而非覆盖其状态行。
  每步某权重的首个梯度复制到主机和，后续梯度相加（`grad_filled` 标记包含该步梯度的和）。
  当 `optim_state="cpu"` 时，和进入主副本的 `.grad`（`expose_grads()`），由
  `CPUAdamW` 更新，再由 `refresh()` 更新流式副本；DDP 下，每块最后一个微批次的梯度加
  主机和成为旁路流上的一个 fp32 缓冲区，归约到块的所有者，并复制到其主机梯度
  （`BlockReducer.reduce_block`）。DDP 下，主副本和流式副本位于节点所有 rank 共用的
  一块主机区域（见 [multi-gpu.md](../guides/multi-gpu.md#shared-host-masters)）。
- **锁页与暂存**：`pinned=False` 保留可分页主机副本，并通过两组交替使用的锁页暂存区
  （每组每权重一个缓冲区）复制每块：重新填充一组之前，主机只等待上次读取该组的 H2D，
  即两次块加载之前的 H2D，因此块 `i+1` 的主机 memcpy 与块 `i` 的 H2D 重叠；可训练环
  还会等待块主副本的写回。

<a id="allocator"></a>
## 分配器

一次运行中每步的已分配内存平稳，但打包批次的 token 数变化，因此缓存分配器每步接收的块
大小集合不同。采用 torch 默认设置时，它拆分缓存段以适配请求，没有合适拆分时预留新段，
于是运行过程中已预留内存（nvidia-smi 显示的数值）增长，而已分配内存不增长。三部分机制让
已预留内存接近已分配峰值，并在各步之间保持稳定：

- **分配器设置**（训练和 `sample.py`，由 `offload/accel.py` 中的 `configure_allocator`
  在运行首次分配前应用，仅 CUDA）：
  - Linux：`expandable_segments:True`。内存段原地扩展和收缩，因此新大小永不需要新段。
  - 其他所有平台（Windows，CUDA 分配器没有可扩展段）：`roundup_power2_divisions:4`。
    每个请求向上舍入到每个 2 的幂区间内四种大小之一，因此多几个 token 的批次会复用相应
    大小类别的缓存块，而非预留新段。单个张量最多可能比请求多占约 19%。

  环境中的 `PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_ALLOC_CONF` 优先。运行开始时记录
  生效设置（`allocator <settings>`）。在 RTX 3090 上关闭可扩展段来模拟 Windows，
  使用用户的 LoRA fp16 设置（`ckpt-ring4-actoff`、20000-token 批次、在线文本 / 潜变量
  编码、60 步）测量：torch 默认设置下已预留内存 7.50 -> 10.51 GiB（缓存空闲块被拆到
  小于所需大小后，为随 token 数变化的 fp8 / LoRA / FFN 反向输出新增 1.2 GiB 段），
  启用舍入后为 7.68 -> 8.00 GiB；两者每步结束后的已分配内存均为 1.2 GiB。预热期间
  调优时的内存不足重试（每次释放整个缓存）由 98 次降至 0 次。
  `max_split_size_mb:256` 更差（7.50 -> 22.9 GiB：不可拆分的大块为每种新大小映射一个段）。
- **旁路流缓冲区一次分配。** 旁路流作为当前流时分配的块属于该流的池，计算流从不从中取用。
  因此卸载引擎只分配一次设备缓冲区并向其中复制：权重环形槽位（`WeightStream`）、
  优化器状态环（`HostStateAdamW`：一个 4 行槽位中保存主副本、矩和 fp32 梯度；
  `StreamedAdamW`：同样四行再加流式 dtype 的已更新权重，常驻状态时仅梯度和权重）、
  梯度卸载暂存环（`GradOffload`）、在主机更新的流式块跨 rank 汇总梯度的 fp32 扁平缓冲区
  （`WeightStream.reduce_with`，在计算流上一次分配），以及在计算流上分配的激活卸载取回
  缓冲区（见激活卸载）。
- **步间保留缓存内存**（见下文）。

<a id="between-step-release"></a>
### 步间释放

训练器在步间保留缓存内存。缓存分配器保留某步释放的块，将其交给下一步同一流的分配；卸载
引擎向上述固定缓冲区复制，因此每步复用前一步预留的空间，已预留内存保持在稳定峰值。内存在
阶段边界归还驱动：构建后、内核与适配器预热后、文本编码器预热后、采样前后（`MossPipeline`
在生成后释放），以及步间重新调优因输入无法容纳而丢弃排队问题之前。内存压力下，分配器本身
会先释放缓存块并重试分配，再报告内存不足。`STEP_RELEASE`（默认 `"none"`）在循环内部
添加释放：`"empty_cache"`（每个优化器步骤）或 N（每 N 步），每次执行设备同步加
`empty_cache`，下一步需付出重新分配的成本。

`kohakumoss.memory.release.CacheReleaseStats` 在 CUDA / XPU 上每步结束后运行，
以释放前预留字节减释放后预留字节，报告该步归还驱动的字节数。绑定 STEP_RELEASE 回调时，
这些数值是该回调释放的内存（在同步 + empty_cache 前后读取 `memory_reserved`，未运行
回调的步为 0）；未绑定时，则是自上一步结束以来分配器预留字节的下降值。它只读取计数器。
指标行包含 `cache_released_gib`（该步归还的 GiB），以及从上一行起平均的
`cache_releases`（发生内存归还的步数比例）和 `cache_release_time`（每步释放的主机秒数）。
未绑定释放时，稳定运行显示零，非零行表示该步内部出现了分配重试或阶段边界
（采样网格、首次编码前的文本编码器预热）。

启用 autocast 时，流式 16 位线性层以 autocast dtype 读取其槽位，因此调用永不会物化
完整精度的权重副本。

<a id="optimisers-and-fused-step"></a>
## 优化器与融合步骤

优化器步骤默认在 GPU 上运行，包括所有卸载控制项下；`CPUAdamW` 是显式选择的主机更新。

- `MasterAdamW`：fp32 master / exp_avg / exp_avg_sq 常驻设备，按参数执行
  `torch._fused_adamw_`，参数接收转换为其 dtype 的主副本。
- `HostStateAdamW`：相同更新，主副本和矩位于主机锁页内存。每个参数状态通过一个 3 槽位
  设备环：H2D、更新和 D2H 各在一条旁路流上。槽位只有在上次写回后才复用，参数状态只有在
  自身上次写回后才加载，下个参数状态按上一步更新到达的顺序预取（第一步：注册逆序）。
  `finish()` 让计算流等待更新。调用者的梯度和有限值标志在更新流上读取，因此在该流上记录
  （`record_stream`）：分配器在更新读取它们之前不会复用其内存。
- `StreamedAdamW`：用于 `swap` 块的逐块 GPU AdamW（见权重流式传输）。
- `CPUAdamW` 是 fp32 主副本和矩位于主机的 `torch.optim.Optimizer`；梯度从设备 ->
  主机锁页内存，更新使用 CPU 上的 `torch._fused_adamw_`，已更新权重从主机 -> 设备。
  `step_param(p)` 更新一个参数。对于设备参数，它在旁路流上排入梯度和有限值标志从设备 ->
  主机锁页内存的复制，然后返回。主机工作线程等待该复制，执行更新，再在第二条旁路流上
  排入新权重的主机 -> 设备复制。`finish()` 等待工作线程并让计算流等待这些复制。因此
  主机更新与其余反向重叠。16 位主机梯度在更新时转换到一个复用的 fp32 主机缓冲区
  （工作线程一个，调用者线程一个）；参数的锁页梯度缓冲区在首次 D2H 时分配，因此读取
  梯度卸载主机和（`use_host_grads`）的参数从不持有该缓冲区。`load_state_dict` 保持
  主副本和矩为主机 fp32 张量（torch 加载器会将优化器状态转换到每个参数的设备和 dtype），
  并将各参数重新绑定到重新加载的组。
- 这四者都接受 torch 风格参数组；每组的 lr / betas / eps / weight_decay 应用于组内
  参数，`param_groups` 保存供调度器修改的实时字典。它们自行将 `decay_to_one` 组应用于
  fp32 主副本（`handles_decay_to_one`），因此会跳过优化器包对它们的步后衰减。
- `FusedStep` 注册梯度累加后钩子，在每个参数梯度完整后立即更新并释放，因此只有一个块的
  梯度存活。`FusedStepBundle.step` 在 `schedule.step()` 前将调度器的优化器标记为
  已更新（`_opt_called`）：反向内部更新就是该优化器的步骤，否则 torch 的 LRScheduler
  会警告调度器先于优化器执行了步骤。

<a id="fused-step-and-non-finite-gradients"></a>
### 融合步骤与非有限梯度

`FusedStepBundle.step_param(p)` 为每个梯度运行一次变换，并将其标志以 `found_inf`
传给更新。变换对梯度遍历两次：先计算缩放梯度的 fp32 范数（其未缩放平方计入该步范数），
再以因子 `inv_scale x
min(1, GRAD_CLIP / G)` 调用 `torch._amp_foreach_non_finite_check_and_unscale_`，
原地反缩放和裁剪，并在遇到非有限值时设置设备端标志。`G` 是上一步的全局梯度范数
（在该步 `step()` 时于设备上计算，跨非有限步骤保留，并保存到检查点）：两种优化器包的
`GRAD_CLIP` 都是全局范数，而融合步骤自身的全局范数只有在每个参数都已经更新后才存在。
运行的第一步不裁剪。带标志梯度保留其值：下列每条更新路径都会在标志被设置时不触碰参数及其
状态，该步范数报告为 NaN。每条更新路径都遵守该标志：

- 卸载 AdamW（`MasterAdamW`、`HostStateAdamW`、`StreamedAdamW`）将其交给
  `torch._fused_adamw_`，标志设置时不改动主副本、两个矩和步数；`CPUAdamW` 跳过更新；
- 使用 `fused=True` 构建的逐参数 torch 优化器以 `found_inf` 接收它
  （参数、矩和设备步数均不变）；
- `AdamW16` 和 `MuonHybrid` 运行 `step_masked(found)`：在工作副本中计算更新，
  每个参数 / 状态张量通过 `torch.where(found, old, new)` 存储；
- 其他所有逐参数优化器（非 CUDA 的 `adamw`、`adamw8bit_kernel`、`prodigy`、
  `schedule_free`）在快照与设备端回写之间运行自身步骤：更新前克隆参数及其设备上的每个
  状态张量，更新后通过 `torch.where(found, old, new)` 恢复。克隆是临时的，一次一个
  参数（其值加其状态），每次更新额外对每个内容读取和写入各一次。
- bitsandbytes 优化器（`OPTIMIZER='adamw8bit'`）直接在该快照内运行逐参数步骤主体
  （`init_state`、`prefetch_state`、`update_step`）。其 `step()` 在每个参数后调用
  `torch.cuda.synchronize()`，会使反向为每个参数停顿一次。V100 上，带 8 位状态和
  `fused_step` 的 LoRA `balanced` 采用该方式为 1575 ms，经 `step()` 则为 2629 ms；
  `low` 分别为 1336 ms 和 6658 ms。

设备无法在不进行主机同步的情况下屏蔽的优化器状态（Python 步数、以主机张量保存的步数、
Prodigy 的 `d` 和 `k` 等组值）在更新前逐参数记录（`training/optim/optimizers/skip.py`，
`host_state`）。`step()` 已经同步一次来读取该步有限值标志；非有限步骤中，它再同步一次
读取逐参数标志，为每个带标志参数回填该记录，并丢弃被跳过更新所创建的状态。有限步骤不再
额外读取。因此，被跳过参数在步末的值及全部优化器状态与之前相同，与 `torch._fused_adamw_`
一致。

非融合优化器包（`training/optim/bundle/base.py`）在 `optimizer.step()` 前读取已达成
一致的有限值标志，非有限时跳过整个步骤，因此其中没有参数或状态发生改变。

因此，梯度非有限的参数不更新；该步其他每个参数已经在反向中更新，并保持更新结果。缩放器
回退，该步报告为非有限：`grad_norm` 为 NaN（与非融合路径一样，不对剩余有限梯度取范数），
`nonfinite_params` 统计带标志参数，数值不稳定转储指出其中第一个。否则，反复出现的非有限
梯度会在运行继续时让这些参数保持冻结，因此 `NONFINITE_PATIENCE`（默认 8，0 = 关闭）
会在损失缩放下限处连续这么多步非有限后（没有动态缩放器的运行中每个非有限步骤都处于下限），
或同一批参数连续这么多步非有限后，停止运行并给出它们的名称。

`step()` 首先运行优化器包的收尾器：计算流等待所有在反向内部更新参数的旁路流
（`GradOffload.finish`、`HostStateAdamW.finish`、`StreamedAdamW.finish`），
然后才读取有限值标志，并让下一次前向读取参数。

<a id="wiring-trainingmemoryoptimizerspy"></a>
### 连接方式（`training/memory/optimizers.py`）

`build_memory_optim` 是将计划转换为优化器的唯一位置，训练器（`build_optim`）和
`MemorySession` 都使用它：

| 计划 | 设备参数 | `swap` 块主副本 | 更新步骤 |
| --- | --- | --- | --- |
| `optim_state='gpu'` | `adamw`（torch AdamW，MemorySession 中也默认使用） | `StreamedAdamW`，常驻状态 | `swap`、`fused_step` 或 `grad_offload` 时在反向内部；否则在 `step()` |
| `optim_state='stream'` | `HostStateAdamW` | `StreamedAdamW`，流式状态 | 同上 |
| `optim_state='cpu'` | `CPUAdamW` | `expose_grads` 后的 `CPUAdamW` | 在 `step()`；启用 `fused_step` 时设备参数在反向内部 |

`resolve` 会为采用 GPU optim_state 的 `swap`（块只能在反向内部更新）和采用 GPU
optim_state 的 `grad_offload`（否则每个求和后的梯度都会在更新前回到设备，峰值包含
所有梯度）设置 `fused_step`，除非覆盖了 `fused_step`；`swap` 与冲突覆盖的组合会被拒绝。
`grad_offload` + `fused_step` + `CPUAdamW` 在设备上完成每个梯度，主机更新从设备
读取（每参数额外一次往返）；无 `fused_step` 时主机更新直接读取主机和。

<a id="host-workers"></a>
## 主机工作线程

`HostAccumulator`（梯度和）和 `CPUAdamW`（主机更新）各自在一个 `HostWorker`
线程上运行主机端任务。`HostAccumulator` 将每个设备梯度复制到所有参数共享的
`STAGING_SLOTS`（4）个锁页暂存缓冲区环的下一个缓冲区，每个缓冲区增长到曾接收的最大
梯度大小，工作线程再将其加入 fp32 和；加法完成后即可复用缓冲区，因此只有工作线程落后
四个梯度时反向才等待。其主机暂存为以梯度 dtype 表示的 4 x 最大梯度，与参数数量无关：
`swap` 且 `GRAD_ACC > 1` 时，主机仅保留 fp32 和（每个可训练块参数 4 B），除此之外
不再为每个参数保留其他内容。线程只持有队列，从不持有其所有者。调用 `close()` 时
（先完成排队任务），或所有者释放时，通过弱引用终结器发送停止哨兵并限时等待线程结束。
运行结束时训练器关闭内存引擎和优化器（`close_memory`），因此在同一进程反复运行
（notebook、扫描、测试会话）不会累积线程。

<a id="gradient-offload-with-accumulation"></a>
## 带累加的梯度卸载

`GradOffload` 为每个设备参数注册梯度累加后钩子：autograd 一旦获得微批次梯度，便在
旁路流上将其从设备 -> 主机锁页内存复制，并释放设备梯度；主机工作线程
（`HostAccumulator`）对 `GRAD_ACC` 个微批次维护 fp32 和。最后一个微批次
（`start(last=True, final=opt.step_param)`）中，每个参数的主机和通过 H2D 流上的
3 缓冲区设备环传回，在更新流上加到新的设备梯度，并由 GPU 优化器当场更新该参数；
`finish()` 让计算流等待更新。主机优化器则直接读取和（`CPUAdamW.use_host_grads`；
仅读取包含本步梯度的和）。流式 `swap` 块通过 `WeightStream` 在主机上累加。

每步某参数的首个梯度复制到其主机和，后续梯度相加；`filled` 标记含本步梯度的和，没有
梯度的参数不获得梯度（`restore` 让 `.grad` 保持 None，与无卸载时一样）。如果某参数
在最后一个微批次未触发钩子，但其和包含之前微批次的梯度（模块 dropout、批次跳过的分支），
则在该次反向后由 `finish_unhooked()` 完成（其和成为梯度，并经过相同最终回调），此时
DDP 归约尚未结束：融合优化器包先运行 `GradOffload.finish`，再运行 `ddp.finish`，
因此迟到参数也参与归约。DDP 下，某些 rank 缺失的梯度在这些 rank 上视为零；所有 rank
都缺失的梯度不交付（每个桶在 all-reduce 中为每个参数携带一个存在计数，仅本地缺失参数时
才在主机上读取），所以与单 GPU 一样不会更新它。

<a id="memorysession-and-parameter-groups"></a>
## MemorySession 与参数组

`MemorySession(model, plan, device, dtype, grad_acc, trainable_base, ...,
param_groups)` 应用计划（`apply_to_backbone`），并通过训练器的 `build_memory_optim`
构建其优化器包，用于训练器之外模型的基准测试和测试。驱动程序为每个微批次调用
`backward(loss, micro)`，每个优化器步骤调用一次 `step()`；`masters(model)`
按名称返回所有 fp32 主副本。

`param_groups` 是基于会话构建前所给模型参数的 torch 风格组（计划替换的参数，例如流式
块权重，按名称跟踪）。每组拆分为设备部分与流式部分，两个部分都保留组超参数及额外键，
因此逐组规则（偏置 / 归一化 / 1-D 参数无权重衰减、衰减到一、LoRA+ 比例）都能传递到
每条优化器路径。每个可训练参数都必须在某个组中。

<a id="invalid-combinations"></a>
## 无效组合

控制项拒绝的组合（带冲突 `fused_step` 的 `swap`）由 `validate` 拒绝；无效计划组合由
`resolve` 拒绝。量化基础权重不能训练（全量 FT）。

<a id="hot-paths"></a>
## 热路径

逐块、逐张量和逐参数路径在构建时特化：运行器通过 `bind(layers)` 绑定到逐块可调用对象
元组，卸载 / 流式 / 梯度引擎在构造时绑定设备或主机的复制和等待函数，`start()` 每个
微批次切换一次钩子主体，`QuantWeight.bind` 将 `linear` 设置为内核或反量化路径。
逐步和构建时选择保留普通 `if`。

<a id="quantized-base"></a>
## 量化基础权重

`quantize_model` 将接受的 `MossLinear` 基础权重替换为 `QuantWeight`
（逐输出行缩放：int8 `round(w / s)`，`s = max|w_row| / 127`；fp8 e4m3 `w / s`，
`s = max|w_row| / 448`）。`QuantWeight.bind` 为设备一次性解析 `quant_linear`
内核；没有内核时每次调用反量化权重。每个 `quant_linear` 条目，包括 eager，都是同一
自定义算子（`kohakumoss_kernels.quant.make_quant_linear`），其 autograd 保存
`q` 和 `scale`，并为输入梯度再次反量化，因此冻结的 int8 / fp8 基础权重永不会从前向
到反向保留 16 位权重副本。`split_quantized` / `attach_quantized` 加载 ComfyUI
fp8 / int8 检查点（`scale_weight` 或 `weight_scale` 键）。文本编码器线性层使用
`kohakumoss.quant.text`（按通道或按组，仅权重量化或 fp8 计算）。

<a id="text-encoder"></a>
## 文本编码器

`apply_to_text_encoder(text_model, TextEncoderPlan, device, dtype)` 是训练
（`build_encoders`、在线文本缓存、采样提示词）、`MossPipeline` 及离线缓存工厂的
唯一入口。它量化线性层（`precision`），解析 `memory="auto"`，再放置模型
（`kohakumoss.offload.encoders.text`）：

| memory | 两轮之间在设备上保留 | 一轮期间 | 每轮主机到设备传输 |
| --- | --- | --- | --- |
| `resident` | 全部 | 全部 | 0 |
| `offload` | 无 | 除嵌入表外的全部 | 所有权重（锁页副本） |
| `ring` | 最终归一化、旋转位置表 | + 每种层类型 `stream_resident` 个槽位的环 | 所有层权重，提前一层预取 |
| `cpu` | 无（在主机上运行） | 无 | 0 |

在 `offload` / `ring` 下，2 GiB 嵌入表永不移动：执行轮次在主机查找行，只发送这些行。
每种放置方式的 `end()` 都在调用者下一项工作之前释放该轮设备权重缓冲区（释放环形槽位，
丢弃卸载副本），因此 DiT 步骤永不与它们同时占用设备；显存时间线
（`vram_te_peak`、`vram_te_after`、`vram_step_peak`）对此记录。
`LayerStreamer` 通过前向钩子（transformers 前向）或 `enter(i)` / `leave(i)`
（内核前向，按层优先处理一轮中的所有打包组，因此流式层每轮只经过总线一次），在进入时
将层张量换入槽位，退出时换回主机副本；结构不同的层使用独立槽位池；`h2d_bytes` 统计复制。

`free_after_use` 在首次执行后丢弃所有权重（空提示词嵌入在该轮编码并保留）。文本离线
缓存时，训练会设置此项：采样提示词在构建时编码一次，随后释放编码器。

内存模型加入文本编码器（`kohakumoss.offload.encoders.text_memory`）：常驻权重计入
DiT 步骤峰值；卸载 / 环形流式传输一轮的峰值是 DiT 持久部分加该轮权重和激活（所有打包组
隐藏状态、一次调用的 MLP 中间结果、delta 规则暂存空间）。`with_text` 和
`auto_plan(text=...)` 使用它；`memory="auto"` 选择能与步骤共存的最快放置方式：
`resident`，否则 `ring`（环形执行受自身峰值限制：DiT 步骤在该轮释放缓冲区后运行）。
永不选择 `offload`：流式传输在前一层计算时预取每层，比 `offload` 的批量复制更快，
且只保留一个环而非所有权重（Qwen3.5-9B，每轮 64 条描述，打包，每调用 32 条：
3090 bf16 为 4328 对 3777 tok/s，V100 fp16 为 4613 对 2735；
单轮峰值 2.9 对 14.2 GiB）。

<a id="vae"></a>
## VAE

`VAE_ENCODE`（`kohakumoss.offload.encoders.vae.VaePlan`：`memory`、`batch`、
`ahead`、`tile`）为在线潜变量编码和采样解码放置 FLUX.2 VAE
（`PlacedVae`，在 `build_encoders` 中由 `place_vae` 构建）：

| memory | 两轮之间 | 一轮期间 |
| --- | --- | --- |
| `resident`（默认） | 权重在设备上 | 权重 + 激活 |
| `offload` | 无（主机锁页副本） | 权重移入，执行后释放 |
| `stream` | 无 | 编码器 / 解码器块经由 2 槽位环，其余部分每轮移入 |
| `cpu` | 主机 | 主机（训练：仅离线潜变量缓存，即采样解码） |
| `auto` | 能与 DiT 步骤预留空间共存的 resident / offload / stream 中最快者 | |

`offload` 在一轮开始时于计算流上从主机锁页副本复制权重，结束时丢弃设备副本。它们在
该流上分配和读取，因此分配器只按该流的顺序将其内存交给后续工作，单轮结束无需设备同步
（与文本编码器的 `offload` 相同）。此时 `vae_time` 是该轮主机时间，与 `resident` 一样。

`tile` 启用 diffusers VAE 的分块与切片编码 / 解码，使高分辨率下激活峰值保持有界。
使用在线潜变量缓存时，`LatentAhead` 在一轮 VAE 中编码接下来 `ahead` 个主机批次的
缓存未命中项，并保留潜变量直到批处理器取用，因此卸载的 VAE 每 `ahead` 批只移动一次，
并在 DiT 步骤前释放。`VaeEncodeStats` 逐步记录 `vae_time`、`vae_share`、
`vae_images` 和 `vae_h2d_gb`；显存时间线将 VAE 执行轮次
（`vram_vae_peak`、`vram_vae_after`）与 DiT 步骤分开标记。
`8g` 和 `12g` 显存配置层卸载 VAE，`auto` 自行选择。

<a id="vae-dtype-compile-and-batching"></a>
### VAE dtype、编译与批处理

`VAE_DTYPE`（训练、`scripts/data/cache_latents.py`、`scripts/sample/sample.py`；
`kohakumoss.encoders.vae.resolve_vae_dtype`）选择 VAE 权重及计算 dtype。
其值为 `auto` | `bf16` | `fp16` | `fp32`。

- `auto` 在支持 bf16 的显卡上为 bf16，其他显卡（V100、2080 Ti）为 fp16；CPU 上为 fp32。
- 一个 dtype 覆盖所有路径：训练的在线潜变量编码与采样解码（均为 `PlacedVae`）、
  离线缓存执行（`vae_encoder`）和流水线解码（`MossPipeline.from_pretrained(vae_dtype=...)`）。
  潜变量始终以 fp32 输出。
- 无论 dtype 如何，潜变量归一化（VAE BatchNorm 的 128 个运行均值和方差）保持 fp32：
  `FluxVae.from_pretrained` 以 fp32 加载 VAE，保留这些统计值，再转换 VAE
  （diffusers 的 `torch_dtype` 会将它们舍入到 VAE dtype，使每个潜变量发生逐通道偏移和缩放）。
- VAE dtype 是潜变量缓存键的一部分（`LATENT_ENCODER_SPEC`，
  [data.md](../concepts/data.md#cache-keys)）。
- 允许 fp16，因为它在真实图像上保持有限（下表）：`FP16_PROBLEM = None`。
  将 `FP16_PROBLEM` 设置为某个原因，会以该原因拒绝 `VAE_DTYPE='fp16'`，
  此后 `auto` 也永不选择 fp16。

相对于 fp32 VAE 的精度：16 张数据集图像，中心裁剪至 1 MP（1024²）和 2 MP（1440²）。
编码对比潜变量，解码从 fp32 潜变量开始对比像素。数值为图像间均值；“最差 cos”取最低的图像。

| GPU | dtype | 面积 | 执行 | 有限 | cos（均值 / 最差） | 绝对误差均值 | 绝对误差中位数 | 相对误差中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 3090 | bf16 | 1MP | 编码 | 是 | 0.999887 / 0.999695 | 8.25e-03 | 5.66e-03 | 9.84e-03 |
| RTX 3090 | bf16 | 1MP | 解码 | 是 | 0.999994 / 0.999988 | 1.96e-03 | 1.61e-03 | 3.22e-03 |
| RTX 3090 | fp16 | 1MP | 编码 | 是 | 0.999983 / 0.999958 | 2.96e-03 | 1.75e-03 | 3.16e-03 |
| RTX 3090 | fp16 | 1MP | 解码 | 是 | 1.000000 / 1.000000 | 2.12e-04 | 1.70e-04 | 3.42e-04 |
| RTX 3090 | bf16 | 2MP | 编码 | 是 | 0.999870 / 0.999676 | 8.83e-03 | 6.05e-03 | 1.08e-02 |
| RTX 3090 | bf16 | 2MP | 解码 | 是 | 0.999995 / 0.999990 | 1.89e-03 | 1.54e-03 | 3.13e-03 |
| RTX 3090 | fp16 | 2MP | 编码 | 是 | 0.999980 / 0.999953 | 3.21e-03 | 1.91e-03 | 3.50e-03 |
| RTX 3090 | fp16 | 2MP | 解码 | 是 | 1.000000 / 1.000000 | 2.12e-04 | 1.71e-04 | 3.47e-04 |
| RTX 5090 | bf16 | 1MP | 编码 | 是 | 0.999887 / 0.999696 | 8.24e-03 | 5.64e-03 | 9.82e-03 |
| RTX 5090 | bf16 | 1MP | 解码 | 是 | 0.999994 / 0.999988 | 1.97e-03 | 1.61e-03 | 3.23e-03 |
| RTX 5090 | fp16 | 1MP | 编码 | 是 | 0.999983 / 0.999958 | 2.96e-03 | 1.74e-03 | 3.14e-03 |
| RTX 5090 | fp16 | 1MP | 解码 | 是 | 1.000000 / 1.000000 | 2.22e-04 | 1.79e-04 | 3.58e-04 |
| RTX 5090 | bf16 | 2MP | 编码 | 是 | 0.999870 / 0.999677 | 8.83e-03 | 6.05e-03 | 1.08e-02 |
| RTX 5090 | bf16 | 2MP | 解码 | 是 | 0.999995 / 0.999990 | 1.88e-03 | 1.54e-03 | 3.13e-03 |
| RTX 5090 | fp16 | 2MP | 编码 | 是 | 0.999980 / 0.999953 | 3.21e-03 | 1.91e-03 | 3.50e-03 |
| RTX 5090 | fp16 | 2MP | 解码 | 是 | 1.000000 / 1.000000 | 2.12e-04 | 1.71e-04 | 3.47e-04 |
| V100 | fp16 | 1MP | 编码 | 是 | 0.999989 / 0.999970 | 2.39e-03 | 1.48e-03 | 2.64e-03 |
| V100 | fp16 | 1MP | 解码 | 是 | 1.000000 / 1.000000 | 2.18e-04 | 1.74e-04 | 3.51e-04 |
| V100 | fp16 | 2MP | 编码 | 是 | 0.999987 / 0.999966 | 2.59e-03 | 1.60e-03 | 2.90e-03 |
| V100 | fp16 | 2MP | 解码 | 是 | 1.000000 / 1.000000 | 2.09e-04 | 1.69e-04 | 3.43e-04 |

- 溢出：fp16 在两种面积、编码和解码的每张图像上都保持有限。其误差比 bf16 低约
  3x（编码）至 9x（解码），因为 fp16 有 10 位尾数，而 bf16 有 7 位。
- 速度：bf16 和 fp16 速度相同。二者都比 fp32 快 1.7-2.3x，使用约 60% 的显存。
- `auto` 在 bf16 显卡上保留 bf16，以获得与 fp32 相同的范围。fp16 在此换来的是精度，
  而非速度。

`VAE_ENCODE['compile']`（`auto` | `on` | `off`）以 channels_last 布局通过
`torch.compile(dynamic=True)` 运行编码和解码（`FluxVae.accelerate`）。`auto`
为不使用分块的常驻 CUDA VAE 启用它。缓存编码器和流水线解码也在 CUDA 上编译
（流水线仅在 VAE 常驻时）。

- 二者作为一对测量。单独 channels_last 比默认布局慢 40-75%；编译后，它在所有 GPU
  的 batch 1 上都是最快变体（1 MP、bf16、每图）：
  - RTX 5090：编码 39 ms，eager 为 59；解码 53，eager 为 79；
  - RTX 3090：编码 109，eager 为 134；解码 153，eager 为 177；
  - V100（fp16）：编码 137，eager 为 143；解码 172，eager 为 193。
- 编译峰值也更低：1 MP 编码时 1.16 GiB，而原来为 1.65。
- fp32 下，编译在 RTX 3090 和 RTX 5090 上持平或胜出，但 V100 上慢 8-10%。
  V100 上使用 `VAE_DTYPE='fp32'` 时，设置 `compile='off'`。`auto` 在 CUDA 上
  永不选择 fp32。

批处理：`encode_images` 将连续的同尺寸图像一起编码，每次调用至多
`VAE_ENCODE['batch']` 张图像和 `BATCH_PIXELS` 个像素。默认 batch 为 1。

- batch 4 的 eager 编码每图快 11-22%，但需要 4x 激活。
- batch 4 的编译编码比 batch 1 每图慢 10-18%。
- 解码从不因批处理获益。

`tile` 在 1 MP 时不改变任何内容，因为分块仅在超过 diffusers 分块尺寸时启用。
2 MP、16 位时，它将峰值保持在 1.4-2.0 GiB，未分块则为 2.7-3.7 GiB，代价是多
19-45% 时间。逐 GPU 速度表见 [performance/vae.md](../performance/vae.md)。

<a id="memory-model"></a>
## 内存模型

`estimate(Workload, MemoryPlan)` 将权重、可训练状态、保存激活、重计算窗口、文本
细化器激活、反量化临时量及 IO 相加，乘以计划碎片因子（`fragmentation(plan)`：1.08，
有旁路流缓冲区时为 1.12），再加分配器之外的 0.88 GiB
（[profiles.md](../concepts/profiles.md#estimated-peak-vram)）。全量 FT 卸载优化器
缓冲区与引擎实际分配一致：`swap` 块的 `StreamedAdamW` 槽位环
（`stream_optim.slot_bytes`）、`optim_state='stream'` 下的 `HostStateAdamW`
状态环（`gpu_adamw.state_ring_bytes`）和梯度卸载暂存环（`grad.grad_ring_bytes`），
后两者均按最大设备参数（`largest_device_param`）定尺寸。FSDP2 下，全量 FT 梯度按分片
计算，除非 `GRAD_ACC > 1` 且 `FSDP_REDUCE='step'`，此时每个 rank 在微批次之间
持有整个模型未分片的梯度（`Workload.sharded_grads`）。EMA 平均值按设备上每个可训练
参数、每个衰减值 4 B 计算（`Workload.ema_decays`）；`swap` 下块主副本及其平均值
位于主机内存，因此只计设备常驻参数。未使用梯度检查点时，每 token 每块保存激活为
`12·D + 3·F` 个元素，块级梯度检查点下为 `D`（块输入）；`act_factor` 缩放块内部
部分（融合内核算子为 1.0，eager 算子实测为 3.6）。`auto_plan` 按速度从快到慢遍历
配置档，采用首个估算可容纳的可运行配置。预算是规划时空闲内存乘以 0.92；空闲内存已排除
CUDA / XPU 上下文，因此每个估算（含上下文）都与预算加其上下文部分比较。训练器和流水线
仅根据估算规划。

MOSS 尺寸的合成块（仅复制 / 计算成本，无模型输出），代码树 a3cfbe0。V100 32 GB
（PCIe gen3 x8），fp16，一个 4096 + 300-token 样本，归一化 + 嵌入器权重可训练，
TileLang 块算子。“步骤峰值”和“常驻”是 torch 分配器的已分配字节计数器
（活跃张量，而非预留量或设备用量；见 [../guides/logging.md](../guides/logging.md#vram)）：

| 配置档 | ms / 步 | tokens / s | 步骤峰值 GiB（已分配） | 常驻 GiB（已分配） | 估算 GiB |
| --- | --- | --- | --- | --- | --- |
| speed | 2513 | 1751 | 26.31 | 9.64 | 34.00 |
| balanced | 3392 | 1297 | 11.36 | 9.63 | 12.87 |
| low | 3667 | 1200 | 6.13 | 4.76 | 7.09 |
| minimal | 5446 | 808 | 3.61 | 0.53 | 2.54 |

该次运行的 `minimal` 行使用 int8 基础权重，每次调用反量化，受主机设备复制限制：
每块 int8 权重每步经 PCIe 传输两次，前向与反向各一次，另加激活数据包。`minimal`
配置档本身保持 `base_precision="compute"`（`kohakumoss.memory.profiles`）；
int8 基础权重是 `BASE_PRECISION` 选择，在 sm70+ 上运行 TileLang 的 int8
`quant_linear` 内核（`kohakumoss_kernels.backends.tilelang.quant`）。

V100 32 GB 和 RTX 4090 上 MOSS 尺寸合成块的同样基准（每个控制项单独使用、硬件极限
检查、复制带宽）：[performance/offload-synthetic.md](../performance/offload-synthetic.md)。

<a id="real-model-benches-rtx-5090"></a>
### 真实模型基准（RTX 5090）

三个卸载基准（`scripts/bench/offload/{stream,knobs,limits}.py`）在真实 MOSS-Image
基础权重和真实数据上运行训练器本身，每个案例一个进程：除全量 FT 案例外均为 LoRA
rank 16，bf16，每微批次一张 1 MP 图像（约 4000 图像 token），离线潜变量及文本缓存，
先预热 3 步再测量 8 步。每个案例报告测量步骤的中位步耗时及最小-最大值、整个运行中的
 torch 已分配峰值和 nvidia-smi 设备峰值。代码树内容 id 7209a988（2026-10-18），
RTX 5090 32 GB、Linux。

复制带宽（`stream.py`）：

| GB/s | 锁页 H2D | 锁页 D2H | 可分页 H2D | 可分页 D2H |
| --- | --- | --- | --- | --- |
| 4 MB | 18.44 | 25.79 | 10.13 | 0.88 |
| 64 MB | 26.04 | 26.32 | 10.52 | 0.89 |
| 256 MB | 26.27 | 26.33 | 10.53 | 0.89 |

流式传输与激活卸载（`stream.py`，GRAD_ACC 1）：`speed` 配置档加各案例的
`MEMORY_OVERRIDES`（`stream.py` 中的 `CASES`）。除 `resident` 外，所有案例
均使用块级梯度检查点，`ring` 案例保持 `stream_resident=2`：

| 案例 | ms / 步（最小-最大） | tokens / s | 峰值 GiB（alloc） | smi GiB | H2D GB / 步 |
| --- | --- | --- | --- | --- | --- |
| resident（checkpoint none） | 内存不足 | | | | |
| 块检查点 | 818 (785-844) | 4884 | 12.02 | 12.84 | 0 |
| 检查点 + 激活卸载 | 831 (784-1117) | 4775 | 10.94 | 11.83 | 1.06 (D2H 1.06) |
| 检查点 + `ring` 流式传输，锁页 | 918 (875-1170) | 4361 | 3.27 | 4.28 | 18.12 |
| 检查点 + `ring` 流式传输，暂存 | 1055 (1006-1389) | 3767 | 3.27 | 4.25 | 18.12 |

`speed` 配置档下的检查点粒度（`knobs.py`，GRAD_ACC 2，每步约 8000 token）：

| 检查点 | ms / 步（最小-最大） | ms / 1k tokens | 峰值 GiB（alloc） | smi GiB |
| --- | --- | --- | --- | --- |
| none | 内存不足 | | | |
| block | 1622 (1392-1924) | 205 | 11.86 | 12.80 |
| sub-block | 1655 (1435-6024) | 207 | 12.69 | 13.63 |
| selective | 1485 (1422-1806) | 189 | 23.28 | 24.08 |
| partial:0.5 | 1413 (1182-4208) | 176 | 25.14 | 26.64 |

该代码树未运行全量 FT 控制项行（设备 / CPU AdamW、梯度卸载、batch 1 和 4 下的
`swap` + 梯度卸载 + 流式状态）；V100 与 RTX 4090 上的合成块数值见
[offload-synthetic.md](../performance/offload-synthetic.md#each-vram-knob-in-isolation)。

<a id="hardware-limit-check"></a>
### 硬件极限检查

`scripts/bench/offload/limits.py` 在独立进程中运行每个组合（块级梯度检查点、
`stream_resident=2`），并将步耗时除以 `max(copy bound,
compute)`。复制界限为以下三者的最大值：H2D 字节 / 锁页 H2D 带宽、D2H 字节 / 锁页
D2H 带宽、全部字节 / 实测并发（双向）带宽。计算为相同 GRAD_ACC 下的常驻步骤，
若常驻步骤无法容纳则采用 MMA FLOP 界限。1.00 = 该步骤仅花费其界限成本。

适配器（LoRA）案例，GRAD_ACC 1：`balanced` 配置档加 `limits.py` 的 `case_knobs`
中每个案例的 `MEMORY_OVERRIDES`（块级梯度检查点、`stream_resident=2`、案例基础
精度及 `CASES` 中的控制项）：

| 案例 | ms（最小-最大） | 复制界限 ms | 重叠 | H2D / D2H GB | 峰值 GiB（alloc） | smi GiB |
| --- | --- | --- | --- | --- | --- | --- |
| resident | 813 (765-1046) | 0 | 1.00 | 0 / 0 | 11.69 | 12.74 |
| 激活卸载 | 828 (780-1121) | 42 | 1.02 | 1.06 / 1.06 | 10.94 | 11.83 |
| `ring` | 913 (875-1138) | 639 | 1.12 | 18.12 / 0 | 3.27 | 4.28 |
| `ring` + 梯度卸载 | 907 (894-1196) | 639 | 1.12 | 18.12 / 0 | 3.27 | 4.12 |
| `ring` + 激活卸载 | 1181 (1083-1499) | 676 | 1.45 | 19.18 / 1.06 | 2.50 | 3.42 |
| `ring` + 激活 + 梯度卸载 | 992 (960-1279) | 676 | 1.22 | 19.18 / 1.06 | 2.50 | 3.27 |
| `ring`，int8 基础权重 | 951 (901-2850) | 320 | 1.17 | 9.07 / 0 | 3.02 | 4.05 |

极限检查进程内的带宽：锁页 H2D 28.36 GB/s、D2H 28.28 GB/s、双向合计 50.54 GB/s。
bf16 MMA 254 TFLOP/s（8192 x 8192 x 8192 GEMM）。常驻适配器步骤的 MMA FLOP
界限为 570 ms。

- 激活卸载相对常驻步骤增加 2% 成本：每步每方向 1.06 GB 与计算重叠。
- 单独 `ring` 以及配合梯度卸载时，均为计算耗时的 1.12 倍。
- `ring` + 激活卸载是最慢的适配器案例（重叠 1.45），比再加入梯度卸载的同一案例（1.22）更慢。
- 该代码树未运行全量 FT 极限案例（常驻、梯度卸载、流式状态、CPUAdamW、常驻 / 流式
  状态的 `swap`、GRAD_ACC 4）；其 V100 和 RTX 4090 合成块数值，以及复制受限和
  主机受限读数，见 [offload-synthetic.md](../performance/offload-synthetic.md#hardware-limit-check)。

<a id="windows-and-xpu"></a>
## Windows 与 XPU

引擎从设备模块（`offload/accel.py`：`torch.cuda` 或 `torch.xpu`）获取流和事件，
因此 CUDA 和 Intel XPU 运行同样的旁路流路径；其他设备运行同步主机路径。

| 功能 | Windows（CUDA） | XPU（Arc B50、Windows） |
| --- | --- | --- |
| 内存配置档、估算、`auto` 规划器 | 可用 | 可用 |
| 块运行器（所有检查点模式） | 可用 | 可用 |
| 激活卸载 | 可用 | 可用 |
| 权重流式传输，锁页与暂存 | 可用 | 可用 |
| 梯度卸载 + 累加 | 可用 | 可用 |
| GPU AdamW（常驻、流式状态、流式块）、CPUAdamW、融合逐参数步骤 | 可用 | 可用 |
| `MemorySession`、参数组 | 可用 | 可用 |
| 量化基础权重（int8 / fp8）、流式量化基础权重 | 可用（sm89 上支持 fp8） | int8 |
| 文本编码器放置（resident / offload / stream / cpu）、提前编码 | 可用 | 可用 |
| 推理计划（`apply_inference_plan`） | 可用 | 可用 |
| 多 GPU（DDP、FSDP2、流水线、线程式） | 设计上仅限 Linux（控制项校验会拒绝） | 同左 |

XPU 列来自 B50 上的 `tests/memory/offload`（激活卸载、流式传输、梯度卸载、各优化器
单独运行及在会话中运行）。
