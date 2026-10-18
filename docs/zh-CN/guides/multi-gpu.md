[English](../../guides/multi-gpu.md)

<a id="multi-gpu-training"></a>
# 多 GPU 训练

在训练配置中设置 `PARALLEL`；启动器会为 `DEVICES` 中的每个条目启动一个进程
（见[分布式启动](../internals/distributed-launch.md)）。

多 GPU 训练按设计仅支持 Linux：在 Windows 上，配置项验证会拒绝
`"none"` 以外的任何 `PARALLEL`，以及 `DEVICES` 中多于一个的条目。

| `PARALLEL` | 引擎 | 适用情况 |
| --- | --- | --- |
| `ddp` | `GradReducer` / `AsyncReducer`：对可训练梯度执行分桶 SUM all-reduce，与反向传播重叠，支持所有内存配置档 | LoRA / LyCORIS（只有适配器梯度经过通信链路）；使用权重流式传输的全量微调 |
| `fsdp` | FSDP2 对每个条件线性层和每个块（联合块及文本精炼块）执行 `fully_shard`，然后处理根模块，每个参数保留其自身 dtype；适配器随其适配的模块一起分片 | 单张显卡无法容纳的全量微调 |
| `pipeline` | 在连续块区间上划分 `torch.distributed.pipelining` 阶段 | 没有高速互连的显卡，或单张显卡无法容纳的模型 |
| `threaded` | 单进程，阶段位于不同设备上 | 混合 GPU（如 3090 + 2080 Ti） |

并行层：`configs/moss/parallel/parallel_{ddp,fsdp,pipeline,threaded}.py`。

<a id="run"></a>
## 运行

```bash
# adapters over two GPUs (the ddp layer); --algo full gets fsdp where the sharded
# footprint fits the card, else ddp over the swap profile (full-finetune.md)
kohakumoss-config new train --weights models/MOSS-Image-Base --data <dataset> --gpus 2 --name my-lora
kogine run scripts/train/moss.py --config configs/user/my-lora.py

# a shipped recipe (set WEIGHTS / DATASETS in a file that use_config's it)
kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_ddp.py
```

- 对于 `pipeline` 或 `threaded`，在训练配置的显存层之后，使用 `use_config`
  组合 `configs/moss/parallel/parallel_pipeline.py`（或 `parallel_threaded.py`）
  （[configs.md](configs.md)）。
- `kogine run` 本身就是启动操作：训练脚本会自行为每个设备启动一个进程。
  `DIST_BACKEND="auto"` 在 CUDA 上选择 nccl。
- `scripts/dist/check_comm.py` 可在长时间运行前检查进程组并测量每条链路。

<a id="supported-memory-plans"></a>
## 支持的内存计划

配置项验证会解析 MemoryPlan，并对表格范围之外的每种组合各给出一条消息并拒绝：

| `PARALLEL` | stream | optim_state | base_precision | act_offload | fused_step / grad_offload |
| --- | --- | --- | --- | --- | --- |
| `none` | 任意 | 任意 | 任意 | 任意 | 任意 |
| `ddp` | 任意（`swap`：单节点，共享主机主权重） | 任意 | 任意 | 任意 | 任意 |
| `fsdp` | `off` | `gpu` | `compute` | 否 | 否 |
| `pipeline`、`threaded` | `off` | `gpu` | `compute` | 否 | 否 |

- `ddp` 可运行所有能在单 GPU 上运行的内存配置档（`ckpt-ring*`、`*-gradoff`、
  `*optfused*`、`*-optcpu`、`*-optstream`、`low`、`minimal`、全量微调 `swap`
  配置档）；每个可训练梯度都会在优化器读取之前跨 rank 求和
  （[提前离开反向传播的梯度](#gradients-that-leave-backward-early)）。
- `fsdp` 不支持训练期间采样（`SAMPLE_EVERY=0`）：采样前向仅在 rank 0 上运行，
  无法对分片块执行 all-gather。验证则在每个 rank 上运行。
  `fsdp` 下拒绝 `checkpoint='sub-block'`（块的各部分会绕过块 `__call__`
  上的 FSDP2 解分片）。
- FSDP2 检查点保存完整张量：每个 rank 收集可训练 / EMA 张量和优化器状态
  （`get_optimizer_state_dict(full_state_dict=True)`），由 rank 0 写入；
  恢复时重新分片。最终导出会在每个单元均解分片的状态下运行，
  并读取当前参数。

<a id="loss-normalisation"></a>
## 损失归一化

每个 rank 的损失都除以该步骤的全局 token 数，因此梯度跨 rank 求和，
日志中的损失是各 rank 所占份额的总和：

- `GradReducer` 执行求和；`fsdp.shard` 将 FSDP2 除数设为 1，并强制使用普通
  SUM 集合通信。
- 在 DDP 和 FSDP2 下，该步的损失份额与非有限值标志合并在一次 SUM all-reduce 中。
- 在流水线下，每个阶段看到相同的步骤：在最后一个阶段，每个微批次的损失
  都除以该步骤的 token 数，微批次损失求和（调度器以 `scale_grads=False`
  运行），该步损失由持有最后阶段的 rank 广播
  （V 型调度下为 rank 0）。

在 FSDP2 且 `GRAD_ACC > 1` 时，`FSDP_REDUCE` 选择何时对梯度执行
reduce-scatter：

| `FSDP_REDUCE` | reduce-scatter | 微批次之间保留的梯度 |
| --- | --- | --- |
| `"step"`（默认） | 仅在最后一个微批次（`set_requires_gradient_sync`）；每步一次 | 整个模型未分片的梯度（fp32 主权重每个参数 4 B；内存估算包含它们） |
| `"micro"` | 每个微批次之后；每步 `GRAD_ACC` 次 | 仅分片后的梯度 |

<a id="data-parallel-ddp"></a>
## 数据并行（`ddp`）

`GradReducer` 将可训练参数（逆序、相同 dtype 和设备）分组到
25 MiB 的桶中：

- 在该步骤的第一个微批次，每个桶缓冲区清零并成为其参数的梯度
  （每个 `.grad` 都是缓冲区的视图），因此 autograd 在桶内原地累加。
- 在最后一个微批次，桶内最后一个参数的累加后钩子会启动其异步 SUM
  all-reduce，与其余反向传播重叠；`finish`（在优化器更新步骤前）
  使计算流等待每个桶。
- 如果某个梯度被其他逻辑替换为独立张量，则在启动前将其复制回桶中。
- 桶按桶顺序启动（已完成的桶会等待前面的桶），因此即使某个 rank 上
  缺少一个梯度，每个 rank 也会以相同顺序发出相同的集合通信。

<a id="gradients-that-leave-backward-early"></a>
### 提前离开反向传播的梯度

当内存计划在优化器更新步骤之前移动或消费梯度时，
`AsyncReducer`（`kohakumoss.parallel.ddp`）会先在其独立旁路流上对其求和，
使归约与其余反向传播重叠。在最后一个微批次：

| 计划 | 梯度去向 | 归约 |
| --- | --- | --- |
| stream 为 `off` / `ring`，无融合更新步骤，无梯度卸载 | 留在设备上直至更新步骤 | `GradReducer` 桶（见上文） |
| `fused_step` | 反向传播内部的逐参数更新步骤 | 累加后钩子将梯度复制到其静态 `AsyncReducer` 桶（逆序、单一 dtype、25 MiB）；完整的桶会异步执行 all-reduce，完成后其参数在旁路流上更新 |
| `grad_offload`（+ `fused_step`） | 主机 fp32 跨微批次求和 | 最后一个微批次将主机累加和加到设备梯度上（与单 GPU 一样），然后执行桶归约；求和后的梯度随后在反向传播内更新（融合模式），或替换主机累加和（在旁路流上 D2H），供主机 / 设备优化器使用 |
| `swap` + `optim_state` 为 `stream` | 反向传播内部在块所有者上执行逐块 `StreamedAdamW` 更新步骤 | 每块的 fp32 梯度和（前面微批次的主机累加和 + 当前微批次）在每个环形槽位中形成一个展平缓冲区，在更新流上归约到块所有者；仅由所有者变换和更新它（[共享主机主权重](#shared-host-masters)） |
| `swap` + `optim_state` 为 `gpu` | 反向传播内部在每个 rank 上执行逐块 `StreamedAdamW` 更新步骤（状态常驻） | 同一个展平缓冲区，执行 all-reduce；每个 rank 更新其设备状态，所有者写入共享主机主权重和流式传输副本 |
| `swap` + `optim_state='cpu'` | 块所有者的主机 AdamW 对主机主权重执行更新 | 每块梯度（主机累加和 + 当前微批次）在旁路流上收集到一个 fp32 展平缓冲区中，归约到所有者，并复制到其主机梯度 |

前面的微批次与单 GPU 完全相同地累加（原地或在主机上）；只有最后一个
微批次的总和经过通信链路。

<a id="reduction-dtype"></a>
### 归约 dtype

`DDP_REDUCE_DTYPE="fp32"`（默认，也是 FSDP2 遵循的规则）以 fp32 求和：

- 16 位桶在启动时复制到 fp32，求和完成后复制回去；
- 提前离开反向传播的 16 位梯度进入 fp32 桶，并以其参数的 dtype 返回；
- 流式传输块的展平缓冲区本来就是 fp32。

跨多个 rank 进行 16 位求和时，精度会随 world size 增大而损失；经过损失缩放的
fp16 梯度即使在每个 rank 上都没有溢出，也可能在求和时溢出。16 位桶的 fp32
副本从启动一直存活到该步骤的归约结束（在反向传播结束时同时容纳所有
16 位可训练梯度；可训练适配器和 fp32 主权重不需要此副本）。

`DDP_REDUCE_DTYPE="param"` 以每个参数自身的 dtype 求和。所有集合通信均在
设备上运行（NCCL）。

<a id="step-end"></a>
### 步骤结束

`finish`（在更新步骤之前）：

- 启动仍未完成的桶。本 rank 缺失的梯度视为零；所有 rank 都缺失的梯度不会
  交付，因此也不会更新，与单 GPU 一样：每个桶会在 all-reduce 中携带逐参数
  存在计数，仅当桶中有本地缺失的参数时才在主机上读取；
- 使计算流等待旁路流和更新流。该等待即 `ddp_comm_wait_ms`
  （包含排在最后一次归约之后的反向传播内部更新）；`ddp_span_ms` 从最后一个
  微批次开始计时，直到该等待结束，`ddp_reduce_mb` 则统计该步归约的每个字节。

由主机执行更新的梯度（`optim_state='cpu'`、梯度卸载）会在每个更新步骤
统一同步到主机一次。初始广播覆盖可训练的设备参数（rank 0 的值）；
流式传输的主权重本身就只有一个共享副本。

<a id="stock-torch-ddp"></a>
### 原生 torch DDP

`DDP_BACKEND="torch"` 在原生 torch `DistributedDataParallel`
（`wrap_torch_ddp`）下运行步骤，而非使用本项目的归约器，作为 A / B 基线：

- 训练步骤调用包装器，`no_sync()` 覆盖除最后一个之外的所有微批次，
  且不会在每次前向时广播缓冲区；
- 一个通信钩子将其桶归约改为 SUM（使用 `DDP_REDUCE_DTYPE`），与本项目一致；
- 它仅归约留在设备上直至更新步骤的梯度：配置项验证会拒绝将它与
  `fused_step`、`grad_offload`、流式传输的基础模型，或
  `DROPOUT['module'] > 0` 组合使用（未使用的参数会使 torch DDP 的归约器停止）。

<a id="shared-host-masters"></a>
### 共享主机主权重

在 DDP 且 `stream='swap'`（单节点、Linux）时，各 rank 共享一份 fp32
主机主权重及其计算 dtype 的流式传输副本：

- Rank 0 创建一个 `/dev/shm` 区域（`kohakumoss.offload.weights.shared.SharedHost`，
  通过 `cudaHostRegister` 锁页），并用加载的权重填充；其他 rank 在步骤循环
  之前映射该区域，所有 rank 都持有映射后即解除文件链接。
- 块 `b` 属于 rank `b % world`（`block_owners`）：它的梯度归约
  （非 all-reduce）到该 rank，仅由该 rank 持有主机优化器状态并执行更新
  （主机上的 ZeRO-1），再将新的主权重和流式传输副本写入共享区域。
- 每步回写后，所有 rank 都在屏障处会合，因此下一次前向会从共享区域读取
  新权重。
- 全局梯度范数会跨 rank 累加各所有者的范数平方（`ShardedNorm`、
  `sum_block_stats`）；检查点在 `optim.<rank>.pt` 中保留每个 rank 的优化器分片。

全量微调流式传输的主机内存，以每个流式传输块参数的字节数计，单节点 W 个 rank
（16 位流式传输副本、GRAD_ACC = 1）：

| `optim_state` | 单 GPU | DDP 每个 rank（共享区域 + 自身分片） | DDP 整个节点 |
| --- | --- | --- | --- |
| `stream`（`StreamedAdamW`） | 主权重 4 + 副本 2 + 矩估计 8 = 14 | 6 共享 + 8 / W | 14 |
| `cpu`（`CPUAdamW`） | 6 + 优化器 20（主权重、矩估计、梯度、影子副本）+ 主机梯度 4 = 30 | 6 共享 + 24 / W | 30 |
| `gpu`（状态常驻） | 6 | 6 共享 | 6 |

当 `GRAD_ACC > 1` 时，每个 rank 还会在最后一个微批次归约之前，在主机上
累加自身微批次的块梯度（每个 rank 为 4，所有块）。rank 报告会记录每个 rank 的
`host_rss_gib`、`host_private_gib` 和 `host_shared_gib`，以及作业的
`host_ram_gib`（私有部分求和，共享部分只计一次）。

<a id="fsdp2-fsdp"></a>
## FSDP2（`fsdp`）

`fsdp.shard(backbone, block_units(backbone), conditioning_units(backbone))` 将每个
条件线性层（`time_embedding.0`、`time_embedding.2`、`shared_block_modulation`、
`video_head_modulation`）、每个文本精炼块和每个联合块各设为一个单元，
然后处理根模块（嵌入器、输出头）。

dtype：

- 每个参数都以其自身 dtype 收集，即单 GPU 运行时用于计算的 dtype：16 位冻结
  基础模型、fp32 KEEP_FP32 条件模块、适配器及全量微调权重的 fp32 主权重
  （`OPTIMIZER='adamw16'` 下为 16 位）。梯度以 fp32 归约。
- FSDP2 对每个单元仅以一种梯度 dtype 执行 reduce-scatter，因此条件线性层
  在 `adamw16` 下持有 fp32 参数，而根模块参数为 16 位时，它们会成为独立
  单元（适配器位于其适配模块的单元中）。出于同样原因，`PARALLEL='fsdp'`
  配合 `adamw16` 时，`KEEP_FP32` 中仅允许这些条件模块。
- 持有多种 dtype 参数的单元以字节方式 all-gather（FSDP2 的 uint8 路径）。
  对全量微调权重，以 fp32 收集 fp32 主权重的 all-gather 字节数是
  16 位收集的两倍；适配器主权重很小。
- 前向输入保留其 dtype（`cast_forward_inputs=False`：sigma 保持 fp32，fp32
  残差流保持 fp32）。

重叠与归约：

- 根模块和条件单元在前向与反向之间保留已收集的参数；根模块预取条件单元和
  第一个块，每个块在前向中预取下一块的 all-gather，在反向中预取前一块的
  all-gather（`prefetch=1`）。
- 梯度使用普通 SUM（`set_gradient_divide_factor(1.0)` 配合
  `set_force_sum_reduction_for_comms(True)`），这是 NCCL 的零拷贝归约，
  也是 gloo 实现的方式。

<a id="pipeline-pipeline"></a>
## 流水线（`pipeline`）

<a id="memory"></a>
### 内存

每个 rank 在其设备上构建完整模型，保留自己阶段的层，并将其他层
（以及没有本地参数的顶层模块）移到主机内存。

- 检查点在 rank 0 上合并所有 rank 的可训练 / EMA 条目，并保留每个 rank 的
  优化器状态。每个 rank 在 `gather_object` 前将其张量移到主机。
- 最终导出会先将每个 rank 训练后的阶段条目合并到 rank 0 当前的可训练集合
  （`PipelineParts.before_export`，一次集合通信），因此导出的适配器 /
  合并权重包含所有阶段的训练结果。

<a id="schedules"></a>
### 调度

`PIPELINE_SCHEDULE`（默认 `1f1b`）：

| 调度 | 每个 rank 的阶段数 | `MICROBATCHES` |
| --- | --- | --- |
| `gpipe`、`1f1b` | 一个 | 至少等于 rank 数 |
| `interleaved-1f1b`、`looped-bfs`、`interleaved-zb` | 两个（循环式：rank `r` 持有阶段 `r` 和 `r + W`） | 至少等于 rank 数；`interleaved-1f1b` / `interleaved-zb`：为轮数 `max(1, MICROBATCHES // W)` 的整数倍 |
| `zbv`、`dualpipe-v` | 两个（V 型：rank `r` 持有阶段 `r` 和 `2W-1-r`） | 至少等于 rank 数（`dualpipe-v`：两倍） |

- `gpipe` / `1f1b` 的气泡占该步骤的 `(W-1) / (MICROBATCHES + W - 1)`；`1f1b`
  在第一阶段最多保留 `W` 个微批次的激活，而 `gpipe` 会保留全部。
- 零气泡调度（`interleaved-zb`、`zbv`、`dualpipe-v`）将阶段的反向传播拆成
  输入梯度执行和稍后的权重梯度执行，共用一个保留的计算图，因此这些调度下
  编译的反向传播会保留缓冲区（`retained_backward`，在第一次编译前设置）。
- 保留的计算图还会将每个使用梯度检查点的块在输入梯度执行中重计算的激活保留到
  权重梯度执行，因此这些调度在每个微批次的两次执行之间保留完整的
  （未使用梯度检查点的）激活：在 24 GB 显卡上运行 MOSS LoRA 时，`zbv` 在
  4608 和 9216 token 的微批次下会内存不足，而 `1f1b` 的峰值为 11 GiB。

<a id="what-crosses-a-stage-boundary"></a>
### 跨越阶段边界的内容

- 每个 rank 抽取相同的步骤（rank 0 的数据、sigma 和噪声流），并自行构建
  每个微批次的布局；第一步会检查各 rank 是否一致（尺寸和 sigma 位模式，
  一次性 all-gather），否则在每个 rank 上抛出错误。
- 只有 `(x, modulation, head modulation)` 跨越边界。`x` 使用残差流的 dtype
  传输（fp32-residual 修正下为 fp32）；调制仅在条件参数参与训练时
  （全量微调、`+modulation` 适配器）才携带梯度，因此 LoRA 运行中唯一计算并
  跨边界回传的梯度是 `x` 的梯度。
- 微批次索引作为调度器的关键字参数（`kwarg_mbs`）到达每个阶段，最后阶段的
  损失将其作为目标（`target_mbs`），因此每个阶段都从主机获取索引；
  调度器以 `return_outputs=False` 运行（不合并输出）。
- 每个微批次的适配器步骤状态（T-LoRA 掩码、DyLoRA 秩、模块 dropout 抽样）
  每步按微批次顺序，基于其填充后的布局抽取一次，并在其每次分块前向和反向
  之前重新安装（`TimedStage.on_chunk`），因此梯度检查点重计算看到的状态
  与前向时相同。

<a id="static-shapes"></a>
### 静态形状

每个微批次都填充到流水线的 token 预算和样本数：

- 填充被拆成最多 `PipeShape.pad_segment`（512）个 token 的零权重样本，
  因此填充的注意力开销保持较小，调制的样本行数为
  `MAX_SAMPLES + ceil(MAX_TOKENS / 512)`。
- `PipeShape.from_knobs` 从数据配置项（`MAX_TOKENS` 联合 token、`MAX_SAMPLES`
  样本、`MICROBATCHES`）获得形状，并拒绝使这些值不再是规划器硬上限的配置项。
  它要求 `BATCHING='tokens'` 或 `'tokens_ffd'`、`MAX_SAMPLES >= 1`、
  `TOKEN_BUDGET='exact'` 和 `TEXT_CACHE='offline'`。
- 会报告填充 token 的占比（`pipe_pad_frac`）；首次适应打包
  （`tokens_ffd`）以及能容纳整数个典型样本的 `MAX_TOKENS` 可降低它。
- token 丢弃（`OVERLONG='drop_tokens'`）通过 `MicroBatch.img_keep` 传递每个样本
  保留的网格索引；保留的 token 维持其网格位置。

<a id="stage-split"></a>
### 阶段划分

`PIPELINE_SPLIT`（默认 `"measured"`）选择层区间
（`kohakumoss.parallel.pipeline.split.stage_bounds`）：

| 值 | 成本 |
| --- | --- |
| `"measured"` | rank 0 在由典型样本填满形状的微批次上，对每个块（通过本次运行的梯度检查点执行器）、前序部分（条件模块、嵌入器、文本精炼块）和后序部分（输出头 + 损失）的前向 + 反向计时；成本会广播，使每个 rank 构建相同的阶段 |
| `"flops"` | 对相同三部分使用解析 FLOP 模型 |
| `"even"` | 每个块成本相同，不考虑前序 / 后序部分 |
| `(n0, n1, ...)` | 显式指定每个虚拟阶段的层数 |

- 划分是在连续切分点上的精确动态规划，使最慢阶段的成本（其层，加上首阶段的
  前序部分和末阶段的后序部分）除以其 rank 速度的值最小；相同时，按各阶段
  参数量平方和（内存）决胜。
- 相同设备上的 rank 速度为 1；混合设备使用实测的稠密矩阵乘法速率。
- V 型放置先按 rank 划分（rank 0 承担前序和后序部分），然后将各 rank 的层
  均分到其两个阶段。
- 划分表会在启动时记录。

<a id="threaded-pipeline"></a>
## 线程式流水线

`ThreadedPipeline` 将层区间放置在单个进程的各设备上：

- 内核启动在每个设备上都是异步的，因此一个线程即可使阶段重叠；
  `schedule="async"`（`PIPELINE_SCHEDULE`，`"gpipe"` 或 `"async"`）
  在辅助线程上运行各微批次的反向传播，同时发出下一次前向。
- 每个阶段可绑定自身的算子集（`opset_for(device)`），因此 3090 阶段运行
  Triton 内核，2080 Ti 阶段运行 TileLang 内核。

<a id="measuring-a-run"></a>
## 测量一次运行

<a id="step-breakdown"></a>
### 步骤分解

每个引擎通过设备事件记录自己负责的步骤部分，每条指标行读取一次：

| 键 | 引擎 | 内容 |
| --- | --- | --- |
| `ddp_comm_wait_ms` | ddp | 反向传播后，计算流等待梯度 all-reduce 的时间（未被计算掩盖的通信） |
| `ddp_span_ms` | ddp | 最后一个微批次的前向 + 反向 + 上述等待 |
| `ddp_reduce_mb` | ddp | 每步 all-reduce 的兆字节数 |
| `pipe_fwd_ms`、`pipe_bwd_ms`、`pipe_loss_ms` | pipeline | 本 rank 每步的阶段计算 |
| `pipe_wall_ms`、`pipe_idle_ms`、`pipe_bubble_frac` | pipeline | 本 rank 上的调度总时长、非计算部分（气泡加上未被掩盖的传输）及其占比 |
| `pipe_pad_frac` | pipeline | 微批次 token 的填充占比 |

<a id="profiling"></a>
### 性能分析

| 配置项 | 效果 |
| --- | --- |
| `TRACE_STEPS = (first, last)` | 使用 `torch.profiler` 在每个 rank 上分析步骤 `first + 1 .. last`（任何引擎，单 GPU 也可），并写入 Chrome 跟踪文件 `<output>/ranks/trace.rank<r>.json.gz` |
| `PROFILE_STEPS = (first, last)` | 在步骤 `first` 结束时调用 `cudaProfilerStart`，在步骤 `last` 结束时调用 `cudaProfilerStop`（CUDA 设备），使通过 `ncu --profile-from-start off` 或 `nsys --capture-range=cudaProfilerApi` 启动的外部分析器仅记录步骤 `first + 1 .. last` |

`TRACE_STEPS` 的内核时间线会逐步给出：

- `trace_compute_ms`（计算内核时间的并集）、`trace_comm_ms`（NCCL 内核时间的
  并集）、`trace_comm_exposed_ms`（没有计算内核运行时的 NCCL 时间）、
  `trace_copy_ms`、`trace_idle_ms`（没有内核或复制运行的时间），以及忙碌 /
  计算占比。
- 流水线下，NCCL 接收内核等待对端时会自旋，因此其中的 `trace_comm_ms`
  包含等待；`1 - trace_compute_frac` 就是气泡占比。

`TRACE_STEPS` 中的步骤运行较慢（CUPTI 跟踪）。任一窗口中的步骤都会从
稳态摘要中排除。

<a id="per-rank-report"></a>
### 逐 rank 报告

每个 rank 在每个指标步骤（`METRICS_EVERY`）向
`<output>/ranks/rank<r>.jsonl` 写入一行：

- 它所看到的数据的 samples/s 和 tokens/s、步骤时间；
- 它所负责模型部分的 TFLOP/s 和 MFU（流水线下为其阶段，数据并行下为
  整个模型；`threaded` 下逐设备记录）；
- 自上一行以来的显存已分配 / 预留峰值（`kohakumoss_kernels.hardware.memory`
  窗口）、正在使用的设备内存、其设备的 `nvidia-smi` 显存峰值和平均利用率
  （后台 `nvidia-smi -lms 200` 采样器）；
- 引擎的分解数据和该步骤的其他指标。

最后一步会收集每个 rank 的稳态摘要（第 3 步之后、性能分析窗口之外的行），
由 rank 0 写入 `<output>/ranks/summary.json`：包含每个 rank 及整个作业的数据
（对数据并行 rank 的 samples/s 求和、最慢 rank 的步骤时间、跨 rank 取峰值）。

<a id="scaling-runs"></a>
### 扩展性运行

`configs/moss/scale/` 包含扩展性配置层：

| 层 | 设置 |
| --- | --- |
| `scale_lora_bf16` | 静态微批次配置项上的 LoRA dim 32：~1024^2 样本、最多 4 个样本的 9216-token 首次适应批次、离线缓存、`ckpt-resident`、30 步、每步一条指标行、分析第 26-28 步 |
| `scale_lora_fp16` | 面向 V100 / 2080 Ti 的相同设置 |
| `scale_full_fp16` | 用于 `fsdp` 的全量微调 |
| `scale_full_fp16_offload` | 其单 GPU 参考 |

主机层添加 WEIGHTS、DATASETS、CACHE_DIR 和 OUTPUT_DIR；每次运行通过
`--set` 设置 PARALLEL、DEVICES、GRAD_ACC、MICROBATCHES 和 PIPELINE_SCHEDULE，
并使用独有的 RUN_NAME。

`kogine run scripts/dist/scaling_report.py --set ROOT=<output dir> --set HOST=<label>`
读取 ROOT 下的所有运行并写入 `scaling.json` / `scaling.md`。按引擎 x GPU
数量 x 精度 x 算法分别给出：

- 作业的 samples/s，以及相对于相同设置的单 GPU 运行的扩展效率；
- 步骤分解和显存峰值；
- 前 20 步与看到相同全局批次的单 GPU 运行之间的损失一致性
  （`GRAD_ACC` = 数据 rank 数 x `GRAD_ACC`，流水线则使用 `MICROBATCHES`）。

流水线使用 rank 0 的数据和噪声，因此其一致性参考是精确的。数据并行 rank
使用各自的 sigma 和噪声流，因此 DDP / FSDP2 运行与单 GPU 运行的数据一致，
但噪声不同，一致性属于统计意义上的一致（相同 world size 下 FSDP2 与 DDP
之间是精确一致的）。

<a id="heterogeneous-devices-library-api"></a>
## 异构设备（库 API）

- `balance.measure_speed` 对每个设备上的一次稠密矩阵乘法计时；设备不同
  时，`split.rank_speeds` 收集每个 rank 的速度。
- `split.partition` 查找连续的阶段划分，使最慢阶段的
  成本 / 速度最小。

混合 3090 + 2080 Ti 的作业使用 fp16，因为 2080 Ti 不支持 bf16。

<a id="the-parallel-slot"></a>
## `PARALLEL` 插槽

`PARALLEL` 是注册表插槽（`kohakumoss.training.layouts.LAYOUTS`）：每种类型
都是一个 `Layout`，承载运行中所有与类型相关的决策，在开始时一次性解析：

| 字段 | 决定内容 |
| --- | --- |
| `spawns` | 为 `DEVICES` 的每个条目启动一个进程（ddp / fsdp / pipeline），或仅使用一个进程（none / threaded） |
| `shared_data` | pipeline：每个 rank 抽取 rank 0 的批次、sigma 和噪声（一个数据 rank）；其他类型：每个 rank 抽取自己的份额，数据分布于所有 rank |
| `pipelined` | 一个步骤通过流水线引擎执行 `MICROBATCHES` 个微批次，而非 `GRAD_ACC` 个 |
| `stage_rng` | 检查点保留每个阶段设备的 RNG（threaded） |
| `begin` | 构建前的逐运行设置（pipeline：调度器的保留反向传播补丁） |
| `hooks` | 循环的 reduce / agree / norm / accumulate 钩子（`training.build.parallel`） |
| `stages` | 流水线引擎、其目标函数，以及本 rank 的可训练名称和参数 |
| `parts` | 检查点组成部分（FSDP2 完整状态字典、逐阶段流水线部分、分片的主机优化器状态） |
| `driver` | 训练循环：`DRIVER`（native / lightning）或 threaded / pipeline 驱动 |
| `full_weights` | 每个 rank 持有完整权重以供导出的上下文（FSDP2：解分片） |
| `problems` | 布局的配置项问题（`problems(k)`），由配置项验证报告 |
| `plan_problems` | 布局的内存计划问题（`plan_problems(k, plan)`），由配置项验证报告 |

`PARALLEL` 中的点分路径指定其他位置定义的 `Layout`；针对具体类型名称
（`'ddp'`、`'fsdp'`、...）的配置项检查不适用于它。
