[英文原文](../../concepts/training.md)

<a id="training"></a>
# 训练

`scripts/train/moss.py` 是训练脚本：每个配置项都是一个 UPPER_CASE 全局变量，
通过配置文件或命令行设置：

```bash
kogine run scripts/train/moss.py --config configs/user/my-lora.py --set LR=2e-4
```

`kohakumoss.training.entry.run(globals())` 一次性验证所有配置项
（通过一次 `SystemExit` 列出所有问题，对拼写错误给出“你是否想输入”的提示），
然后按固定顺序构建，并将循环交给驱动器。
操作演示见 [first-lora.md](../guides/first-lora.md)；本页介绍训练器的行为。

<a id="build-order"></a>
## 构建顺序

1. 精度（`PRECISION`、`KEEP_FP32`、`FP16_FIX`、`FP16_FIX_KWARGS`）。
2. 主干：将 `WEIGHTS` 检查点直接加载到设备。
3. `FP16_FIX`（`apply_fp16_fix`），然后是内存计划（量化基座、权重流式传输、
   激活卸载、块运行器）。
4. 可训练参数集合：LyCORIS 适配器（`kohakumoss.adapters`），或在 `ALGO="full"`
   时使用 `FullTrainable`（主干自身的参数；除非启用 `TRAIN_TEXT_REFINER`，
   否则文本细化器保持冻结）。可训练参数转换为主副本数据类型。
5. `bind_ops()`、编码器、内核预热、流水线阶段、并行钩子、编译、
   TrainStep、OptimBundle、EMA、数据、日志记录器、回调、检查点、恢复、驱动器。

所有内容都在这里一次性解析；循环调用构建时解析出的对象。

- **每次运行都使用全新状态。** 每次运行都从内核库在全新进程中的进程状态开始
  （`kohakumoss_kernels.run_state.reset_run_state`：没有内存中的后端失败或探测结果、
  默认后端优先级、没有已确定的调优选择、空的决策日志；调优表和磁盘上的失败记录保留），
  因此同一进程中的第二次运行（notebook、参数扫描）不会继承第一次的状态。
- **编码器放置。** 构建后，日志会在推理放置行中指出文本编码器和 VAE 的放置位置，
  或说明该运行为何不持有它们。
- **种子设置。** 全局 RNG 在运行开始时以及构建之前再次使用 `SEED` 设置种子，
  因此构建时的初始化不依赖数据集扫描（每个 rank 的模型和适配器初始化相同），
  构建之后使用 `SEED + rank`。训练步骤的 σ 和噪声生成器使用 `(SEED, data_rank)`
  （`layout.data_rank(rank)`）设置种子：每个数据并行 rank 抽取自己的 σ 和噪声；
  在 `PARALLEL="pipeline"` 下，每个阶段都使用 rank 0 的随机流
  （所有阶段看到相同的微批次）。

<a id="kernel-prewarm"></a>
### 内核预热

第 0 步之前，主干已解析的 `attention_varlen` 和块算子（归一化、RoPE、
SiTU、门控残差）在临时输入上运行前向 + 反向，覆盖计划批次可能达到的每个长度桶。
它们在任何梯度检查点、`no_grad` 或推理模式之外运行，
因此不会在启用梯度检查点的前向内部进行内核选择。

| 预热输入 | 来源 |
| --- | --- |
| 最长样本（注意力分段长度不超过它） | `min(MAX_TOKENS, largest planned latent grid + max_text_tokens)` |
| 最大批次 | 词元批次划分时为 `MAX_TOKENS`，按数量划分时为 `BATCH_SIZE` 个样本 |
| 块算子词元桶 | 256、512、……直到最大批次，每次运行使用桶上限或最大批次中的较小者 |
| 记录的算子 | 主干的无权重孪生模型（结构、基座存储、数据类型和梯度标志相同；不占用权重内存），在训练设备上、在 `PARALLEL="pipeline"` / `"threaded"` 将模型拆分为阶段之前 |

- **文本编码器。** 在线文本编码器的算子在其第一次处理之前以同样方式调优，
  因此描述文本全部已缓存的运行不会调优任何文本编码器算子
  （[tuning.md](../internals/tuning.md#prewarm)）。
- **线程化阶段。** 后续线程化阶段设备的内核算子从同一显卡的表中绑定；
  在临时输入上运行的适配器预热会在第 0 步前为每个线程化阶段设备确定选择
  （`adapter_prewarm_devices`）。
- **多进程。** 运行随后广播 rank 0 的调优表。
- **适配器。** 随后，适配器集合的 `prewarm` 会在临时输入上，
  遍历直到最大批次的适配器词元桶（从 256 开始的 2 的幂次），
  为每种不同的适配器自动调优算子确定选择（融合内核与编译公式比较），
  因子保留其值且不留下梯度，也不调用各层，
  因此流式层同样被覆盖，每个适配器算子都在第 0 步前绑定。
- 所有预热都在安装梯度同步钩子之前运行，且不留下梯度。
- **`TUNE="off"`** 不执行这两种预热绑定：每个适配器算子采用其已存储的选择，
  否则采用融合内核，不做测量（LyCORIS 自身的 tile 调优也关闭）。
  CPU 设备会被跳过（它们的算子使用未经调优的即时执行公式）。

预热后，调优器被冻结：之后首次遇到的问题键会绑定其第一个候选项，不计时，
并记录一行 WARNING。调优仅在这些窗口重新开放
（[tuning.md](../internals/tuning.md#tuning-windows)）：

- 在线文本编码器的预热，紧接其第一次处理之前；
- 前 `RETUNE_STEPS` 步（默认 3）中的每一步之后，对截至当时排队的问题
  重新调优内核与编译实现的选择；
- 第一个采样网格的第一个采样器步骤，针对它自身的形状。

最后一次重新调优步骤之后，内核与编译实现的选择被锁定：不再将任何问题加入队列或重新调优。
如果第一个采样网格稍后才出现，其第一个采样器步骤仍会对自身形状进行调优。

`release_workspaces`（在每个已预热设备上执行 gc、设备同步、`empty_cache`）
在内核预热后运行，并在适配器预热后再次运行。第一次释放必须先于适配器预热：
否则内核预热释放的工作区（在 RTX 3090 上测得为 512 MiB）会容纳适配器预热产生的
小型持久分配，使其内存段在整个训练期间一直保持预留。

<a id="adapter-step-state"></a>
### 适配器步骤状态

围绕每个微批次，适配器集合的 `begin_step(sigma, seq, step)`
设置 T-LoRA 掩码并抽取 DyLoRA 秩 / 模块 dropout 跳过决策（以步骤和微批次设置种子）；
该步最后一次反向后，以及验证处理后，`end_step()` 会清除掩码，
因此采样和导出看到的是满秩适配器。

<a id="linear-algebra-backend"></a>
### 线性代数后端

在 CUDA 设备上，每个训练工作进程启动时调用一次
`torch.backends.cuda.preferred_linalg_library("cusolver")`，
为整个进程与运行设置后端（OFT 的批量 Cayley 逆；使用 cu130 wheel 时，
MAGMA 的批量求逆会在 sm86 上出错）。训练期间通过 `MossPipeline` 采样时，
每次 generate 调用都选择 cuSOLVER，并恢复之前的后端，
在训练进程中，之前的后端也是这一相同设置。

<a id="data-and-online-caches"></a>
### 数据与在线缓存

- `MAX_SAMPLES` 限制每个词元批次的样本数（0 = 不限制）；
  `TOKEN_BUDGET="exact"` 使用缓存的文本长度规划，因此 `MAX_TOKENS` 是硬上限。
- 使用 `online` 缓存时，如果一个批次的所有键都已缓存，
  到达的批次已完成编码，可直接使用。否则对其编码
  （潜变量按完整计划网格编码，再收集为词元丢弃子集；文本截断到每个样本的 `text_limit`），
  完整编码交给数据侧的后台 `CacheWriter`，在每个检查点之前刷写，并在结束时关闭。
- 新编码在使用前按缓存的存储方式精确舍入，因此未命中时使用的训练值
  就是后续命中会返回的值。
- 当批次文本长度为最终值（离线文本缓存或在线命中）时，批次布局（SeqInfo）
  在加载器工作进程中构建，并随批次传递；只有提示词在设备上编码的批次
  才在主进程构建布局（CPU 上每个 4 样本、16k 词元批次约 0.7 ms）。

<a id="objective"></a>
## 目标函数

`z_σ = (1-σ) z0 + σ ε`；模型在 σ 处求值，并预测 `ε - z0`。

- **损失。** 对打包批次中的每个图像词元计算 MSE，按该优化器步骤的**全局**
  图像词元数归一化：先获取该步全部 `GRAD_ACC` 个微批次，
  对其词元计数求和（并在各 rank 间执行 all-reduce），
  每个微批次的损失都除以同一个分母。因此，累积 `GRAD_ACC` 个加载器批次
  恰好得到一个包含全部这些数据的大批次的梯度。
- **`LOSS`** 选择 `mse`、`huber` 或 `l1`（流水线引擎的最后阶段使用相同损失）；
  注册表中存在 `masked_mse`，但验证会拒绝它，因为没有数据源生成损失掩码。
- **`WEIGHTING`** 将一个依赖 σ 的权重乘到数据权重上（`none` 是 MOSS 方案；
  `x0` 和 `eps` 分别复现 x0 空间 / 噪声空间的 MSE）。

<a id="timestep-samplers"></a>
## 时间步采样器

| `TIMESTEP` | 采样器 |
| --- | --- |
| `moss`（默认） | 下文的技术报告采样器 |
| `uniform`, `logit_normal` | 连续替代方案 |

`moss` 使用离散网格 σ_j = j/1000，具有截断的 logit-normal 概率质量函数
（`S = 1 - sigmoid(Z)`、`Z ~ N(μ, s²)`，截断值 `a` 将 S/a 重缩放到网格上），
再混入比例为 `uniform_mix` 的均匀分布，将质量 `p_full` 移到 σ = 1，并应用静态偏移。
默认参数（μ = -1.09、s = 0.95、cutoff 0.95）使 σ 中位数位于 0.78。
`schedule="tech_report"` 完全按照报告中的四个 Stage-3 区间，按步骤切换参数。

给定 `F(x) = P(S <= x) = 1 - Phi((log((1-x)/x) - mu)/s)`，
在移动 `p_full` 和应用偏移映射之前，网格点 j 的质量为
`(1 - rho) * [F(a j/T) - F(a (j-1)/T)] / F(a) + rho/T`（rho = `uniform_mix`，T = 1000）。
该表以 fp32 构建：中位数以下的区间采用 `F` 值之差，以上的区间采用生存值
`1 - F` 之差（各自使用自己的 `erfc`），因此不会有任何区间对两个接近 1 的数做减法；
每个区间与 fp64 表的绝对差均在 1e-6 内（相对差 3e-4）。

<a id="optimisers"></a>
## 优化器

| `OPTIMIZER` | 含义 |
| --- | --- |
| `adamw` | torch AdamW（CUDA 上融合） |
| `adamw16` | 面向 16 位参数的 AdamW：fp32 计算、16 位状态，存储 `sqrt(v)`，使二阶矩不在 fp16 中下溢；`OPTIMIZER_KWARGS={"rounding": "nearest" \| "stochastic" \| "kahan"}`（随机舍入仅支持 bf16） |
| `adamw8bit` | bitsandbytes 8 位 AdamW，可选依赖 `quant` |
| `adamw8bit_kernel` | KohakuMoss 自身的 8 位 AdamW：m（int8）和 sqrt(v)（uint8）以分块状态保存，由内核库的 `adamw8bit_step` 执行更新（Triton / TileLang / 即时执行，按设备选择）；`OPTIMIZER_KWARGS={"rounding": ...}` 与 `adamw16` 相同 |
| `muon` | 对二维隐藏矩阵使用 Muon（融合 `qkv` 拆为 3 份，`ffn_up_gate` 拆为 2 份，每份通过 5 次 Newton–Schulz 迭代正交化，更新按 0.2·√max(rows, cols) 缩放以匹配 AdamW 的 RMS），其余参数使用 AdamW，统一在一个优化器中 |
| `prodigy`, `schedule_free` | prodigyopt / schedulefree，可选依赖 `optim`（schedule-free 在采样与验证时切换为评估权重） |

- **`OPTIMIZER` 与 `OPTIM_STATE`。** `OPTIMIZER` 指定算法及其实现；
  `OPTIM_STATE`（`gpu`、`stream`、`cpu`）仅放置 `adamw` 的状态并选择其放置构建器
  （`gpu` 对应 torch AdamW，`stream` 对应 `host_state_adamw`，`cpu` 对应 `cpu_adamw`，
  权重流式传输将可训练主副本保存在主机时也如此）。放置构建器
  （`cpu_adamw`、`host_state_adamw`、`master_adamw`）不是 `OPTIMIZER` 名称；
  验证会指出应改用的 `OPTIMIZER` / `OPTIM_STATE` 组合。
- **`OPTIMIZER_KWARGS`** 根据已注册构建器自身的参数检查
  （`adamw16`：`rounding`；`muon`：`momentum`、`ns_steps`；`schedule_free`：
  `warmup_steps`；`adamw8bit_kernel`：`rounding`；`adamw`、`adamw8bit`、`prodigy`：
  无）。`LR`、`BETAS` 和 `EPS` 各有独立配置项，不检查以点分路径指定的 `OPTIMIZER`。
- **`WD_RULES="moss"`** 对偏置、归一化层、一维张量和嵌入器禁用权重衰减，
  并在全量微调中对最后一个联合块的 q/k 归一化权重应用向 1 衰减
  （`p -= lr·wd·(p-1)`）。
- **`SCHEDULE`** 是 AnySchedule 配置；`end: -1` 表示 `STEPS`。
  加载检查点会将恢复步骤的调度值（学习率和其他任何被调度的键）写入调度所控制的
  优化器参数组，包括逐参数和卸载优化器中的学习率持有对象，
  因此恢复后的第一步会使用调度规定的学习率。

<a id="the-optimiser-step"></a>
## 优化器更新步骤

`OptimBundle.step()`：

1. 撤销梯度缩放；
2. 计算梯度的 fp32 全局范数（逐张量范数也使用 fp32）；
3. 检查范数和损失是否为非有限值，并在各 rank 之间统一该标志
   （在 DDP / FSDP2 下，损失份额与标志合并到一次 SUM all-reduce 中）；
4. 通过**一次**传输将损失、范数和标志送到主机：这是一个训练步骤中唯一的主机同步
   （该步的词元计数通过锁页内存传到设备，不产生主机同步）。

异常步骤会被跳过（fp16 缩放乘以 0.5，下限为 1），并计数
（`skipped_steps` 和 `nonfinite_steps`）；裁剪通过设备端系数乘以梯度实现。

<a id="fused-step"></a>
### 融合更新

内存计划启用 `fused_step` 时（`low` / `minimal` 配置档），每个参数在该步最后一个
微批次的反向内部执行更新（`FusedStepBundle`）。完整梯度从不会同时存在，因此：

- **裁剪。** `GRAD_CLIP`（所有 bundle 中都表示全局范数）将每个梯度乘以
  `min(1, GRAD_CLIP / the previous step's global norm)`，
  即参数更新时最后已知的全局范数（运行的第一步不裁剪；非有限步骤保留上一次的系数；
  检查点保存该系数）。因此，梯度尖峰出现的当步使用上一步的系数
  （上一次范数小于 `GRAD_CLIP` 时不裁剪）；它自身的范数决定下一步的系数，
  非有限尖峰则由非有限值防护跳过。
- **`grad_norm`** 是在设备上累积的未裁剪梯度的全局范数。
- **不支持 Muon。** Muon 在反向之后更新完整矩阵，因此配置项验证会拒绝
  将 `OPTIMIZER='muon'` 与 `fused_step` 计划配合使用（`kohakumoss.training.knobs`）：
  使用 Muon 的全量微调应使用不带 `optfused` / 流式传输的配置档
  （例如 `ckpt-resident`），或在 `MEMORY_PROFILE='manual'` 下设置 `FUSED_STEP=False`。
- **DDP。** 每个参数基于其跨 rank 求和后的梯度更新
  （见 [multi-gpu.md](../guides/multi-gpu.md)）。
- **非有限参数。** 梯度非有限的参数不更新（状态不变）；该步其余更新仍然应用。
  这样的步骤计入 `nonfinite_steps`（缩放回退，写出不稳定性转储），
  不计入 `skipped_steps`，EMA 也随之更新；其 `grad_norm` 为 NaN，
  `nonfinite_params` 统计被冻结的参数数量。

<a id="stopping-on-non-finite-steps"></a>
### 遇到非有限步骤时停止

在两种模式中，`NONFINITE_PATIENCE`（默认 8）会在损失缩放达到下限后连续出现
这么多次非有限步骤，或连续这么多步出现相同非有限参数时停止运行，并指出参数名称。
停止前，它将失败步骤写入 `<output>/instability/repro_step.pt`
（`training/callbacks/repro.py`，一个 `torch.save` 字典）：

| 键 | 内容 |
| --- | --- |
| `z`, `text`, `sigma`, `target`, `hw`, `text_len` | 该步最后一个已准备的微批次，以主机张量保存（`z`：加噪潜变量） |
| `weight` | 加权后的逐样本损失权重 |
| `img_keep` | 每个样本保留的网格索引或 None |
| `denom` | 该步的词元数，整数 |
| `step` | 步数 |
| `adapters` | 按名称保存的每个可训练张量，其值为该步之后的值（跳过的步骤保持前向时看到的值） |

错误消息会指出该文件；`KOHAKUMOSS_REPRO_STEP=<file>` 会在
`tests/training/checkpoint/test_repro_step_nan.py` 中重放它。
原生循环和 Lightning 循环会记录它；流水线不会。

<a id="precision"></a>
## 精度

| 配置项 | 值 |
| --- | --- |
| `PRECISION` | `bf16`、`fp16` 或 `auto`（每个训练 CUDA 设备都支持 bf16 时使用 bf16，否则 fp16；非 CUDA 上使用 bf16） |
| `FP16_FIX` | `none`、`online`（在运行时缩放残差流；基座使用 `FP16_FIX_KWARGS={"c": 1/16}`，其流峰值接近 62k）、`fp32_residual`（以 fp32 保存流）；参见 [numerics.md](numerics.md#fp16_fix-modes) |

- 可训练参数保存在 fp32 主副本中（`adamw16` 为 16 位），`KEEP_FP32`
  中的模块保持 fp32，前向在 autocast 下运行。
- `fp16` 启用动态损失缩放器（初始值 2^16，每 2000 个无异常步骤增长 ×2）。
- fp16 修复检查点（折叠已固化到权重中）使用 `FP16_FIX="none"`
  并将该文件作为 `WEIGHTS` 进行训练；`FP16_FIX="weights"` 不是训练可用值
  （推理可加载两者）。
- 在 `VARIANT="base"` / `"sft"` 下，验证会拒绝 `PRECISION="fp16"`
  与 `FP16_FIX="none"` 的组合；`DEVICES` 任一条目不支持 bf16 时，
  也会拒绝 `PRECISION="bf16"`。

<a id="memory-plan-in-the-loop"></a>
## 循环中的内存计划

- **优化器状态。** `OPTIMIZER="adamw"` 时，优化器遵循计划中的 `optim_state`：
  `gpu` 对应 torch AdamW，`stream` 对应状态常驻主机的 GPU AdamW
  （`host_state_adamw`），`cpu` 对应主机 AdamW（`cpu_adamw`；
  可训练权重从主机主副本流式传输时也强制使用）。
- **梯度卸载。** 启用 `grad_offload` 时，每个梯度在累积后立即离开设备，
  并在 fp32 主机内存中对 `GRAD_ACC` 个微批次求和；主机 AdamW 直接读取这些和，
  设备优化器在更新前将它们取回，而使用 `fused_step` 时，
  最后一个微批次的反向会原地完成每个参数的更新。
- 每条优化器路径都保留参数组（权重衰减规则、LoRA+ 输出侧学习率），
  调度保持每组的学习率比例。
- 步骤之间，空闲的缓存设备内存仍归分配器持有，除非 `STEP_RELEASE` 要求释放
  （[profiles.md](profiles.md#releasing-cached-memory)）。

<a id="pipeline-layouts-linux"></a>
## 流水线布局（Linux）

| `PARALLEL` | 布局 | `PIPELINE_SCHEDULE` |
| --- | --- | --- |
| `threaded` | 单个进程内，在 `DEVICES` 上分布各阶段 | `gpipe` 或 `async` |
| `pipeline` | 每个阶段一个进程（torch.distributed pipelining） | torch 的调度方案 |

- 一步包含 `MICROBATCHES` 个加载器批次；每个流水线 rank 读取相同批次，
  只更新自身阶段的参数，检查点和最终导出在 rank 0 上合并各阶段。
- 每个微批次的适配器步骤状态（T-LoRA 秩掩码、DyLoRA 秩、模块 dropout 抽样）
  按微批次顺序抽取，与原生循环相同，并在该微批次自身的前向和反向之前
  再次安装（包括其梯度检查点重计算）；`pipeline` 引擎根据其运行的
  已填充静态布局抽取这些状态。
- 线程化 `async` 调度在下一个微批次运行前向的同时运行当前微批次的反向，
  因此不允许这些适配器在那里使用梯度检查点计划
  （使用 `gpipe` 或 `checkpoint='none'`）。
- `pipeline` 阶段针对包含 `MAX_TOKENS` 个词元和 `MAX_SAMPLES` 个样本的静态微批次构建，
  因此需要按词元划分批次、`MAX_SAMPLES >= 1`、`TOKEN_BUDGET="exact"`
  和离线文本缓存；支持词元丢弃（每个样本的损失覆盖其保留的词元）。
- 两者在 Windows 上都不被允许，且这些布局不提供采样 / 验证回调。

<a id="ema"></a>
## EMA

`EMA = 0.9999`（或衰减系数列表）维护 fp32 平均值，在每个成功步骤之后更新。

- **放置。** 除非规格指定 `device`，否则每个平均值位于其参数所在设备。
  在 `stream='swap'` 下，块主副本是主机张量，因此其平均值为主机缓冲区，
  在反向内部的块优化器完成向这些主副本的回写后
  （`StreamedAdamW.synchronize`），在主机上更新；
  主机更新对每步、每个参数都会消耗 CPU 时间。
- **内存。** 内存估算按每个平均值、每个可训练设备参数 4 B 计入。
- **转换。** 不在平均值所在设备上以 fp32 保存的参数，每次更新转换一次，
  每个分块最多 64 M 个元素，并从这一份副本更新所有衰减系数对应的平均值；
  将加速器参数平均到主机缓冲区时，通过两个锁页暂存分块复制
  （一个在复制的同时，另一个用于求平均）。

<a id="power-ema"></a>
### 幂函数 EMA

`EMA = {"kind": "power", "sigma_rel": [0.05, 0.10]}` 维护 EDM2 幂函数平均值，
并在每个检查点写入快照；

```bash
python -m kohakumoss.training.ema.posthoc --run <dir> --sigma-rel 0.07 --out ema.safetensors
```

之后可重建任意宽度的平均值。

- 给定 `sigma_rel` 的幂函数 EMA 指数为
  `g^3 + 7 g^2 + (16 - t) g + 12 - t`（t = `sigma_rel^-2`）的最大根，
  从 `g = sqrt(t)` 开始以 fp32 Newton 法求得，三次函数在该处为凸且递增。
- 事后拟合是快照曲线 `p(tau) = (g+1) tau^g / t^(g+1)` 的最小二乘组合，
  在单位范数曲线的基底中以 fp32 求解：其 Gram 矩阵为
  `sqrt((2g_a+1)(2g_b+1))/(g_a+g_b+1) (lo/t_a)^(g_a+1/2) (lo/t_b)^(g_b+1/2)`
  （lo = min(t_a, t_b)），每项都为 O(1)，而原始 Gram 矩阵会使 fp32 溢出
  （`t^(g+1)`）。

<a id="drivers-checkpoints-callbacks"></a>
## 驱动器、检查点与回调

<a id="drivers"></a>
### 驱动器

`DRIVER="native"`（默认）和 `"lightning"`（手动优化）运行相同的 TrainStep 和 bundle，
以相同方式开始每个微批次（融合更新、梯度卸载、DDP 归约器），并得到相同参数。

Lightning 在当前进程的单个设备上驱动循环；进程、梯度同步与放置由 KohakuMoss 负责。
LightningModule 将主干保存在其模块树之外（普通属性），因此自身没有任何参数：
Lightning 在 fit 开始时将模块移动到设备、在拆卸时移回 CPU，都不会移动任何内容，
每个内存配置档（`swap` 下的主机 fp32 主副本、流式权重、主机优化器状态、
DDP 共享主机区域）都保持其放置；训练后导出读取的模型放置与原生驱动器下相同。

<a id="checkpoints"></a>
### 检查点

检查点目录包含可训练状态、EMA、优化器 + 调度 + 缩放器、加载器位置，
以及每个 rank 一个 `rng.<rank>.pt`（全局 torch / numpy / python RNG、
该 rank 各设备的 CUDA 状态、σ 和噪声生成器）。

- **精确恢复。** 每个 rank 恢复自己的 RNG 文件，因此恢复运行与不间断运行
  抽取相同的数据、σ 和噪声流。在 CPU 上，恢复后的参数逐位精确一致；
  在 CUDA 上，通过原子操作累加的内核（融合注意力反向、cuBLAS split-K、
  cuDNN 自动调优）按到达顺序舍入，因此恢复运行在舍入精度内一致。
- **原子写入。** 检查点写入 `step_XXXXXXXX.partial/`（每个 rank 写其 RNG 文件，
  rank 0 写其余内容），完成后重命名；只列出或恢复已重命名的目录，
  如果保存周期已保存最后一步，就不会重复写入它。
- **`SAVE_MINUTES`** 每 `LOG_EVERY` 步检查一次，每个 rank 都采用 rank 0 的决定。

<a id="callbacks"></a>
### 回调

| 回调 | 输出 |
| --- | --- |
| 吞吐量 / MFU、显存峰值 | 指标 |
| 指标记录 | 控制台 + `metrics.jsonl`、KohakuBoard 或 wandb |
| 样本 | 在当前模型上通过 `MossPipeline` 生成：每步的图像 + `settings.json` |
| 验证 | 每个 σ 的固定噪声验证损失（所有 rank 验证批次上的全局逐词元平均值，全部使用条件：无文本 dropout） |
| 不稳定性转储 | 出现非有限步骤或损失超过其运行平均值的 5× 时，写入 `instability/step_X.json` |

<a id="shutdown"></a>
## 关闭

`training.shutdown.install`（`entry.run` 执行的第一件事；基准矩阵的运行封装也使用它）
使 SIGTERM 在主线程中抛出 `SystemExit(143)`，从而让运行像异常一样展开清理
（finally 块、退出处理器），并在退出时终止所有仍存活的 multiprocessing 子进程
以及该进程的 forkserver。

Linux 上的加载器工作进程由 forkserver 派生：它们监视 forkserver，
而 forkserver 在训练器被直接终止后仍会存活，因此退出处理器在运行被终止时
会终止 forkserver 及其工作进程（每个约 0.5 到 0.8 GB RSS）。
SIGKILL 无法处理；基准矩阵运行器为每次运行创建独立进程组，
并终止整个组（`kohakumoss_bench/run/proc.py`）。
