[English](../../performance/bench-matrix.md)

<a id="the-real-gpu-bench-matrix"></a>
# 真实 GPU 基准测试矩阵

工作树的一份冻结副本（以及 LyCORIS 等额外仓库）会在每台主机的每张显卡上运行：

- 真实权重、真实数据、每种速度 / 显存设置；
- 带对比网格图的训练质量运行，以及推理检查；
- ncu 效率分析轮次和功能覆盖运行。

每次运行都将其配置、日志、指标行、nvidia-smi 跟踪和结果记录写入磁盘，并由一份报告按显卡型号回答速度、显存、步骤是否已优化、质量和覆盖情况等问题。

```
scripts/bench/matrix/launch.py                   local driver: plan, freeze, ship, install, start, status, join, collect, report
scripts/bench/matrix/host.py                     host runner (on the host, from the frozen tree)
scripts/bench/matrix/src/kohakumoss_bench/       the dev package both use: run/, plan/, measure/, report/
```

该包独立于 kohakumoss wheel 安装，通过 `uv pip install -e scripts/bench/matrix` 与可编辑的 kohakumoss 安装放在一起（主机默认的 `install` 命令会从冻结工作树安装两者）。下文的模块路径均相对于 `kohakumoss_bench/`。

<a id="launch"></a>
## 启动

`launch.py --phase <phase>` 每次调用针对一份[主机文件](#hosts-file)运行一个阶段：

| 阶段 | 执行内容 |
| --- | --- |
| `plan` | 试运行，不执行任何训练：逐主机列出 nvidia-smi 看到的显卡（通过 ssh；主机会使用哪些显卡及其当前内存使用量）、只读预检（venv 解释器、权重、修复文件、数据集、`/dev/shm` 空间、`ncu --version`）、显卡池，以及各池队列中的每次运行及其预计耗时（[计划](#plan)），写入 `runs/bench/<bench>/plan.md` 和 `plan.json` |
| `start` | 冻结、传送并安装工作树，然后启动[主机运行器](#host-runner)（见下文） |
| `status` | 打印每台主机的 `status.json` |
| `join --gpus 2,3` | 将 `join.json` 写入正在运行的基准测试：这些显卡一旦空闲，就加入其型号对应的池 |
| `collect` | 将 `runs/<bench>/<host>/` 复制到 `runs/bench/<bench>/<host>/` |
| `report` | 在该处写入 `report.md` 和 `report.json`（[报告](#report)） |

`start` 列出每个仓库要传送的文件（包括已跟踪和未跟踪但未被忽略的文件，排除 `run/remote.py` 的 `SKIP` 中仅限本地的目录），并将它们哈希为每个仓库的单一标识（`<HEAD>-<sha1 of paths and bytes>`）。每个仓库作为一个 tar 包传送（标识已存在时跳过），分别运行安装命令，上传主机运行器配置（`configs/<bench>-<host>.json`），并启动主机运行器（通过 ssh 执行 `nohup`）。后续编辑永远不会传入已在运行的基准测试：每台主机运行的始终是它收到的工作树。

<a id="hosts-file"></a>
## 主机文件

一个 JSON 文件 `{"bench": {...}, "hosts": [...]}`。bench 块填充 `Options`（`plan/config.py`；命令行上的 `--option key=value` 可覆盖其中一个选项）；每个 host 填充 `Host`：

| 字段 | 含义 |
| --- | --- |
| `name`, `transport`, `ssh` | "ssh" 配合命令前缀（`["ssh", "user@host"]`）、"local"（本机，前台运行）或 "manual"（打印命令） |
| `root` | 主机上的基准测试根目录：`trees/<identity>/`、`trees/<repo>-<identity>/`、`configs/`、`runs/<bench>/<host>/` |
| `python`, `install` | venv 解释器，以及将冻结工作树安装到其中的命令（`{python}`、`{tree}`；"" 表示跳过：也可以改用环境中的 `PYTHONPATH` 指向这些工作树） |
| `device_type`, `gpus`, `xpu_cards` | CUDA 显卡来自 nvidia-smi（`gpus` 限定索引）；XPU 显卡通过列表指定 |
| `ram_gib` | 决定是否允许执行卸载到主机的全量微调 |
| `weights`, `fix_file`, `copies` | `{"source", "stage"}`：复制一次到快速本地存储；`hf:<repo>` 表示下载；`copies` 用于暂存更多内容（数据集） |
| `datasets`, `cache_dir`, `cache_from` | 主机上的 DATASETS 条目、离线缓存目录，以及可复制以免重新构建的现有缓存 |
| `env` | 每次运行的额外环境变量；`{tree}` 和 `{tree:<repo>}` 展开为冻结工作树 |
| `quality`, `kinds`, `profiles`, `bases`, `catalog`, `runs` | 质量运行名称、运行种类、命名配置档（不含目录点）、保留的基座精度、主机的目录点（"all" / "frontier" / "none"；"" = 基准测试的设置）、保留的运行名称模式（fnmatch；保留某运行时也会保留其依赖项） |
| `ncu` | Nsight Compute 命令（计数器仅管理员可用时为 `["sudo", "-n", "/usr/local/cuda/bin/ncu"]`）；为空则不执行 ncu 分析 |

bench 块的 `repos`（`{"lycoris": "<local path>"}`）列出在主工作树旁一并冻结、传送和安装的额外仓库。

<a id="host-runner"></a>
## 主机运行器

准备一次，随后每种显卡型号对应一个池：

1. **暂存。** 权重、fp16-fix 文件和待复制项会被复制到其暂存路径（NVMe 或 `/dev/shm`）；已存在且大小相同的文件会被保留，因此每台主机只复制一次。每次运行都从暂存位置加载。
2. **探测。** 对每张显卡测量：稠密 GEMM 速率（8192^3，fp32 累加；显卡支持时使用 bf16，同时测量 fp16，以及 sm89+ 上的 fp8 `_scaled_mm`）、张量核心 MMA 峰值（在 Triton 循环中对驻留寄存器的 128 x 128 x 64 `tl.dot` 执行 fp32 累加，每个 SM 8 个 CTA，dtype 同前；报告中的 "MFU (measured MMA)" 以它为分母），以及页锁定 / 可分页主机 <-> 设备复制带宽（1 GiB）。这些就是报告用于衡量各次运行的上限。
3. **缓存。** 在主机显存最大的显卡上一次性构建每次运行读取的离线潜变量和文本缓存（`cache_latents.py`、`cache_text.py`，使用与各次运行相同的数据块），并以该数据块作标记；数据块不变时跳过构建。
4. **网格图提示词。** 选择目录中出现最频繁的画师，按种子抽取其条目，每条描述文本均通过训练描述文本模板（变体 0）渲染。
5. **池。** 同一型号的空闲显卡（nvidia-smi 中无其他计算进程）共享一个队列，队列被划分为单元（`matrix.units`：一次运行及所有依赖它的运行，按顺序在同一张显卡上执行）。第一个单元（调优运行）独自执行；结束后，池中的其他显卡才开始运行。最后一个单元（重启运行）在所有其他单元结束后开始。每张显卡空闲时领取下一个单元；如果显卡变为忙碌状态，则先等待，随后跳过该次运行。相同型号的显卡绝不会重复运行同一个配置。

每次运行一个进程：权重从暂存位置在数秒内加载，内核调优表、torch.compile 缓存及 Triton / TileLang 缓存都存放在磁盘上，因此一个池只有首次运行会调优；之后每次运行（以及末尾的重启运行）都从表中绑定。调优表按显卡型号区分；每次写入前先合并文件（[tuning.md](../internals/tuning.md)），因此同一个池中的显卡同时调优时会保留彼此的行。

<a id="matrix"></a>
## 矩阵

每个池（`plan/matrix.py`）：

| 种类 | 运行内容 |
| --- | --- |
| speed | 精度（显卡支持时使用 bf16，以及 fp16）x 基座（所有显卡均有 compute 和 int8 weight-only，sm89+ 另有 fp8_compute 和 MXFP8；`bases` 保留其中一个子集）x 算法（LoRA r32、LoKr factor 16 full matrix）x 能放入显存的命名配置档（speed、balanced、low、minimal、auto）；DoRA r16、LoHa r16 和 OFT block 64 各有一行 `auto`；参考组合（第一种精度、compute 基座、LoRA r32）遍历能放入显存的目录点 |
| cache | 在三种缓存模式下运行旗舰设置：无缓存（`LATENT_CACHE` 和 `TEXT_CACHE` 为 "off"：VAE 和文本编码器每步都运行）、离线潜变量缓存、两种离线缓存同时启用。设置包括：第一种精度和 fp16 下的 LoRA r32 `balanced`、LoRA r32 `minimal`、LoKr 16 `balanced`、fp8_compute LoRA r32 `balanced`（sm89+），以及 int8 基座上采用 `ckpt-ring2-actoff-optstream-gradoff` 的 LoKr 16 fp16，并将文本编码器提前 8 个批次流式传输；在不支持 bf16 的显卡上，对应的 fp16 设置只运行一次。编码器使用配置档指定的放置方式（`low` / `minimal` 下文本编码器流式传输，VAE 卸载）。这些运行紧接池的首次运行之后，先于所有其他 rank-1 运行 |
| placement | 无缓存，遍历文本编码器 / VAE 的放置方式：LoRA r32 `balanced`（文本为 `auto`、`resident`、`offload`、`ring`，VAE 为 `resident` 和 `offload`）、LoRA r32 `minimal` 和低显存 LoKr 设置（文本为 `auto`、`resident`、`offload`、`ring`；VAE 使用配置档中的设置）。超出显卡显存的放置方式以 `oom` 结束。报告的 "auto picked" 列是同一设置的 `auto` 行在日志中记录的选择（`text encoder placement:` / `VAE placement:`）。这些运行紧接缓存行之后 |
| batch | 在第一种精度下以 `balanced` 和 `ckpt-resident` 运行 LoRA r32，在支持 bf16 的显卡上另加 fp16 `balanced`，每步 token 数为 `batch_scales`（2、4、8）x `max_tokens`（MAX_TOKENS，即 BatchLimits 预算）；超出显卡显存的行以 `oom` 结束。这些运行紧接池的首次运行之后 |
| online | 参考组合使用 `auto`，启用在线潜变量和文本缓存并写入空目录 |
| cap | `minimal`、LoRA r8 和 LoKr 16、每种基座，在低于显卡显存的各个显存目标（8 GB、6 GB）下运行 |
| quality | LoRA r8 和 LoKr 16 在 bf16 和 fp16（`fp16_fix`）下运行 400 步；如果主机有修复文件，另运行使用另一种修复方式的 LoRA r8 fp16；每 100 步保存检查点，训练期间采样，导出 LyCORIS + ComfyUI |
| export, grid | 每个检查点导出为 LyCORIS，然后运行 `compare_grid`（行依次为基座模型、步骤 100 .. 400，列为网格图提示词） |
| infer | `sample.py`：基座模型、合并适配器、旁路适配器、fp8 DiT + TE（sm89+）、低显存放置方式 |
| coverage | 仅主机上最大的型号：`plan/coverage_runs.py` |
| ncu | 在 Nsight Compute 下以 `auto` 运行参考组合（配置了 `ncu` 的主机；见下文） |
| restart | 在末尾再次运行池的首次速度运行：不得进行任何调优 |

fp16 运行使用 `fp16_fix`："online"（运行时缩放因子，c = 1/16）或 "file"（主机的 fp16-fix 检查点；没有该文件的主机使用 online）。

当配置档的估计峰值（`kohakumoss/memory/estimate.py`）加上 `overhead_gib`（默认为 2.5：估计值未包含的上下文、已加载内核和分配器缓存）不超过显卡显存时，即认为它能放入显存。目录点为能放入显存的条目（`catalog="all"`），或这些条目的前沿（`catalog="frontier"`，默认值）：如果另一个条目在每个轴上的流式传输、卸载和重计算都不更多，且所需估计内存最多只多 `frontier_slack`（3%），则丢弃当前条目。

速度运行：

- 40 步，每步一行指标（每步一次主机同步，所有运行都相同）。
- 第 38 .. 39 步（`trace_steps`，2）在 `TRACE_STEPS`（torch.profiler）下运行，以得到步骤分解（每步的 `trace_compute_ms`、`trace_copy_ms`、`trace_idle_ms`，[multi-gpu.md](../guides/multi-gpu.md)）。
- 中位数取第 10 步之后、性能分析步骤之外的步骤。

<a id="plan"></a>
## 计划

计划中的时长均为估计值：

| 部分 | 估计方式 |
| --- | --- |
| 一次训练步骤 | 模型 FLOPs（`kohakumoss/model/flops.py`，包含重计算）按标称峰值的 `plan_mfu`（0.4）计算（表中缺失的显卡使用 `plan_tflops`）；如果按 `plan_link_gbps`（20 GB/s）计算流式传输权重和激活字节数所需的时间更长，则取后者 |
| 每个进程 | + `plan_overhead_s`（120 s） |
| 一个池的首次运行 | + `plan_tuning_s`（1800 s） |
| 每个 ncu 步骤 | + `plan_ncu_step_s`（300 s） |
| 图像 | 采样器步数 x 2（CFG）次 1 MP 前向传播 |

池的墙钟时间估算在第一个单元之后，将各单元按耗时从长到短分配到负载最小的显卡。

<a id="ncu"></a>
## ncu

ncu 分析对每个池的一次训练运行记录第 `warmup + 1 .. warmup + ncu_steps` 步（第 11 步）：`PROFILE_STEPS` 使用 `cudaProfilerStart` / `Stop` 将其括起，ncu 以 `--clock-control none --cache-control none` 运行（不改变时钟和缓存：其他用户的显卡保持原样，每个内核看到的都是该步骤留下的缓存状态），重放模式由 `ncu_replay` 选择（`measure/ncu.py` 的 `REPLAYS`）：

| `ncu_replay` | 结果 | 一个被分析步骤的成本 |
| --- | --- | --- |
| `app-range`（默认） | 每个分析器范围一个结果：整个步骤作为一个单元 | 每轮指标采集完整运行一次（每轮约 1 min） |
| `application` | 每个内核一个结果（`--profile-from-start off`，宽松匹配） | 每轮串行执行所有内核：~0.12 s x ~9-11k 个内核，每轮约 20 min |
| `kernel` | 每个内核一个结果（`--profile-from-start off`） | 与 `application` 相同，但每个内核前后额外保存 / 恢复设备内存 |

四个指标需要 6（ga100）到 9（gb202）轮采集，因此按内核分析一个步骤需要 2-3.5 h。`app-range` 在数分钟内测量同一步骤，提供瓶颈判定结果所用的全步骤张量 / SM / DRAM %，但不提供逐内核分解（速度运行的 torch 跟踪包含内核耗时）。ncu 在范围重放模式下拒绝 `--profile-from-start`：范围由分析器调用本身界定。

每个结果收集：`gpu__time_duration.sum`、张量流水线指标 `.avg.pct_of_peak_sustained_elapsed`、`sm__throughput` 和 `dram__throughput`（相对峰值的百分比，按经过时间计算）。张量指标取 `ncu --query-metrics --chip <chip>` 为该显卡列出的 `sm__pipe_tensor_subpipe_hmma_cycles_active`、`sm__pipe_tensor_op_hmma_cycles_active`、`sm__pipe_tensor_cycles_active` 中第一个可用的指标（否则 sm90+ 使用 subpipe 名称，更早的架构使用 op 名称）。

`sudo` 会重置环境，因此运行环境通过环境文件传入（`run/child.py --env-file`），运行缓存（HOME、Triton、inductor）和一份内核表副本位于运行目录内：root 进程写入的任何内容都不会进入运行器账户自己的缓存。

报告为 `ncu.ncu-rep`，导入为 `ncu.csv`，并汇总到 `ncu.json`：

- 每个内核（`app-range` 下为每个范围；按名称分组）：调用次数、每步 ms、时间占比、按持续时间加权的张量 / SM / DRAM %，以及分类：计算（SM % 领先且达到 `ncu_bound_pct`，60）、内存（DRAM % 领先且达到该阈值）或延迟（均未达到）；
- 总计：按持续时间加权的张量 / SM / DRAM %、每步内核 ms、各分类的时间占比；
- 相对匹配的速度运行（精度、基座、算法、配置档相同）：内核忙碌占比（每步内核 ms / 未进行性能分析的步骤时间）、整个步骤的张量活跃度（张量 % x 忙碌占比，即 DCGM 的 tensor-active 指标报告的全步骤数值），以及结果：PCIe 受限（速度运行的传输占比 >= `--transfer-bound`）、主机受限（内核占据不到步骤的一半时间），否则取时间占比最大的分类。

ncu 进程失败（sudo、不支持的指标）记为 `profile-failed`：加入待审查队列，队列继续执行。

<a id="coverage"></a>
## 覆盖情况

覆盖运行测试矩阵未覆盖的内容：其他 LyCORIS 算法（LoCon、分解式 LoKr、BOFT、GLoRA、IA3、DyLoRA、T-LoRA、dropout）、优化器（8-bit AdamW、Prodigy、schedule-free、Muon）、EMA、时间步采样器、huber 损失、x0 加权、调度、Lightning 驱动器、块编译、LyCORIS torch 后端、训练期间的导出 / 采样 / 验证、分辨率规划器和批处理方式、翻转 / 随机裁剪、手动内存选项、不启用梯度检查点的流式传输、可分页主机内存、fp8 weight-only 基座、编码器卸载 / 流式传输下的在线缓存、sm89+ 上的 fp8 变体（16 位输入梯度、fp8 文本编码器）、检查点 + 恢复，以及主机 RAM 能容纳主权重与状态时的直接全量微调。报告将 `plan/coverage.py` 的 `FEATURES` 中的每个功能映射到其运行或测试。

<a id="host-ram"></a>
## 主机 RAM

一台主机上的并发运行共享该主机的空闲内存（`plan/hostmem.py`）。

- **预算：** /proc/meminfo 的 `MemAvailable`，其中已扣除每个进程的常驻、页锁定和共享用量（包括主机上的其他任务），再减去 `host_ram_reserve_gib`（16）及最近 3 分钟内启动的运行所占份额（尚未驻留）。没有 /proc 时：取 `ram_gib` 的 `host_ram_share`（0.6）。
- **一次运行的份额：** 进程基线（8 GiB），加上配置档启用流式传输时的流式传输基座权重、启用激活卸载时卸载的激活（token 数 x 每个 token 保存的元素数 x 深度 x 2 字节：8192 个 token 下，不启用梯度检查点时约 42 GiB，使用块级梯度检查点时约 2 GiB），以及全量微调时的主机优化器状态和梯度。
- **准入：** 运行等待直到其份额能够放入预算；如果运行所需份额大于整个预算，则等待直到没有其他运行持有任何份额。

<a id="release"></a>
## 发布

`priorities` 包含 0 的主机会运行发布阶段，而不是其池队列（`run/release.py`）：LoRA r32 `balanced` 和 `minimal`、LoKr 16 `balanced`、低显存 LoKr 设置、fp8-compute LoRA r32 `balanced`（sm89+），以及采用 `vram_8g` / `vram_6g` 层（configs/moss/vram，其中的内存配置档、token 预算和编码器放置方式）、分配器上限为 8 / 6 GB 的 LoRA r32，以上均使用离线缓存；另加不使用缓存的 LoRA r32 `balanced`。

报告的 "Release" 表（`report/build.py` 的 `release_rows`）包含吞吐量、每步编码器秒数、稳定步骤峰值，以及预热阶段（进程启动到第一步）相对步骤阶段的对比（`report/results.py` 的 `phase_peaks`，GiB）：

- 每个阶段的 nvidia-smi 峰值（进程；WDDM 下取显卡相对空闲基线的增量）；
- torch 最大已分配 / 预留量：预热阶段的值（`prewarm_peak_alloc` / `prewarm_peak_reserved`，在第 1 步记录）和步骤阶段的值（取第 1 步自身峰值——从其 `VRAM peak: prewarm ...; step 1 ...` 日志行解析——与之后每步指标行峰值的较大者：`vram_step_peak`、`vram_peak_alloc`、`vram_peak_reserved`；排除第 1 步的指标行，因为它的窗口可能包含预热）；
- "prewarm > step"：任一预热峰值超过步骤阶段的峰值时为 yes。调优预热受内核预热的计划最大值重放约束；在发布行上，其峰值等于步骤峰值，或高出不超过 0.8 GiB（minimal LoRA 行：RTX 5090 上为 3.1，对比 2.8 GiB）。峰值与峰值比较：如果一次运行中各打包序列的长度变化，取各步峰值的中位数会低估该运行。

离线缓存按主机显卡解析出的每种编码器 dtype 构建一次（`knobs.encoder_dtype`：显卡支持时为 bf16，否则为 fp16；VAE dtype 和文本编码器 dtype 都是每个缓存键的一部分），在对应 dtype 的显卡上将两个编码器都固定放置于该卡后构建。每种 dtype 的标记（`bench-cache-<dtype>.json`）记录数据块、dtype 和缓存键布局（`CACHE_LAYOUT`），因此其中任何一项变化都会触发重建。

<a id="disk"></a>
## 磁盘

`run/disk.py`，按池处理：

- **清理**（`cleanup`，启用）：某次运行及所有读取其目录的排队运行（其选项中使用 `{run:<name>}`：导出读取质量运行的检查点，网格图和推理检查读取导出，恢复运行读取检查点运行）都结束后，删除该次运行的 `train/checkpoints/` 和其导出的 `.safetensors`。保留结果、指标行、日志、跟踪、nvidia-smi 样本和 ncu 报告。上一次启动已完成的运行，会在构建池时、任何显卡启动之前交给清理器。不会触及主机运行目录之外的任何内容。
- **空闲空间下限：** 只有运行磁盘至少有 `max(disk_floor_gib, 2 x its checkpoint
   estimate)` 空闲空间时才启动一行（50 GiB；检查点估计值在全量微调时为 `full_checkpoint_gib`，60，否则为 `adapter_checkpoint_gib`，2）。低于该值时显卡暂停，只报告一次暂停状态（`status.json` 中的 `held`，一行日志），并每隔 `disk_poll_s`（60 s）重新检查。
- **重新运行：** 在前一次尝试用过的目录中再次执行运行时，先将前一次尝试的 `train/` 移动到 `train.attempt-<n>/`（删除其中的检查点），使新指标行从空文件开始。

<a id="resume"></a>
## 恢复

在 `start` 中使用 `--option resume=true` 会跳过该基准测试中 `result.json`（主机任意显卡上）以最终状态结束的所有运行（ok、oom、config、nonfinite、retuned、profile-failed）；崩溃、超时或被中断的运行会重新执行，其旧记录变为 `result.superseded.json`。运行器故障（磁盘已满、文件丢失）记为 `runner-error` 并停止该主机；崩溃或超时只进入待审查队列（[失败](#failures)）。

<a id="vram"></a>
## 显存

一个每 500 ms 采样的 nvidia-smi 采样器面向整台主机运行（两个 `--loop-ms` 进程）。对每次运行，它将子进程的已用内存及其显卡的已用内存写入 `vram_smi.csv`，包括：

- 进程峰值和稳定值（后半段样本的中位数）；
- 显卡峰值，以及相对启动前测得的空闲基线的显卡峰值和稳定值。Windows WDDM 下，逐进程列为 [N/A]，使用显卡相对基线的增量代替；
- 对于训练运行，还在第一步开始时（第一行指标的 `time` 减去其 `step_time`）作相同的划分，将启动 + 预热阶段和步骤循环分开，各自具有独立的峰值与稳定值（`vram_smi_phases`），使预热尖峰不会掩盖循环本身的用量。

指标行还加入 torch 的已分配 / 预留峰值、每步峰值（`vram_step_peak`）和设备使用量分解（[logging.md](../guides/logging.md#vram)）。

显存上限运行将缓存分配器限制为目标减去 `cap_allowance_gib`（1.5），nvidia-smi 峰值保持在目标以内（含目标）时即通过。

<a id="failures"></a>
## 失败

每次运行的进程在退出前对异常分类（`run/child.py`）：

| 分类 | 含义 | 影响 |
| --- | --- | --- |
| `oom` | 内存不足 | 记录：该设置无法放入显卡 |
| `config` | 选项被拒绝（验证、缺少可选包） | 记录并加入待审查队列；队列继续执行 |
| `crash` | 模块内部的任何其他异常或断言失败 | 该行停止并加入待审查队列；队列继续执行 |
| `timeout` | 运行超出其超时时间（池的首次运行使用调优超时时间） | 与 `crash` 相同（向运行进程组发送 SIGTERM，一分钟后发送 SIGKILL；ncu 分析通过 sudo 执行） |
| `terminated` | 运行收到来自运行器外部的 SIGTERM（排空） | 非最终状态：恢复基准测试时会重新运行 |
| `runner-error` | 主机运行器本身在运行前后发生故障（磁盘已满、文件丢失） | 主机停止：其他显卡完成当前运行，不再开始新运行 |

每次运行都以自身进程组的组长身份启动（Linux 上为新会话，Windows 上为新进程组：`proc.session`）。超时时向整个组发送信号；运行退出后，组中剩余的任何进程（比训练器存活更久的数据加载器 worker 或 forkserver）都会被终止（`proc.reap`）。Windows 上，超时会终止整个进程树。

结束时存在非有限值步骤的训练运行记为 `nonfinite`，内核或适配器预热对任何内容进行了计时的重启运行记为 `retuned`，失败的 ncu 进程记为 `profile-failed`：加入待审查队列，队列继续执行。不会重试任何内容。`review-queue.jsonl` 保存每个条目及其回溯、栈帧和运行目录。

<a id="results"></a>
## 结果

`<root>/runs/<bench>/<host>/`：`cards.json`、`staging.json`、`probes.json`、`prompts.json`、`queues.json`（按池）、`status.json`、`kernels.json`（按型号）、`review-queue.jsonl`、`tuning/<model>/` 下的调优表，以及每张显卡的 `<card>/<run>/`，其中包含 `config.py`、`log.txt`、`train/metrics.jsonl`（训练运行）、`train/ranks/`（各 rank 的指标行、摘要和性能分析跟踪）、`vram_smi.csv`、`failure.json`、`ncu.*`（ncu 分析）和 `result.json`（摘要：各项指标的中位数、最大值、步骤分解、内核和适配器预热行、损失、导出、nvidia-smi 峰值及稳定值）。

<a id="report"></a>
## 报告

按显卡型号，依次列出：

| 章节 | 内容 |
| --- | --- |
| 探测 | 每张显卡探测出的上限 |
| 缓存模式 | samples/s、tokens/s、TFLOP/s、文本编码器和 VAE 每步秒数及墙钟时间占比、每步 torch 峰值的中位数、预热阶段和步骤循环的 nvidia-smi 峰值 / 稳定值、fp16 损失缩放值、非有限值步骤 |
| 速度 | 显卡、samples/s、s/step、tokens/s、标称 MFU、实测 GEMM 速率占比、nvidia-smi 峰值 / 稳定值、已分配 / 预留量、每步计算 / 复制 / 空闲 ms、每步移动的 GB、传输占比、瓶颈分类、预热时调优 / 从表读取、非有限值步骤、首个 -> 最后一个损失 |
| 显存目标 | 通过 / 失败 |
| ncu | 效率表和主要内核 |
| 质量 | 质量、网格图和推理 |
| 内核 | 对每个已调优内核：问题、选定配置，以及在表中对 PyTorch / compile 路径计时的情况下，选定配置相对该路径的加速比 |

随后是功能覆盖情况（`plan/coverage.py` 的 `FEATURES`：正常 / 损坏 / 无法放入显存 / 未运行，或来自 `--junit` XML 的测试通过 / 失败 / 已存在）以及每个失败。

速度运行的瓶颈分类：

- 传输占比：(h2d GB / 页锁定 h2d GB/s + d2h GB / 页锁定 d2h GB/s) / 步骤时间，涵盖 DiT 引擎（`xfer_*_gb`，[logging.md](../guides/logging.md#transfer)）和文本编码器 / VAE 的权重流量；
- 计算占比：模型 TFLOP/s（kohakumoss/model/flops.py）/ 探测的 GEMM 速率（fp8 基座使用 fp8）；
- 传输占比达到 `--transfer-bound`（0.7）或以上时为传输受限；否则，计算占比达到 `--compute-bound`（0.5）或以上时为计算受限；否则为软件受限：链路和计算单元都不是限制因素。

<a id="nominal-peaks"></a>
### 标称峰值

标称 MFU（训练器的 `mfu`、报告的 MFU 列）用模型 TFLOP/s 除以 `kohakumoss/model/flops.py` 的 `PEAK_TFLOPS`，通过设备名称的子串匹配选择（第一个匹配项优先）。每个值都是 FP16 / BF16 张量核心输入、FP32 累加下的稠密峰值（无 2:4 稀疏），与训练器矩阵乘法使用的精度一致：

| 键 | TFLOP/s | 依据 |
| --- | --- | --- |
| `V100`, `PG503` | 125 | Tesla V100 数据表，SXM2 张量性能 |
| `2080 Ti` | 53.8 | Turing 架构白皮书，FP16 张量与 FP32 累加（FP16 累加时为 107.6） |
| `Tesla T4` | 65 | T4 数据表，FP16 / FP32 混合精度 |
| `Tesla T10` | 68.8 | NVIDIA 未公布张量性能数值；取列出的 FP32 峰值 8.6 TFLOPS 的 8x（3072 个 CUDA 核心，384 个张量核心，约 1.40 GHz），即 T4 数据表中的 FP16 张量与 FP32 之比（65 / 8.1） |
| `3060` | 25.6 | 112 个张量核心，1.78 GHz boost 时钟，FP32 累加为半速，与 GA102 GeForce 显卡相同 |
| `3090` | 71 | Ampere GA102 白皮书，FP16 张量与 FP32 累加（稀疏时为 142） |
| `4090` | 165.2 | Ada Lovelace 架构白皮书，FP16 张量与 FP32 累加（稀疏时为 330.4） |
| `5090` | 209.5 | RTX Blackwell 架构白皮书，FP16 张量与 FP32 累加（稀疏时为 419） |
| `RTX PRO 6000 Blackwell` | 500 | RTX PRO 6000 Blackwell Server Edition（Colab G4）数据表：稀疏 FP4 为 4 PFLOPS，因此稀疏 FP16 为 1000，稠密为 500（专业卡的 FP32 累加为全速）；数据表未列出 FP16 行。Workstation Edition 的时钟更高（FP32 为 125，对比 120 TFLOPS），比此值约高 4% |
| `A100` | 312 | A100 数据表，BF16 / FP16 张量（稀疏时为 624） |
| `H100 PCIe` | 756 | H100 数据表，PCIe，BF16 / FP16 张量（稀疏时为 1513） |
| `H100` | 989 | H100 数据表，SXM，BF16 / FP16 张量（稀疏时为 1979） |
| `NVIDIA L4` | 121 | L4 数据表，BF16 / FP16 张量（稀疏时为 242） |

对于 fp8 基座（fp8_compute、MXFP8），报告增加 "MFU (fp8 nominal)"：用相同的 TFLOP/s 对比 16 位峰值的两倍，这是 Ada Lovelace（4090：330.3，对比 165.2）和 RTX Blackwell（5090：419，对比 209.5）白皮书中，FP32 累加下的 FP8 与 FP16 张量性能之比。"of GEMM peak" 将 fp8 运行与探测中实测的 fp8 `_scaled_mm` 速率比较。

设备名称未匹配任何键时，不报告 MFU。基准测试的计算占比改用探测中实测的 GEMM 速率，因此不依赖本表。
