[English](../../guides/logging.md)

<a id="logging-and-progress"></a>
# 日志与进度

KohakuMoss 通过标准库 `logging` 记录日志：每个模块一个日志记录器，统一的行格式，
由入口点一次性配置。训练显示一条进度条；采样为每个批次显示一条覆盖采样器步骤的
进度条，解码时再显示一条。

```
[12:00:00] [inference.pipeline] [INFO] estimated peak 9.60 GiB while sampling [dit_gib=6.2, tokens=8704]
[12:00:03] [rank1] [training.entry] [INFO] training from step 0 to 1000
```

<a id="settings"></a>
## 设置

| 设置 | 效果 |
| --- | --- |
| `KOHAKUMOSS_LOG_LEVEL` | 由 `setup()` 读取：`DEBUG` / `INFO`（默认）/ `WARNING` / `ERROR` 或数字 |
| `FORCE_COLOR=1` | 在管道或文件中使用 ANSI 颜色（例如将输出同时写入文件的包装脚本）；优先于 `NO_COLOR`，因此即使环境全局导出了 `NO_COLOR`，仍可按命令覆盖 |
| `NO_COLOR`（任意值） | 不使用 ANSI 颜色 |
| stderr 不是终端 | 不使用 ANSI 颜色 |
| Windows 控制台 | `setup()` 为 stderr 控制台开启 VT 处理 |

除非级别为 DEBUG，`setup()` 会应用第三方库的最低日志级别（`NOISY_LIBRARY_LEVELS`）：
diffusers 和 transformers 为 ERROR，huggingface_hub、urllib3 和 httpx 为
WARNING。

训练配置项：

| 配置项 | 默认值 | 设置内容 |
| --- | --- | --- |
| `LOG_EVERY` | 10 | 控制台步骤日志行（仅在关闭进度条时）；`METRICS_EVERY` 的默认值 |
| `METRICS_EVERY` | None（= `LOG_EVERY`） | 向接收端（`metrics.jsonl`、KohakuBoard、wandb）写入指标行，以及填充这些行的吞吐量、显存和编码器占比监视器；1 = 每步 |
| `PROGRESS` | True | 进度条 |
| `LOGGER` | `console` | 除 `metrics.jsonl` 外的接收端（[接收端](#sinks)） |

<a id="what-a-training-run-shows"></a>
## 训练运行显示的内容

<a id="the-bar"></a>
### 进度条

一条覆盖 `STEPS` 的 tqdm 进度条（rank 0），每步更新后缀：由 EMA（beta 0.9）
平滑的损失、lr、grad_norm、s/it、samples/s、tokens/s，以及 `vram` = 该步的
已分配峰值 / 预留峰值 / 设备占用，单位 GiB（[显存](#vram)）；tqdm 自身的
字段显示预计剩余时间。

- 当 `PROGRESS=True`（默认）、stderr 是终端或设置了 `FORCE_COLOR=1`，
  且日志级别为 INFO 或更低时，开启进度条。
- 显示进度条时，控制台不会打印逐步日志行：数值由后缀承载。事件日志行
  （`LOG_EVERY` 日志行、采样、验证、保存、导出、警告、调优器日志行）
  通过 `logging_redirect_tqdm` 打印在进度条上方。
- 控制台步骤日志行 `step N [loss=, grad_norm=, lr=]` 仅在关闭进度条时打印
  （无终端且未设置 `FORCE_COLOR`、`PROGRESS=False`，或级别高于 INFO）。
- 这一选择在构建时确定一次：运行会得到进度条或 StepLine
  回调之一。

进度条与日志行一样写入 stderr，级别高于 INFO 时隐藏（采样时也可通过
`generate(..., progress=False)` 隐藏）；进度条开启期间打印的日志行通过
`tqdm.write`（覆盖各软件包根的 `logging_redirect_tqdm`）输出，因此互不干扰。

<a id="cadence"></a>
### 频率

- 两种频率（`LOG_EVERY`、`METRICS_EVERY`）也会在运行的第一步（恢复后：
  恢复后的第一步）和最后一步触发（带有 `edges = True` 的回调）。
- 频率在构建时一次性解析为各回调的 `every`；每步开销
  只有运行器的一次整数比较。
- 每条发出的指标行需要一次主机同步（将该步的设备指标读为
  浮点数），加上监视器的主机侧工作，因此 `METRICS_EVERY=1` 每步增加一次同步；
  进度条本身只读取主机值。

<a id="sinks"></a>
### 接收端

每次运行都会写入 `metrics.jsonl`，`LOGGER` 用于选择除此之外的接收端：

| `LOGGER` | 接收端 |
| --- | --- |
| `console` | 无其他接收端；控制台显示进度条或步骤日志行 |
| `board` | 位于 `<output>/board` 下的 KohakuBoard |
| `wandb` | 使用 wandb 自身默认模式的 wandb 运行（设置了 `WANDB_MODE` 则使用该值，否则在线），写入运行目录下；`WANDB_MODE=offline` 使其保留在本地（之后可用 `wandb sync` 同步） |

无论使用哪种日志记录器，控制台显示的内容都与 `LOGGER="console"` 完全相同：
wandb 静默运行，不捕获控制台输出；KohakuBoard 将自己的日志行写入
`<output>/board/kohakuboard.log`，不捕获 stdout / stderr。

<a id="fresh-and-resumed-runs"></a>
### 新运行与恢复运行

- 如果不恢复的运行（`RESUME=""`，或 `"latest"` 但尚无检查点）使用了已有
  `<output>`，则首先将先前尝试的 `metrics.jsonl`、`log.txt` 和 `board/`
  移到首个可用的 `metrics.<n>.jsonl`、`log.<n>.txt`、`board.<n>`，
  每次移动都会发出一条 WARNING（`fresh run: an earlier metrics.jsonl in <output> moved to
  <output>/metrics.1.jsonl`）。
- 恢复的运行会追加写入这些文件。`metrics.jsonl` 中恢复步骤之后的行
  （某次尝试在该检查点后继续运行留下的行）会先移到 `metrics.<n>.jsonl`
  （`resumed at step <s>: <n> later rows of <path> moved to <path>`），因此文件中
  每步只保留一行。

<a id="other-training-lines"></a>
### 其他训练日志行

- 运行摘要；
- 模型 / 适配器 / 编码器的构建及其内存计划；
- 随后的文本编码器与 VAE 放置日志行（[推理日志行](#what-sampling-shows)；
  VAE 也可显示为 `stream (... 2-layer ring on
  cuda:0 ...)` 或 `cpu (... runs on the host)`；运行未保留这些组件时，显示
  `text encoder: not kept, TEXT_CACHE='offline'` / `VAE: not loaded, ...`）；
- 内核预热（逐算子进度日志行、各算子及总计的墙钟时间）和
  适配器预热；
- 数据加载器构建、恢复、每隔 `LOG_EVERY` 步的一条日志行；
- 第 1 步的 s/step 与第 2-10 步中位数的对比（预热的收益）；
- 每次导出。

<a id="what-sampling-shows"></a>
## 采样显示的内容

推理（`MossPipeline`、`scripts/sample/sample.py`、`compare_grid.py`）会记录：

- 各组件读取过程（transformer 的张量、GiB 和秒数；读入主机内存的文本
  编码器和 VAE 权重及耗时秒数；transformers 自身的加载进度条关闭），随后
  为每个组件输出一条放置日志行，说明权重存放位置、计算位置及大小
  （`kohakumoss.inference.placement_report`、
  `kohakumoss.offload.encoders.report`），再输出一张放置表，列出使用 CFG 的
  单张 1024x1024 图像的预计 GPU 峰值：

  ```
  transformer: <path> -> resident on cuda:0 (9.63 GiB, fp16, fp16 fold c=1/16 from metadata)
  transformer: <path> -> streamed from host, 16 of 42 blocks resident (4.10 GiB on cuda:0, 5.53 GiB host; bf16)
  text encoder: <path> -> ring (weights in host RAM, pinned, 14.80 GiB; ring of 2 layers on cuda:0, ~0.70 GiB, 0.20 GiB resident; precision fp8, bf16)
  VAE: <path> -> offload (host RAM, moved to cuda:0 per pass, 0.23 GiB)
  placement: estimated GPU peak 12.10 GiB (one 1024x1024 image, CFG)
    component    | placement | GPU GiB | host GiB
    -------------+-----------+---------+---------
    DiT          | resident  |    9.63 |        0
    text encoder | ring      |     0.9 |     14.8
    VAE          | offload   |       0 |     0.23
  ```

  主机侧工作的后端选择（在 CPU 上量化文本编码器的线性层）以 DEBUG
  记录；
- 后端选择和内核决策（[内核调优](#kernel-tuning)；训练
  记录相同的内容）；
- 提示词编码（提示词数、token 数、秒数）、采样计划（图像数、批次数、步数、
  cfg、尺寸）、每个批次覆盖采样器步骤且带有当前 sigma 的进度条、VAE 解码
  （多张图像时显示进度条）、总耗时，以及 generate 的显存：已分配峰值、
  预留峰值和设备占用（`GenerateResult.settings["vram"]`，也会以 `run.vram`
  写入采样脚本的元数据）；
- 采样脚本写出的每个文件。

`scripts/sample/sample.py` 从 `OUT` 中现有的最大 `NNNN_` 索引之后开始为图像编号，
并为每次运行写入一个 `metadata-<first index>.json`，因此复用 `OUT` 会保留
之前所有图像及其设置。

<a id="kernel-tuning"></a>
## 内核调优

| `TUNE` | 效果 |
| --- | --- |
| `table`（训练默认） | 绑定该显卡已存储的决策；表中没有的问题会调优一次并存储（训练：在第 0 步前的预热中，对计划的形状和 dtype 执行前向 + 反向，之后释放工作区；随后仅有文本编码器预热、前 `RETUNE_STEPS` 步的内核与 compile 对比重新调优，以及首次采样网格的第一个采样器步骤；推理：问题首次使用时） |
| `off`（推理默认） | 绑定该显卡已存储的决策，否则使用默认配置；不计时也不存储任何内容 |

- `kohakumoss-tune --device cuda:0 --dtype bf16 --sizes 1024x1024 832x1216`（或
  `kogine run scripts/bench/tune/tune.py`）会为一张显卡、一种 dtype 和一个尺寸列表
  一次性填充调优表（训练形状使用 `--backward`）；该表位于内核缓存下
  （`KOHAKUMOSS_KERNEL_CACHE`，默认为 `~/.cache/kohakumoss_kernels/tuning/<card>/`）。
- 拥有自身模型的流水线会将 `MossPipeline.from_pretrained(tune=...)` 应用为
  进程的 TUNE；共享训练模型的流水线（`from_components`）则使用训练器的设置。
  见 [../internals/tuning.md](../internals/tuning.md#tune)。

<a id="backend-lines"></a>
### 后端日志行

后端选择对每个算子、架构和 dtype 只记录一次（`attention_varlen on sm89/bfloat16:
backend triton`）。回退日志行
（[kernel-dev.md](../internals/kernel-dev.md#fallback)）：

| 情况 | 日志行 |
| --- | --- |
| 后端失败 | 每次重新绑定一条 WARNING：`<op> on <arch>/<dtype>: backend <failed> failed, using <next> (<n> bound ops): <error>` |
| 因先前运行记录的工具链失败而跳过后端 | `backend <b> skipped, failed here (<error>)` |
| 文本编码器算子的探测失败 | `text encoder op <name> on <device>: backend <b> failed its probe, using eager: <error>` |

<a id="decision-lines"></a>
### 决策日志行

内核决策（`kohakumoss_kernels.tuning.decisions`）：每个经过调优的内核及内核与
compile 对比问题，在配置确定时记录一次：来自显卡调优表、实测，或未计时的
默认值（`TUNE="off"` 且没有存储行，或只有一个候选）。

- **DEBUG，逐问题：** 键、配置和计时
  （`choice_triton_rms_norm_mod 4096x3072bfloat16|...: compile (kernel 0.21 ms, compile
  0.17 ms), measured`）。并非决策的绑定（排队等待重新调优、期间先运行自身
  内核的内核与 compile 对比问题，或在该进程中未计时的键）只记录
  `undecided <kernel> <key>`。
- **INFO，逐批决策：** 一张表，在预热后、实际测量了内容的重新调优后
  （训练：前 `RETUNE_STEPS` 步）、训练第 1 步结束时，以及 generate 之后记录。
  `x` 列统计采用各选定配置的问题数：

  ```
  after a retune (0.41s): 3 kernel decision(s)
    kernel                     | from table | measured | inherited | default | chosen  | x
    ---------------------------+------------+----------+-----------+---------+---------+--
    choice_triton_rms_norm_mod |          2 |        1 |         0 |       0 | compile | 2
                               |            |          |           |         | kernel  | 1
  ```

- **重新调优结束（训练）：** 在 `RETUNE_STEPS` 的最后一步之后，输出一条 INFO 日志行：
  `kernel-vs-compile retune closed after step N of this process: M queued
  kernel-vs-compile problem(s) left on the kernel`。

- **第 1 步之后：** 新测量的问题（预热遗漏的问题）会输出一条 INFO 日志行，
  包含键、选择结果和计时；第 1 步之后的其他决策保持 DEBUG 级别。
- **WARNING：** 首次在调优窗口之外遇到的键会不经计时直接绑定，并输出一条日志行
  （`met outside a tuning window`）。
- **采样：** generate 会绑定显卡已存储的决策，否则使用默认值；在
  `TUNE="table"` 下，会在缺失决策首次使用时调优并存储，且表中会列出
  首次使用时和调用后测量的问题。

<a id="vram"></a>
## 显存

进度条中的 `vram=9.6/10.7/12.7G` 表示每步的：

| 数值 | 含义 | 数据来源 |
| --- | --- | --- |
| 已分配峰值 | 该步存活张量的最高字节数 | 峰值累加器上显存时间线的步骤窗口 |
| 预留峰值 | 该步 torch 缓存分配器从驱动持有的最高字节数 | 同一个窗口 |
| 设备 | 设备占用，与 nvidia-smi 显示的一致（所有进程，包括本进程的 CUDA / XPU 上下文） | 驱动：`mem_get_info`（总量 - 空闲量），最多每秒重新读取一次 |

这三者均为主机侧读取：每次窗口重启读取一次分配器统计信息，每秒
调用一次 `cudaMemGetInfo`。

报告、基准测试和元数据都携带这三个带标签的数值：

- `peak_alloc`：存活张量；
- `peak_reserved`：torch 分配器持有量；
- `device_used`：整个设备、所有进程（即 nvidia-smi 显示的内容）。Windows
  （WDDM）上，nvidia-smi 显示整个进程，其数值为第三个数加上其他进程的
  后台占用。

文档表格中仅标为“峰值 GiB”的列，使用已分配计数器测量。

<a id="one-peak-accumulator"></a>
### 单一峰值累加器

项目中的所有峰值（进度条、指标行、预热和第 1 步报告、`generate` 的显存记录、
基准测试行）都是同一个进程级峰值累加器
（`kohakumoss_kernels.hardware.memory`）上的窗口：

- `with window(device) as w`（或 `open_window` / `close_window`）给出窗口
  开启期间的已分配 / 预留字节峰值。
- 每次开启、读取和关闭，都会先将分配器的峰值计数器并入该设备的所有已开启
  窗口，再重置计数器，因此嵌套测量（训练步骤中的采样 `generate`、generate 中的
  预热重放、编码器的阶段标记）不会破坏任何外层窗口的完整性。

<a id="device-use-breakdown"></a>
### 设备占用分解

设备占用与 torch 数值之间的差距，会在构建后、预热后和第 1 步之后以表格记录，
并写入指标行（`vram_device`、`vram_other`、`vram_context`、`vram_non_torch`、
`vram_fragmented`、`vram_cached`、`vram_allocated`，单位 GiB）：

| 部分 | 内容 |
| --- | --- |
| `other` | 设备上的其他进程（以及驱动自身预留量），在本进程第一次设备调用前通过 NVML 测量；没有 NVML（pynvml）时为 0，此时由 `context` 包含这一部分 |
| `context` | CUDA / XPU 上下文（驱动与运行时），按第一次设备调用新增的占用测量；没有 NVML 时也包含 `other`（第一次设备调用时的设备占用） |
| `non_torch` | 设备占用减去 other、context 和 torch 预留量：已加载的内核模块（torch 自身延迟加载的 CUDA 内核、cuBLAS / cuBLASLt / cuDNN 库状态、Triton / TileLang cubin）、NCCL 缓冲区，以及所有在 torch 缓存分配器之外分配的内容 |
| `fragmented` | torch 预留量中拆分块内部的空闲部分（`inactive_split_bytes`） |
| `cached` | torch 预留量中的完整空闲块（由 `empty_cache` 归还） |
| `allocated` | 存活张量 |

- 这些部分按定义加总为 `device`，而 `device` 与进度条第三个数值和
  `device_used` 使用相同的驱动数值。
- cuBLAS / cuBLASLt 工作区计入 `allocated`：torch 通过其缓存分配器分配它们
  （torch 2.14、V100：第一次 fp16 1024x1024 矩阵乘法新增 8.1 MiB 已分配量，
  可由 `torch._C._cuda_clearCublasWorkspaces()` 释放；另在分配器之外为库内核
  和句柄新增 10 MiB 设备占用）。
- 若在测量 `other` 后其他进程释放或占用内存，差值会落入 `non_torch`
  （此时它可能变为负数）。
- 默认情况下，训练器在步骤之间保留 `cached` 内存
  （`STEP_RELEASE="none"`，[../internals/offload.md](../internals/offload.md#between-step-release)）：
  分配器在下一步复用这些内存，因此第 1 步后的表格显示的是运行的稳态
  预留内存。

<a id="transfer"></a>
## 传输

带有 DiT 侧卸载的运行会在指标行中添加 `xfer_h2d_gb` / `xfer_d2h_gb`：
每步在主机与设备间搬移的 GB 数，对记录自身字节数的各引擎加总
（权重流式传输、梯度卸载、激活卸载、主机状态优化器和 CPU AdamW
优化器）。

- 计数器是主机整数：每个引擎将其发出的每次复制的字节数加到自身
  `stats` 字典（`bytes_h2d` / `bytes_d2h`；激活卸载使用 `packed_bytes`，
  传出和传回的字节数相同），回调在各日志点读取这些字典。
- 文本编码器和 VAE 分别报告自身的流量（`te_h2d_gb`、`vae_h2d_gb`）。
- 将它们除以步骤时间，并与链路实测带宽比较，即可说明传输需要占用
  步骤的多大比例（基准测试矩阵报告的传输占比，`scripts/bench/matrix/`）。

带有梯度卸载或由主机执行更新的权重流的运行，还会添加
`grad_offload_wait_s`：每步反向传播等待共享梯度暂存环形缓冲区
（`HostAccumulator`，4 个槽位）中忙碌槽位的主机秒数，仅在槽位繁忙时测量；
若数值接近步骤时间，说明主机加法限制了反向传播。

<a id="log-format-and-api"></a>
## 日志格式和 API

<a id="line-and-colour"></a>
### 日志行与颜色

- **日志行。** `[HH:MM:SS] [module.name] [LEVEL] message [key=value, ...]`。模块
  列会去掉软件包根（`kohakumoss.inference.pipeline` 显示为
  `inference.pipeline`，`kohakumoss_kernels.tuning.tuner` 显示为 `tuning.tuner`），
  仅在仍然过长时通过前置 `...` 截短为 26 个字符；分布式运行中，`[rankN]`
  紧随时间。字段中的浮点数以四位有效数字打印。
- **颜色。** 每个 token 都有自己的颜色。每段都以重置（`0`）结束。

| token | 颜色 |
| --- | --- |
| 时间 | 灰色（`90`） |
| rank | 蓝色（`34`） |
| 模块 | 洋红色（`35`） |
| 级别 token | 对应级别的颜色：DEBUG 青色 `36`、INFO 绿色 `32`、WARNING 黄色 `33`、ERROR 红色 `31`、CRITICAL 红底粗体白字 `1;37;41` |
| 消息 | 终端颜色；WARNING 及以上染为对应级别颜色（CRITICAL：红色） |
| 字段键 | 暗青色（`2;36`）；`=` 无样式 |
| 字段值 | 粗体（`1`） |
| 数字及带单位的量（GiB、MiB、s、ms、it/s、s/step、tokens、x、%） | 亮黄色（`93`） |
| 路径 | 带下划线的蓝色（`4;34`） |

- **高亮。** `hl(text, kind="value")` 包装消息的一段：`value` 选择字段
  规则（数字为黄色、路径为蓝色、其他为粗体），另有 `path`、`ok`（亮绿色）、
  `bad`（亮红色）。除非上一次 `setup()` 的控制台处理器输出颜色，否则返回
  纯文本；文件处理器会剥离每段样式，因此日志文件始终是纯文本：
  `log.info(f"loaded {hl(path, 'path')} in {hl(f'{secs:.1f} s')}")`。加载、计划、
  峰值、调优进度、generate 和步骤日志行会使用它，并将数值作为
  文本段或字段携带。

<a id="setup"></a>
### 设置

- **库代码** 仅调用 `get_logger(__name__)`（来自 `kohakumoss_common.log`，或
  添加了进度条的 `kohakumoss.logging.log`）；处理器由入口点的 `setup()`
  提供。在入口点调用它之前，记录会传播到 Python 根日志记录器
  （因此嵌入应用自身的日志配置可以接收它们）。
- **入口点**（`kogine` 脚本、`kohakumoss-config`、`kohakumoss-tune`、
  训练器及其 DDP worker）调用一次 `setup(level=None, stream=None, path=None,
  rank=None, world=1)`：
  - 它是幂等的：会替换自己先前附加的处理器。
  - 当 `world` > 1 时，日志行带有 `[rankN]`，且仅 rank 0 写入。
  - 它为三个软件包根 `kohakumoss`、`kohakumoss_kernels`、`kohakumoss_data`
    附加 stderr 处理器，每条记录都会刷新（控制台代码页无法表示某字符时
    写入替代字符），并停止它们的传播。
  - `path` 添加无颜色的文件处理器；`add_log_file(path)` 可稍后添加，级别为
    `setup` 设置的级别（训练器的 rank 0 从运行目录创建起写入 `<output>/log.txt`）。
    `set_level(level)` 可稍后更改级别，包括各处理器的级别。
- **单一实现。** `kohakumoss_common/log.py`，位于其他所有包都会导入的基础包中。
  `get_logger` 返回对应名称的标准库日志记录器的包装器
  （标准库日志记录器保留自己的类）。

<a id="fields-records-and-tables"></a>
### 字段、记录和表格

- **结构化字段。** `log.info("transformer loaded", gib=9.6, secs=27.1)`：关键字
  参数成为记录属性，并追加到日志行；与 `LogRecord` 属性
  （`name`、`msg`、`module`、...）冲突的名称会加 `_` 后缀。记录的文件和行号
  来自调用方，即使经过 `timed(log, what)` 也是如此。
- **记录。** `log_object(log, title, obj, only=None, suffix="")` 将
  dataclass 或映射（MemoryPlan、TextEncoderPlan、一组配置项）记录为一条标题行，
  即 `title: <its profile / memory / name / kind>` 加上后缀（`(applied in 17.9s)`），
  随后每个字段一条对齐的 `key  value` 行，嵌套记录和映射缩进。
  与默认值相同的字段在 INFO 下隐藏为一条 `(n defaults hidden)` 日志行，
  在 DEBUG 下显示：

  ```
  [12:00:00] [inference.loading] [INFO] DiT memory plan: balanced on cuda:0 (applied in 17.9s)
    checkpoint      none
    base_precision  fp8_compute
    (12 defaults hidden)
  ```

- **表格。** `log_table(log, title, headers, rows)` 记录 ASCII 表格（`|` 分列、
  `-+-` 分隔线、数字列右对齐）：放置摘要、预计峰值各部分、逐算子预热耗时。
  `render(obj)` 和 `table(headers, rows)` 返回这些日志行。键为暗青色，值使用
  字段颜色，`True` 为绿色，`False` 为暗色；没有彩色控制台时（以及文件处理器中），
  日志行均为纯文本。
