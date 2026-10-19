[English](../../internals/kernel-dev.md)

<a id="kernel-development"></a>
# 内核开发

`kohakumoss_kernels` 包含 MOSS 块算子。每个算子都有一个 eager 参考实现（`backends/eager/`，定义算子的计算内容）以及各架构的实现：

| 架构 | 实现 | 原因 |
| --- | --- | --- |
| sm70 / sm75（V100、2080 Ti） | TileLang | sm80 以下的 Triton 没有张量核心点积（它们会被降级为 FMA） |
| sm80 / sm86 / sm89 / sm120 | Triton（也注册了 TileLang） | Inductor 能看到 `triton_op` 启动 |
| XPU | Triton（Intel 后端），逐算子采用实测更快的实现（[性能](../performance/README.md)） | TileLang 没有 XPU 目标 |
| CPU | eager | |

每个调优内核族都会在其候选短名单中对 PyTorch 路径计时，并按显卡存储优胜者，因此只在内核快于 PyTorch 的地方绑定内核：

| 算子族 | PyTorch 路径 |
| --- | --- |
| 块算子、融合 linear | `compile`（对 eager 参考实现进行 torch.compile），`backends/compile/choice.py` |
| `attention_varlen`（Triton 和 TileLang 入口，`backends/varlen_choice.py`） | 当构建版本和显卡支持 flash 时使用 torch `varlen_attn`，否则使用逐片段 SDPA 循环 |
| `attention_varlen_causal`（文本编码器） | 可用时使用 torch `varlen_attn`（因果），否则使用逐片段 SDPA 循环 |
| `gated_delta_rule_varlen`（文本编码器） | 分块 eager 参考实现（`torch` 配置） |
| `quant_linear` | 反量化 + 矩阵乘法（`quant.PYTORCH_PATHS`） |
| `fp8_linear*` | 厂商的 `torch._scaled_mm` 路径和 16 位反量化路径（`W16`） |
| 优化器步（`adamw_step`、`adamw8bit_step`、`adamw_step_multi`） | 参考步，在临时副本上按键计时一次（`optim/choice.py`） |

XPU 上也注册了 Triton 融合 linear（`linear_situ`、`linear_gated_residual`、`linear_qk_rope`）和 `fp8_linear*` 路径：

- 调优器跳过在 Intel 后端构建失败的每种图块。
- 没有任何候选存活的问题会抛出 `BackendUnavailable`，注册表将算子重新绑定到下一个后端（融合 linear 为 `compile`，fp8 为 `eager`）。
- 没有 fp8 矩阵单元时，在寄存器中反量化的 fp8 路径（`promote`、`fused`）和 16 位路径仍然可用。

<a id="registry"></a>
## 注册表

实现是一个具有算子 eager 签名（`kohakumoss_kernels.ops`）、支持 autograd 和 torch.compile 的可调用对象。

| 调用 | 行为 |
| --- | --- |
| `register(op, backend, fn, supports)` | 添加实现；`supports(query)` 在能够处理 `OpQuery`（算子、架构、dtype、静态选项）时返回 None，否则返回不能处理的原因 |
| `resolve(op, device, dtype, **static)` | 遍历设备架构的后端优先级（`DEFAULT_PRIORITY`，可用 `set_priority` 覆盖），跳过不可用、不支持该查询、在该设备上探测失败，或已有该算子失败记录的后端，返回一个 `Bound` 算子 |
| `available(...)` | 列出每个后端的原因（doctor 诊断视图） |

在 `recording()`（`kohakumoss_kernels.recording`）内部，`Bound` 调用会被记录快照，并由 eager 参考实现返回结果，因此不会构建或调优任何内容；参见 [tuning.md](tuning.md#prewarm)。

<a id="adding-an-op-to-a-backend"></a>
## 向后端添加算子

1. 编写 `backends/<backend>/<group>/<op>.py`（分组：`blocks/`、`attention/`、`textenc/`、`optim/`、`fp8/`）。后端包在导入时会导入其包树中的每个公共模块（以 `_` 开头的文件是辅助模块，会被跳过），因此不需要编辑 `__init__`。
   - 导入时抛出 `ImportError`（缺少可选模块）的子模块会被列入 `backends.<backend>.FAILED`，并记录 WARNING；其余模块仍会注册。
   - 导入子模块时抛出的任何其他错误（其代码中的 `NameError`、`TypeError` 等）都会传播。
   - `tests/kernels/registry/test_backend_imports.py` 断言每个已安装包的后端的 `FAILED` 都为空。
2. 在模块底部调用 `registry.register(op, backend, fn, supports)`，其中 `supports` 通过 `constraints.requires(...)` 构建。`fn` 具有 eager 签名，支持 autograd 和 `torch.compile(fullgraph=True)`，并处理它接受的每种形状（每次构建只解析一次，绝不每次调用都解析）。
3. torch.library 接入：
   - Triton：`torch.library.triton_op` + `register_autograd`；通过 `backends.triton._common.Launcher` 启动（真实张量上直接启动，追踪时使用 `wrap_triton`），以便 Inductor 记录启动。在 torch.compile 之外，块算子通过 `autograd.Function` 调用相同的前向 / 反向主体和 setup（`_common.eager_function`）：自定义算子分派每次前向 + 反向需要几百 us 的主机时间，这对小问题（sm86+ 上 N 为数千 token）具有决定性影响，而 Function 计算相同内容却没有该成本。
   - TileLang：`torch.library.custom_op` + `register_fake` + `register_autograd`。使用 torch 分配输出并传入（`_blockops` 启动接收由调用方分配的输出）：TileLang 分配的张量在自定义算子返回时会丢失 autograd。
4. 图块来自 `tuning.select`（参见 [tuning.md](tuning.md)）。
5. `tests/kernels/` 中的测试：使用 `tests/_util/error_stats.py` 进行 fp16 和 bf16 下的前向与反向误差统计检查，再提供一个该检查必须拒绝的错误变体（`tests/_util/negative.py`）。
6. 在 `scripts/bench/kernel/` 中与所有 PyTorch 替代方案（eager 和编译后）进行基准比较，同时呈现精度与速度（[基准测试](#benches)）。

<a id="per-sample-vectors"></a>
## 逐样本向量

按片段索引的算子（`rms_norm_mod` 的 shift / scale、`gated_residual` 和 `linear_gated_residual` 的 gate）接收骨干网络 fp32 `[B, 6, D]` 块调制和 `[B, 2, D]` 头部调制的 `mod[:, k]` 切片：行步长为 `6 D`（或 `2 D`），而不是 `D`。

- 内核将每个逐样本向量的行步长和列步长作为参数，寻址为 `seg * row_stride + col * col_stride`；不进行复制。
- 只有一个样本时，每一行都读取第 0 行，因此只有含两个或更多样本的打包批次才能暴露布局错误。
- `tests/kernels/blocks/test_modulation_layout.py` 在每个后端上运行这些跨步切片，并与该后端在连续副本上的结果比较。

<a id="tilelang-notes-0115"></a>
## TileLang 说明（0.1.15）

- token 数和片段数为 `T.dynamic`；每个 (heads, head dim, dtype, tile config, arch) 编译一次即可服务所有批次。网格范围可以是标量参数。
- 从符号形状张量执行 `T.copy` 时，会对超出末尾的行进行边界检查（填零）；向符号张量进行元素存储时也会检查边界。
- 对秩 > 2 的张量，在 `T.copy` 中使用切片区域（`X[a : a + bm, head, :]`），而不是起始索引。
- 将一个 GEMM 的累加器移入下一个 GEMM 的 A 操作数时，需要经过共享内存。在两种布局之间进行 fragment 到 fragment 的复制会在每种架构上导致布局推断失败（"Layout infer conflict"），而且 Volta 的 MMA 没有转置 A 形式。
- Warp 划分：每个 warp 拥有每个 `T.gemm` 图块的 16 行和 16 列的整数倍（编译前检查）。
- 循环次数依赖于程序的循环使用 `T.serial`（不做流水化）。
- Windows 上 nvcc 只接受 MSVC；`hardware.toolchain.ensure_toolchain`（首次构建内核时运行，参见[惰性后端](#lazy-backends)）导入 `vcvars64.bat` 环境，并将 `CXX` 设为对应的 `cl.EXE`（参见 [install-windows.md](../guides/install-windows.md#compiler-setup)）。

<a id="fallback"></a>
## 回退

`registry.resolve` 返回一个直接调用其实现的 `Bound` 算子。无法在此处运行的实现会抛出 `errors.BackendUnavailable`：

- 当候选短名单中的所有配置都无法编译或运行时，调优器抛出 `ConfigsFailed`（一个子类）；
- 内核或编译选择算子（`backends/compile/choice.py`）在每设备一次的探测失败时抛出它；
- compile 后端在某设备上首次编译算子失败时抛出它；
- 注意力和文本编码器入口将其调用中的构建 / 启动错误转换为该异常（`errors.unavailable`）。

两类错误不属于后端失败：

- 当 eager 参考实现也拒绝失败调用的参数时，`Bound` 改为抛出 `errors.InputError`：这是调用方错误，不记录任何内容，算子保持绑定。
- 粘滞设备错误（`errors.is_sticky_error`：错误文本包含非法内存访问、未对齐地址、非法指令、设备端断言或未指定的启动失败，或者同步 CUDA 设备时抛出异常）原样传播，不记录，也不重新绑定：上下文已不可用，因此任何下一个后端也会在其上失败（`tests/kernels/registry/test_fallback.py`，反例：关闭该检查）。预热重放也会重新抛出它，而不是跳过该问题（`tests/kernels/tuning/test_replay_sticky.py`）。

<a id="rebind"></a>
### 重新绑定

其他情况下，`Bound` 在 `registry.failures()` 中记录失败，并重新绑定到该架构优先级中的下一个后端，同时也会重新绑定所有查询相同（算子、架构、dtype、静态选项、优先级）且仍位于失败后端的存活 `Bound`：预热孪生模型的算子与模型算子就是这样一对，因此预热中遇到失败时，模型算子也会一同切换。每次重新绑定都通过项目日志器记录一条 WARNING：

```
<op> on <arch>/<dtype>: backend <failed> failed, using <next> (<n> bound ops): <error>
```

固定后端（`resolve(..., backend=...)`）则会重新抛出异常。

<a id="failure-records"></a>
### 失败记录

- **键。** 记录的键为 `op|backend|arch|dtype|static options`（`status.failure_key`），当错误指出具体问题时，还会加上问题（`<tuned kernel>:<key>`，其中键包含分桶后的形状）：`ConfigsFailed`（某个问题的短名单中所有配置都失败）只记录该问题。
- **磁盘上。** 只有构建时遇到的构建 / 工具链失败（`is_toolchain_error`）会写入 `<kernel cache>/failures/backends.json`：发生在 `status.build_phase()` 内，`registry.exercise`（预热重放 `tuning/replay.py`，以及注意力预热 `tuning/prewarm.py`）会进入该上下文。实时调用（训练步、采样）中遇到的失败会在当前进程中重新绑定算子，但绝不会写入磁盘。
- **范围。** 只有相同 dtype 和静态选项下不带具体问题的记录，才会让 `resolve` 为该算子跳过某个后端，因此一种形状的失败不影响其他所有形状使用该后端，fp16 的失败也不影响 bf16。
- **可见性。** 因磁盘记录而跳过的后端，在算子解析时会记录一条 WARNING（`backend <name> skipped, failed here (...)`），因此启动运行时能看到每项持久化降级。
- **进程内。** 调优器在下一次查找某个问题时，会再次抛出已存储的 `ConfigsFailed`，不再重新计时。
- **环境。** 失败记录与环境指纹（包版本、工具链、Intel 源码模式）一同存储，后续具有相同指纹的进程会遵循它；环境改变则忽略。`kohakumoss-doctor` 列出探测结果和当前指纹对应的失败；`kohakumoss-doctor --clear-failures` 会清除它们。

<a id="backward-failures"></a>
### 反向计算失败

回退涵盖前向运行时抛出的失败，因此后端会在该阶段暴露其反向计算失败：

- 注意力内核在某个问题键的首次启用梯度的前向中，构建并启动一次反向计算（`dO` 为零，不记录梯度），与前向使用相同的 `BackendUnavailable` 转换，因此无法构建或启动的反向会在该后端保存任何激活之前将算子交给下一个后端；`TUNE=off` 时，启动的是未经计时的选择（已存储的优胜者，否则为第一个）；
- `registry.exercise(op, run)` 运行前向 + 反向（预热重放会对每个计划桶中的每个已记录调用执行此操作，注意力预热则覆盖每个长度和片段数桶），并在其中任何位置抛出失败时让算子回退，包括反向计算。

仍然到达反向阶段的反向失败（首次在梯度检查点重计算中遇到的新键，此时 `_Guarded` 不运行）会在那里抛出。

**受保护的调用。** 在当前后端的一次反向计算成功之前，记录普通 autograd 图的 `Bound` 调用（开启梯度、未追踪、没有 saved-tensor hooks、有需要梯度的张量）会经过 `registry._Guarded`：

- 实现在分离的叶张量上运行，并通过 `autograd.grad` 执行反向，因此仅在反向抛出的失败也会触发回退，并在下一个后端重新计算梯度；
- 重计算从 `save_for_backward` 读取调用的张量输入，因此如果输入在前向之后被原位修改，就会抛出 autograd 的版本错误，而不是使用新值重算；
- 使用梯度检查点和卸载的调用（saved-tensor hooks）保留自身的计算图语义，不受此保护；
- 反向会释放前向的叶张量、输出和参数（它们作为 `ctx` 属性保留，autograd 不会释放）：
  比其反向存活更久的计算图（驱动器把该步的 loss 带入下一步）不会保留其中任何一个。
  若不释放，未启用梯度检查点的第 1 步会把其整套激活保留到第 2 步的前向；
- 一次成功的反向会把该算子标记为已验证，步边界处（`finish_step`、`MemorySession.step`）
  的 `registry.settle_guards()` 会绑定其普通路径，因此一步（及其每个微批次）对每个算子只
  运行一条调用路径。在一步之外（`registry.exercise`、预热），一次成功的反向会立即解除该
  算子的保护。

驱动器还会以分离形式返回每步的 `StepOut`（`StepOut.detached()`），因此无论还有什么保留着
`ctx` 属性，任何一步的计算图都不会在其反向之后继续存活。

**调用路径。** `Bound` 按状态绑定一次调用路径，而不是每次调用都测试：

| 状态 | 路径 |
| --- | --- |
| 位于 `recording()` 内部 | 记录路径 |
| `guard` 开启 | 受保护路径（逐调用的 `_records_grad` 测试） |
| 其他情况 | 在回退的 `try` 内直接调用实现 |

设置 `guard`（一次成功反向后的 `settle_guards`、发生回退）以及进入或退出记录（`recording.on_change`）时，都会重新绑定每个存活的 `Bound`。

<a id="what-counts-as-a-failure"></a>
### 什么算作失败

“无法在此运行”指 `errors.BUILD_ERRORS`（RuntimeError、OSError、UnicodeError、Triton 的 `TritonError`），绝不包括内存不足：OOM 或程序错误会传播，不会禁用任何内容。

**后端探测。** 在任何算子将后端绑定到设备之前，`resolve` 会按进程和设备各运行一次该后端的探测（`status.register_check`）：

| 后端 | 探测 |
| --- | --- |
| Triton | 构建并启动一个最小内核（这也会在 XPU 上初始化 Intel 的驱动实用工具） |
| compile | GPU 上继承 Triton 的结果；CPU 上需要主机 C++ 编译器 |
| TileLang | 需要用于 nvcc 的主机编译器 |

探测失败时记录一条 WARNING，并在该设备上对所有算子跳过该后端，因此缺少编译器只会付出一次构建成本，而不是每个算子一次。

**文本编码器算子。** 文本编码器算子（`encoders/text/forward.py`、`TextOps`）在构建时探测中遵循相同规则：

- RuntimeError（包括内存不足）会绑定 eager 参考实现并记录一条 WARNING（`text encoder op <name> on <device>: backend <b> failed its probe, using eager: <error>`），保存在 `TextOps.fallbacks` 中，绝不写入磁盘；任何其他错误都会传播；
- 探测中内存不足时，会在设备空闲块归还驱动后重试一次（`forward.probe`），因为 eager 参考实现通常比内核需要更多工作内存，否则瞬时 OOM 会使整次运行都绑定到它（`tests/kernels/textenc/test_textenc_probe.py`，反例：不重试）。

<a id="lazy-backends"></a>
## 惰性后端

`import kohakumoss_kernels` 不执行任何后端工作，因此从不构建内核的数据辅助工具或 CLI 不承担这些成本。首次构建内核（`registry.resolve` 或 `available`，即模型的第一次 `OpSet.resolve`）时，每个进程执行一次 `registry.load_backends()`：

1. 应用进程的编译设置（`compile_setup.apply()`：编译缓存键标签，[编译缓存键](#compile-cache-key)，
   以及编译工作进程池，[Compile 后端](#compile-backend)）；
2. 导入 eager、compile、Triton 和 TileLang 后端包；Triton 和 TileLang 包在导入时设置主机
   工具链（`ensure_toolchain()`：Windows 上的 `vcvars64` 和 `PATH` / `CXX`、Intel Triton
   UTF-8 源码），compile 后端则在首次 CPU 探测时设置；
3. 运行一次循环 gc。compile 后端会导入 `torch._dynamo`，而该导入会经由导入栈留下一个引用环
   （`torch.fx.wrap` 保留自己的栈帧，其 `f_back` 链可达每个调用者栈帧）：若不运行这一遍，
   构建首批内核的那次调用（`load_backbone`，其局部变量中有模型）的栈帧会一直存活到回收器
   运行。见 [reference-cycles.md](reference-cycles.md)。

`CompiledOp` 也会在首次编译前应用这些设置。
`tests/kernels/registry/test_lazy_backends.py` 在新进程中检查这一点（反例：导入时加载后端；
省略 gc 这一遍，在关闭自动 gc 时会让首次构建的调用者所持有的对象继续存活）。

<a id="current-device"></a>
## 当前设备

内核在当前设备的队列上启动，因此每次 Triton 启动（`Launcher`）和 TileLang 启动（`_blockops.compiled`）都会在其张量设备对应的 `hardware.current.on_device(...)` 下运行；否则，当设备 0 为当前设备时，针对 `cuda:1` / `xpu:1` 上张量的启动会读取错误设备（Intel Arc 上为 DEVICE_LOST）。

`hardware.current` 还包含：

| 名称 | 职责 |
| --- | --- |
| `synchronize(device)` | 包中唯一的设备等待辅助函数 |
| `ACCEL`, `is_accel`, `device_module` | 加速器设备类型 |
| `saved_tensor_hooks_active()` | 唯一用于探测 torch saved-tensor hook 栈的函数（启用梯度检查点的前向或其重计算、激活卸载） |
| `rng_kept(devices)` | 在 CPU 和指定加速器设备上执行 `torch.random.fork_rng` |

`rng_kept` 包装调优中的每次随机抽取：`select` 的短名单计时、预热重放的实体化输入、注意力和适配器预热、`retune_pending` 重建的输入、设备探测、记录的孪生执行和文本编码器算子探测。执行后调用方的生成器仍处于原先位置，因此运行的随机流（适配器初始化、噪声、时间步）永远不会取决于内核缓存中的内容；`tests/kernels/tuning/test_rng_kept.py` 检查这一点（反例：不使用 fork 进行抽取）。

<a id="compile-backend"></a>
## Compile 后端

`backends/compile` 为每个块算子的 eager 参考实现和融合 linear 算子的 eager 组合注册 torch.compile（`compile/ops.py`、`COMPILED`）。内核后端按形状与它比较后选择（`compile/choice.py`）；它在 GPU 上的探测沿用 Triton 的探测（inductor 的 GPU 代码是 Triton），CPU 上则需要主机 C++ 编译器。

**调用模式。** `CompiledOp` 为每种调用模式保存一个编译后可调用对象，各自具有独立代码对象和 Dynamo 缓存；调用模式就是 Dynamo 除形状外所守卫的内容。

- 模式包括设备、梯度模式、算子内线程数、参数设备上的 autocast，以及每个参数张量的 dtype / requires_grad / 秩、None 和 Python 标量。
- Dynamo 按设备重新编译，因此跨多个 GPU 的线程式运行会为每个设备获得一个可调用对象，而不是将它们的计算图堆到一种模式上。
- 线程数是其中一部分，因为梯度检查点重计算在 autograd 线程上运行，其算子内线程数不同；这样重计算就有独立的编译后可调用对象，而不是重新编译前向的对象。

**限制和失败。**

- 一种模式需要超过 `GRAPH_LIMIT` 个图时会抛出 `GraphLimitError`；Dynamo 自身的重新编译限制被提升到更高，因此永远不会触及 Dynamo 静默的 eager 回退。
- 首次调用构建失败会抛出 `BackendUnavailable`。
- 在外层 torch.compile 内部会追踪 eager 参考实现。

**绑定。** `CompiledOp.resolved(args)` 是 `args` 对应模式的运行器。内核与编译路径比较算子按调用键（设备、梯度模式、线程数和 autocast dtype，以及每个参数的形状 / dtype / 梯度标志或值，这些内容包含该模式）绑定它，因此绑定后的调用只计算一个键并调用运行器，无需再计算模式。适配器 compile 候选（`adapters/bind/autotuned.py`、`CompiledBody`）按桶将自动调优算子的操作数形状所对应的 `CompiledOp` 绑定一次（`CompiledBody.at`）。

**编译工作进程池。** 在 Linux 上，inductor 会在边车进程中用一组 fork 出的工作进程
（`min(32, cpu count)`）编译 Triton 内核；在 Windows 上则在进程内编译。inductor 的
默认行为是在最后一次编译结果出现 60 s 后让该进程池静默，并在下一次编译时重新 fork 一个，
而此时的边车进程中上一个进程池的线程仍在退出；此时 fork 出的工作进程可能在继承的锁上
死锁，它接下的任务永远不会返回（`future.result()` 没有超时：进程会永久等待）。
`compile_setup.apply()` 会设置 `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0`
（已设置的环境变量优先），并在 inductor 的配置已加载时把该配置从此变量读取的值写入其中
（普通配置赋值只对一个线程有效），因此进程池创建时不带其空闲 Timer，并存活于整个进程。
`src/` 中每个 torch.compile 位置都会先调用它（`tests/rules/test_compile_sites.py`），
运行命令行和 sample.py 会记录它（`compile pool: no quiesce`），
`tests/kernels/registry/test_compile_pool.py` 会在新进程中检查该进程池
（反例：inductor 的默认进程池带有 Timer）。开销：工作进程在进程存续期间一直运行；
在一次集成训练运行中，32 个工作进程占用 0.85 GiB 比例（PSS）主机内存，
计入与边车共享的页则为 12.6 GB RSS，边车占用 0.46 GiB（无论如何都会占用）。
不占设备内存。

<a id="fused-linear-ops"></a>
## 融合 linear 算子

`linear_situ`、`linear_gated_residual` 和 `linear_qk_rope`（GEMM + MOSS 尾处理）在两个内核后端共用一套主机代码，即 `backends/_fused_linear.py`：`FusedLinear` 定义后端的六个自定义算子（`kohakumoss::<backend>_linear_{situ,gres,qk}_{fwd, bwd}`）、它们的 autograd（仅在 autograd 记录期间保存反向所需内容；反向先执行尾处理的反向内核，再通过 cuBLAS GEMM 求输入和权重梯度）、针对每个问题在 `fused`（后端的 GEMM + 尾处理内核，每种图块配置一个入口）与 `split`（cuBLAS + 独立尾处理内核）之间的选择、`force_path`，以及注册表绑定的公共函数。后端模块只包含其融合启动器、图块配置和一个记录差异的 `FusedBackend` 记录：

| 字段 | Triton | TileLang |
| --- | --- | --- |
| `define` | `triton_op`（由 Inductor 追踪） | `custom_op` + 共享元函数 |
| `pick` | 对每个配置使用 `tuning.select` | `_blockops.tuned`（跳过编译失败的配置） |
| `fusable` | 每个问题 | 无偏置；`linear_gated_residual` 还要求残差流使用激活 dtype（否则仅支持 `split`） |
| `dead` / `mark_dead` | 无 | 丢弃抛出 `CompileError` 的融合配置，然后该问题重新选择 |
| `fallback` | 无 | 公共函数抛出 `CompileError` 时运行编译后的 eager 组合 |

两个后端的问题键均为 `<token bucket>-<K>-<N>-<dtype>-<flags>`（`_fused_linear._key`；标志：算子具备的 bias / gate / residual scale / residual dtype / save）。qk 入口将 q / k 归一化权重作为 fp32 传入（在算子边界转换一次；权重梯度以参数 dtype 返回），两个后端的 qk 反向都将归一化权重梯度转换为保存权重的 dtype。`tests/kernels/blocks/test_fused_linear_host.py` 检查两个后端运行相同的公共函数和算子主体（反例：某后端保留自己的副本），以及 fp32 归一化权重（反例：直接传入存储 dtype）；`test_fused_linear.py` 和 `test_saved.py` 在 GPU 上检查数值和保存的张量。

<a id="selective-checkpoint"></a>
## 选择性梯度检查点

`CHECKPOINT="selective"` 保留矩阵乘法和注意力算子的输出，重新计算其余部分（`kohakumoss/memory/runners.py`）。内核后端将注意力和融合 GEMM + 尾处理的前向作为自定义算子启动，梯度检查点的分派模式看到的是它们自己的名称，而不是 aten 矩阵乘法；每个后端在导入时将这些前向添加到 `kohakumoss_kernels.saved_ops.KEPT`（`attn_varlen_triton`、`attn_varlen_tilelang`、Triton 和 TileLang 的 `*_linear_situ_fwd` / `*_linear_gres_fwd` / `*_linear_qk_fwd` 算子，以及随 varlen 入口加入的 torch `torch_attn::_varlen_attn` / `aten::_flash_attention_forward`），梯度检查点策略像保留 aten `mm` / `linear` / SDPA 一样保留 `KEPT` 中的每个算子。`tests/kernels/autograd/test_saved_ops.py` 检查这些算子上的策略（反例：空的 `KEPT`）。

<a id="compile-cache-key"></a>
## 编译缓存键

Torch 的磁盘编译缓存（`TORCHINDUCTOR_CACHE_DIR` 下的 FX 图缓存和 AOT-autograd 缓存）以追踪的前向图作为条目键。自定义算子的 `register_autograd` 反向及其 schema 不属于该键的一部分，因此由不同算子代码构建的缓存条目（例如少一个参数的反向）会重放那份代码的反向，并用它的参数列表调用算子（"expected at most 10 argument(s) but received 11"）。

`kohakumoss_kernels.cache_tag.apply()` 在首次构建内核时（`load_backends`）以及 `CompiledOp` 首次编译前，将 `kohakumoss-<sha1[:12]>` 追加到 `torch.compiler.config.cache_key_tag`：

- 哈希涵盖 `kohakumoss` 包版本，以及 `kohakumoss_kernels` 下所有注册自定义算子或其 autograd 的模块文本（包含 `custom_op(`、`register_autograd(` 或 `torch.library` 的源码）。
- 对这类模块的任何编辑或版本变化，都会让每个编译缓存条目获得新键，绝不会读取其他键下的条目。
- 用户已设置的标签会保留，我们的标签追加在 `|` 之后。
- 计算标签时，每个进程读取这些源码一次，耗时几毫秒。

<a id="peak-windows"></a>
## 峰值窗口

分配器对每个设备只有一个峰值计数器，重置它会丢失其他所有进行中测量的峰值。`hardware.memory` 改为按窗口测量：`open_window` / `close_window`（或 `with window(device) as w`）给出窗口打开期间分配 / 保留字节数的峰值。每次打开、关闭和读取都会先将计数器并入该设备的所有打开窗口，再重置计数器，因此嵌套或重叠窗口绝不会丢失外层窗口的峰值。`memory.fold` 是项目中唯一调用 `reset_peak_memory_stats` 的地方。`memory.MODULES` 按设备类型覆盖设备模块（测试使用模拟分配器）。

<a id="optimizer"></a>
## 优化器

`kohakumoss_kernels.optim` 一次性绑定 `adamw_step`、`adamw8bit_step` 和 `adamw_step_multi`（`resolve`）；`optim.reference` 定义每个算子的计算内容。训练器的 `adamw8bit_kernel` 优化器（`kohakumoss.training.optim.optimizers.adamw8bit`）通过 `adamw8bit_step` 执行步更新；`adamw_step` 和 `adamw_step_multi` 用于内核基准测试和测试。

- **状态。** AdamW 保留 `m` 和 `r = sqrt(v)`，因此平方幅值永远不会进入 16 位存储。8 位变体将 m 和 r 按每块 `BLOCK` 个值存储为 int8 / uint8 编码，编码为对值 / 块绝对最大值进行平方根压扩的结果。
- **写回模式。** `nearest`、`stochastic`（bf16：向 fp32 位模式加上一个随机 16 位整数，再截断）和 `kahan`（参数 dtype 的补偿缓冲区保存舍入丢失的部分）。
- **种子。** 多张量步为张量 t 使用种子 `seed + t * SEED_STRIDE`。
- **选择。** 每个后端内核按问题键与参考实现比较后选择（`optim/choice.py`）；仅在某个键实际计时时才创建参数的临时副本。
- **梯度 dtype。** 两个后端都允许梯度使用自己的 dtype（16 位参数搭配 fp32 梯度）。

<a id="test-budgets"></a>
## 测试预算

首次使用时构建内核的内核测试，每个会话都在空内核缓存上运行，因此首次编译在每次运行中都计入耗时；这类测试使用高于默认值的 `@pytest.mark.budget(N)`（最慢实测通道加约 50%），并保留完整候选覆盖：

| 测试 | 预算 s |
| --- | --- |
| `attention/test_attention.py::test_error_stats`、`::test_empty_and_single_segments` | 60 |
| `attention/test_attention_infra.py::test_backward_failure_falls_back_in_forward` | 90 |
| `attention/test_attention_infra.py::test_forward_failure_falls_back`、`::test_tilelang_backward_reads_no_row_stats_past_the_last_token` | 60 |
| `attention/test_attention_choice.py::test_chosen_implementation_has_no_error_regression` | 30 |
| `textenc/test_textenc_ops.py::test_delta_candidates_precision` | 60 |
| `autograd/test_compile_blockops.py::test_fullgraph` | 30 |
| `blocks/test_norm.py::test_rms_norm_mod` | 30 |
| `quant/test_fp8.py::test_path_forward_and_dgrad` | 25 |
| `tuning/test_kernel_or_compile.py::test_prewarm_binds_every_op` | 20 |
| `tests/adapters/test_kernel_ops.py::test_every_candidate_matches_the_eager_body`（sm75 上的 BOFT 为 48 s） | 75 |
| `tests/training/prewarm/test_adapter_prewarm.py::test_prewarm_leaves_factors_untouched` | 35 |
| `tests/training/prewarm/test_adapter_prewarm.py::test_prewarm_keeps_one_problem_alive` | 20 |

不带 `tests/` 的路径位于 `tests/kernels/` 下。

<a id="benches"></a>
## 基准测试

`kohakumoss_kernels.bench.blockops` 是 `scripts/bench/kernel/` 的块算子测试框架。对每个用例：

- **计时：** 后端内核、eager 组合、对其进行的 torch.compile（静态形状）和注册的编译路径（动态形状），分别测量前向和前向 + 反向（刷新 L2 并预热时钟后取 N 次中的最佳值，另加 CUDA 图重放）。
- **比较：** 将每个输出和输入梯度通过误差统计与 eager 参考实现的 fp64 运行比较，并提供相对于 eager 组合的逐统计项较差列表。
- 按形状选择的入口会报告其选择（`selected`）。
- 各行写入 `runs/kernel/<op>-<card>.json`。

<a id="varlen-attention"></a>
## 变长注意力

`attention_varlen(q, k, v, cu_seqlens, max_seqlen, scale)` 作用于打包的 `[N, H, D]` 张量（token 和 head 步长任意，Triton 要求 head dim 连续）。

- 网格第一轴覆盖最长片段；超过所属片段末尾的程序会退出，因此每个批次不需要主机工作。
- Triton 内核以 int64 计算 token 偏移（对于具有 qkv 步长的视图，约 233k token 之后，片段起点乘行步长就会超过 2^31 个元素）。

数值规则（三条，两个后端均适用）：

1. **双平面行统计。** 前向将原始分数最大值 `m` 和 `log2(l)` 分别存为两个 fp32 平面 `[H, N]`，而不是一个 log-sum-exp。单个 `c*m + log2(l)` 在 `c*m` 很大时会丢失 `log2(l)`。
2. **先平移后缩放。** 概率为 `exp2((s - m) * c - log2 l)`；`s - m` 在行最大值附近是精确的，因此主导概率恰好为 1，`delta = rowsum(dO * O)` 得以抵消。
3. **在累加器上缩放。** dS 在未缩放时舍入为 16 位；softmax 缩放最后乘到 fp32 dQ / dK 累加器上。

设计空间（调优器按架构和问题键选择；每种变体都是单独编译的内核）：

| 内核 | 配置项 |
| --- | --- |
| Triton 前向 | BM、BN、warps、stages |
| Triton 反向 | 策略 `fused`（按 key 块计算 dK/dV，dQ 通过 fp32 原子操作写入一个 `[N, H, D]` 累加器，delta 内核计算 `delta` 时将其清零，最后执行一次缩放并转换）或 `split`（dK/dV 内核 + dQ 内核，无原子操作）；每个内核单独调优，每种策略计时均包含其 delta 启动 |
| TileLang 前向 | BM、BN、threads、`qreg`（query 图块放在寄存器中）、`share`（K 和 V 共用一个共享缓冲区） |
| TileLang dK/dV | BM、BN、threads、`kvreg`（K / V 图块放在寄存器中）、`pds`（P 和 dS 共用一个共享缓冲区） |
| TileLang dQ | BM、BN、threads、`qreg`（Q / dO 放在寄存器中）、`share`（V 然后 K 共用一个缓冲区） |

寄存器 / 共享缓冲区变体以寄存器换取共享内存，让每个 SM 共享内存较小的显卡（Turing：64 KiB）能够容纳更多程序。

fp16 / bf16 MMA 在所有位置都使用 fp32 累加。

<a id="measured-design-per-arch"></a>
### 各架构的实测设计

每个内核的候选首先使用该架构自身实测的种子（`backends/triton/attention/varlen.py` 中的 `ARCH_SEEDS[sm]`、`backends/tilelang/attention/varlen.py` 中的 `SEEDS[(kind, sm)]`），然后是共享种子，最后是规划器排名；调优器对前 `SHORTLIST` 项计时。`TUNE=off` 时绑定已存储的优胜者，否则不经计时运行第一个合法种子，追踪时也如此。

| 架构 | 依据 | 前向（BM、BN、warps、stages） | 融合反向 | dK/dV | dQ |
| --- | --- | --- | --- | --- | --- |
| sm120（RTX 5090，每块 99 KiB 共享内存） | 在 2 MP x1、1 MP x2、512 px x8 下对每个内核完整扫描 512 种图块 | 128, 64, 8, 3（235 TFLOP/s） | 16, 128, 8, 4（200） | 32, 32, 4, 3（209） | 128, 32, 8, 3（241） |
| sm80（A100，163 KiB） | FA2 / Triton 的 A100 图块（未在此测量） | 128, 64, 4, 3 | 64, 128, 8, 2 | 64, 128, 8, 2 | 128, 64, 8, 3 |
| sm86 / sm89（RTX 3090 / 4090，99 KiB） | 在 3090 上进行相同扫描（sm89 共用该表） | 128, 32, 4, 2（76） | 16, 128, 8, 2（63） | 16, 128, 8, 2（65） | 128, 64, 8, 2（78） |
| XPU | 共享种子，按键调优 | | | | |
| sm70 / sm75（TileLang） | 实测的各架构种子 | 参见 `SEEDS` | | | |

在 sm120 上：

- 融合反向（dQ 通过 fp32 原子操作计算）在每个实测形状下都快于拆分反向（2 MP x1 下为 11.6 对 14.7 ms），达到 FA2 反向性能的 86 %。
- FA2 和我们的前向都达到该卡的实测 MMA 速率（探测为 233 TFLOP/s），因此单个长片段与 SDPA 持平（[性能](../performance/attention.md#the-single-segment-ceiling)）。
- 融合内核保留 BN = 128 个 key，dK 和 dV 放在 fp32 寄存器中；大于 16 行的 query 图块会因寄存器压力而变慢。
- TileLang 也能在 sm120 上构建；在那里只会通过注册表回退使用它。

sm80 种子是面向 A100 级显卡的一个设计点；首次调优某个问题时，调优器会用实测选择替换它们。

**各后端的注册表入口**（sm80+ / XPU 上的 `backends/triton/attention/entry.py`、`backends/tilelang/attention/entry.py`）：一个 `VarlenChoice`（`backends/varlen_choice.py`），按问题键对后端自身的内核与 torch 的 `varlen_attn`（torch 构建版本和显卡支持 flash 时）或逐片段 SDPA 循环计时比较，并绑定优胜者（表 `attn_varlen_impl_<backend>`）。

- 键包含反向计算是否经过该调用：`-g1`（梯度模式开启、有输入需要梯度）在叶张量副本上测量前向 + 反向，`-g0`（推理、采样）仅在 no_grad 下测量前向，因此同一卡上的训练运行和推理运行各自绑定自己的优胜者。
- 仅在推理模式、启用梯度检查点的前向、重计算和追踪之外计时；torch.compile 内部绝不使用 SDPA 循环。
- 训练器在首步之前的预热中调用 `tuning.prewarm_attention`，因此每个长度桶和片段数桶都会在启用梯度检查点的训练前向之前决定好（[tuning.md](tuning.md#prewarm)）。

两个后端的图块和策略键为 `<dtype>-h<heads>-d<head dim>-L<longest segment bucket>-S<segment count bucket>`（`tuning.buckets.segment_bucket`：1、2、4，然后将 8 及以上合为一类）：网格是（最长片段的块数）x heads x segments，因此由许多短片段和一个长片段组成的打包与最长片段相同的双片段打包，对显卡的占用方式不同。

**调用路径。** 问题键首次调用时读取表，或在允许计时的地方调优（[调优窗口](tuning.md#tuning-windows)），并将选中的内核绑定到一个启动可调用对象；后续调用按键查找绑定并启动。追踪期间做出的临时选择不会绑定。

**防护。**

- 在求最大值前，被掩码的 key 列设为 `-inf`（完全有效的行始终有有限的最大值）。
- 被掩码的图块显式将 `p` 清零，避免越界行产生 `inf * 0`。
- 空片段不启动任何内容。
- 反向内核只对片段内的行读取行统计（`m`、`log2 l`、delta）；超过片段末尾的行读取 0（显式 `if_then_else`，不依赖 TileLang 自动边界防护）。

<a id="matmul-precision-and-error-statistics"></a>
### 矩阵乘法精度和误差统计

每个 MMA 都使用 16 位操作数和 fp32 累加，与 flash attention 一样：P 在 P·V 和 P^T·dO 之前舍入一次，dS 在 dS·K 和 dS^T·Q 之前舍入一次，delta 来自 16 位 O，而累加器、行统计和 softmax 都保持 fp32，直到存储时进行唯一一次舍入。

测试通过误差统计（`tests/_util/error_stats.py`）检查 o、dQ、dK、dV：

- 内核和 PyTorch 都与 fp64 比较（余弦；绝对和相对误差的均值 / 中位数 / 最小值 / 最大值）。
- 每次检查都将其表格追加到 `$KOHAKUMOSS_PRECISION_DIR/attention.jsonl`。
- 仅在出现退化时测试才失败：非有限值，或余弦差额或均值 / 中位数误差达到 PyTorch 的 `MAX_ERROR_RATIO` 倍及以上。
- “PyTorch”指按其原本方式计算该算子的普通 PyTorch：`pytorch_sdpa`（`tests/kernels/attention/_attention_case.py`），对每个片段以输入的 16 位 dtype 调用一次 `torch.nn.functional.scaled_dot_product_attention`，梯度由 autograd 计算，使用设备所选择的任一 SDPA 后端。

各架构实测统计见 [performance/attention.md](../performance/attention.md#precision)。

**巨大 logits**（聚簇的 q / k，logits ~1e5）。softmax 几乎为 one-hot，`sum_j dS_ij = 0` 仅在舍入之前成立。dQ = `scale * sum_j dS_ij K_j`，而 K 含有所有 key 共有的较大分量，因此将 dS 舍入为 16 位（bf16：8 位）会留下真实 dQ 本会抵消的残余 `(sum_j round(dS_ij)) * K`。每种 16 位操作数实现都有这一问题（包括 flash SDPA）；巨大 logits 的统计见 [performance/attention.md](../performance/attention.md#precision)。

测试汇集 16 次随机抽取（每次抽取的误差由少量元素承载），且仅对余弦和绝对统计应用退化检查：one-hot softmax 下，大多数 fp64 dQ / dK 元素约为 1e-190，远低于任何 16 位 dtype 的范围，因此其相对误差（flash SDPA 和我们的实现的均值、中位数都可达 1e18 - 1e91）测量的是参考值而非内核。相对统计仍写入表格。

内核的 `torch.compile`（fullgraph）结果与 eager 调用通过误差统计而非按位比较：Inductor 使用自己的编译选项重新构建启动的 Triton 内核，可能以不同方式收缩运算。

各后端和策略的按位可重复性（在相同输入和显卡上重复调用）：

| 路径 | o | dK, dV | dQ |
| --- | --- | --- | --- |
| Triton 前向 + `split` 反向 | 按位一致 | 按位一致 | 按位一致 |
| Triton `fused` 反向 | 按位一致 | 按位一致 | 非按位一致（fp32 原子加法按到达顺序执行） |
| TileLang 前向 + 反向（delta、dK/dV、dQ；无原子操作） | 按位一致 | 按位一致 | 按位一致 |
| torch `varlen_attn`（计时候选） | 与 torch 一致 | 与 torch 一致 | 与 torch 一致 |

Triton 策略由调优选择，因此只有 `split` 获胜时 dQ 才按位一致。不同的调优图块配置会产生不同（但同样精确）的位结果。

**Eager 边界。** eager 参考实现对每个片段运行一次 SDPA，并需要主机上的片段边界。它从以 `cu_seqlens` 张量为键的缓存中获取边界：

- `SeqInfo.to` 将计划移到设备时，会注册已持有的边界（`remember_bounds`），因此 eager 路径从不执行同步；未注册的张量会回读一次。
- 只有张量仍存活且版本计数器未变化时，才会使用该条目，因此原位重写或复用地址上的新张量会再次回读。
- 推理张量没有版本计数器，每次调用都会回读。
