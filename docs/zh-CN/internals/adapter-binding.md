[English](../../internals/adapter-binding.md)

<a id="adapter-binding"></a>
# 适配器绑定

每个适配层如何在绑定时根据其类型记录（[adapter-kinds.md](adapter-kinds.md)）一次性选择前向函数：包括绑定的前向函数、它们调用的内核算子、自动调优的内核或编译路径选择、融合的基座加法，以及量化或流式传输的基座。

<a id="bound-forwards"></a>
## 绑定的前向函数

`bind.select.bind_target` 根据每层的类型记录和 `BindCtx` 一次性解析其前向函数：模式、后端、该层的内核算子、乘数、DoRA-detach、模块 dropout、训练状态、内核算子是否适用、随机抽取。

- 每个前向函数都是在绑定时构建的闭包（`bind.forwards`、`bind.eager`、`bind.compose`）。
- 每种变体（是否有标量参数、是否重缩放、乘数是否为 1、是否有偏置、基座加法是否在算子内部）都有自己的闭包，在绑定时选择。
- 除非模块使用输入或秩 dropout 进行训练，否则内核算子均适用；只有 LyCORIS 模块方法支持这些 dropout（`eager.module_bypass`）。

<a id="kernel-ops"></a>
## 内核算子

`bind.ops`（激活侧算子：旁路、旋转、IA3 缩放）和 `bind.weight_ops`（权重算子和 DoRA 分解）将各类型的算子构建为作用于一层模块的 `autotuned.Op`：

- `operands(t)`：运行时输入 `t` 的操作数元组，每次调用构建一次；每个候选读取同一批对象，因此对它们求得的梯度能够传递到模块参数；
- `kernel(backend, ops)`：`lycoris.kernels.autograd` 中的 LyCORIS 融合算子，通过 `backend=` 指定；
- `body(ops, *consts)`：eager 公式，即后端为 `"torch"` 的公共 `lycoris.functional` 入口（`bind.bodies`；对于没有 functional 入口的 IA3 和 GLoRA，则使用其公式）；
- `in_scope(backend)`：该后端的融合内核是否覆盖这一层：静态缩放、LoHa 的秩不超过 `lycoris.kernels.limits.MAX_HADA_RANK`、LoKr 的后半部分为普通形式且每个 kron 因子不超过后端的 `lokr.bypass.MAX_KRON_FACTOR`、diag-OFT 块不超过后端的 `oft.MAX_BLOCK`（默认 `lycoris.functional.diag_oft.MAX_BLOCK`）。

旁路算子接收 (tokens, channels) 激活（低秩类型和输入轴 IA3 使用输入，旋转和输出轴 IA3 使用基座输出），并像模块一样将因子转换为其 dtype；权重算子接收 fp32 基座权重。

`bind.kernel_ops` 将 Op 转换为前向包装器所接收的可调用对象：

- `FixedBackendOps(backend)`（适配器配置指定了后端）：对于融合后端使用融合内核（超出其适用范围时为 None，此时该层走模块路径），对于 `compile` 使用编译后的主体，对于 `torch` 使用 eager 主体；
- `AutotunedOps(backend)`（配置的后端为 `auto`，且设备有融合后端）：每个算子对应一个 `AutotunedOp`；超出内核适用范围的算子运行编译后的主体。

<a id="autotuned-ops"></a>
## 自动调优算子

一个 `AutotunedOp` 有两个候选：

| 候选 | 运行内容 |
| --- | --- |
| `kernel` | 设备上优先级最高的融合后端的融合算子（按 `lycoris.kernels.dispatch` 顺序） |
| `compile` | 通过 `kohakumoss_kernels.backends.compile.ops.CompiledOp` 对 eager 主体进行 `torch.compile`（[梯度检查点重计算](#checkpoint-recompute)介绍其计算图如何缓存）；`AdapterConfig.compile_dynamic`：`tokens`（默认）将 token 轴标记为动态，也可选 `static`、`dynamic` |

- **桶的第一次调用**（token 桶，即从 `TOKEN_FLOOR` 起的 2 的幂、激活 dtype、梯度模式）会使用问题键请求 `kohakumoss_kernels.tuning.tuner.select`（键包括每个操作数的形状、分桶后的 token 轴、dtype 和梯度标志，以及梯度模式）。
- **该卡已存储的优胜者**会立即绑定。否则会在操作数的分离副本上对两个候选进行前向加反向计时，存储并绑定优胜者。抛出构建或适用范围错误的候选会落选。
- **调优器不能计时的地方**（推理模式、反向计算、追踪）会让该桶暂时运行内核而不定案，并在下次调用时再次询问。在调优窗口之外（[tuning.md](tuning.md#tuning-windows)），缺失的键会不经计时绑定第一个候选，并记录一行 WARNING。
- **没有融合内核能接收的操作数**（fp16 / bf16 混用、sm_80 以下的 bf16）会绑定编译后的主体。
- **在外层 `torch.compile` 内部**会追踪 eager 主体。

<a id="checkpoint-recompute"></a>
### 梯度检查点重计算

无论中间运行了什么，适配器调用在启用梯度检查点的前向计算中绑定什么，重计算时就绑定什么：其他层、其他 token 数、其他微批次（`gpipe` 下的线程式和进程组流水线会先运行所有微批次的前向，再进行任何重计算）、另一阶段的线程、autograd 线程。以下三点保证这一性质：

- **全局状态。** Dynamo 使用算子内线程数作为计算图的守卫条件，`CompiledOp` 也将其纳入调用模式的键。重计算在执行反向计算的线程上运行：Linux 上 autograd 设备线程以 1 个线程运行，训练器线程则使用主机线程数（64 核主机上为 32）。线程数不同时，重计算无法命中前向计算的计算图，会根据自身输入追踪一个新图，其分区可能保存不同的张量集合（`CheckpointError`；`gpipe` 下的线程式流水线在 LoRA 上会出现此错误，因为阶段 1 的计算图首次在训练中追踪）。因此，`kohakumoss.memory.runners.CheckpointRunner` 让每次重计算都使用其前向计算的线程数（`forward_state`，一个梯度检查点 `context_fn`；其中的选择性上下文），结束后恢复反向线程的线程数。梯度模式和 autocast 由 `torch.utils.checkpoint` 自身恢复。

- **候选。** 已定案的桶保持绑定。若某个桶首次在 saved-tensor hooks 下调用（启用梯度检查点的前向计算，`autotuned._recomputed`），而调优器不能计时，就会在整个进程内绑定内核：同一签名的所有层共享这一问题，因此另一层若在梯度检查点之外运行，可能在前向和重计算之间将它定案为编译后的主体。这样的桶在本进程之后不会再计时；`AdapterSet.prewarm` 会在第 0 步之前将计划中的桶定案。
- **编译后的计算图。** `autotuned._graph` 按主体及除 token 轴之外的操作数形状（`autotuned.fixed_shapes`）各持有一个 `CompiledOp`，每个实例为每种调用模式保留一个 Dynamo 缓存。token 轴标记为动态后，每个这样的缓存在每个设备上保留一个图，无论之前运行了什么，都根据同一输入规格追踪。若不同层形状共享一个缓存，则会为每种形状收集一个静态图，随后在新形状到来时收集自动动态图；Dynamo 会先尝试最新的图，因此在其他形状之后到来的重计算可能运行一个保存张量与其前向计算不同的图（`CheckpointError`：保存张量的数量或元数据不同）。自动调优算子会将桶的 compile 候选一次性绑定到该 `CompiledOp`（`CompiledBody.at`；token 轴索引按操作数数量计算一次），并按算子内线程数和 autocast 状态各解析一次该 `CompiledOp` 的运行器（桶固定了调用模式的其余部分），因此绑定后的调用只需将 token 轴标记为动态并调用运行器，无需查找形状或模式（`tests/adapters/test_autotuned_pattern.py`，反例：每次调用都处理模式）。一种模式的计算图超过 `GRAPH_LIMIT` 时会抛出异常，而不是运行 eager。

<a id="body-candidate"></a>
### 主体候选

精度是通过 / 不通过的门槛。速度选择只会考虑通过门槛的候选。

- 通过自定义 `torch.autograd.Function` 运行的主体（LoHa：`bodies.loha`、`loha_with_base`、`loha_weight`、`loha_merged` 背后的 LyCORIS `HadaWeight`，集合为 `bodies.AUTOGRAD_FUNCTION_BODIES`）在 eager 中于 autocast 之外执行该 Function 的反向计算，因此其反向计算是 fp32。
- torch.compile 在 autocast 下以 16 位矩阵乘法追踪同一反向计算。编译候选的梯度精度随后低于 PyTorch（fp16 loha.weight 梯度：编译的 abs_mean 为 8.1e-4，而 eager 为 2.0e-7），因此未达到门槛。
- `autotuned.body_candidate` 在绑定时解决这一问题：这类算子的主体候选是 `eager`（eager 主体），而不是 `compile`。基于它的 `AutotunedOp` 对内核与 eager 进行计时比较。超出内核适用范围的层，以及固定的 `compile` 后端，均运行 eager 主体。

<a id="prewarm"></a>
### 预热

`AdapterSet.prewarm(rows, dtype, devices)` 在第 0 步之前，使用临时输入，针对计划中的 token 桶和梯度模式，将每个不同的自动调优算子定案（`AutotunedOp.signature`：名称以及没有 token 轴的操作数布局）；已经存储的问题只进行绑定，因此重启时无需计时。

<a id="forward-wrappers"></a>
## 前向包装器

启用 `AdapterConfig.fuse_base_add`（默认开启）时，绑定过程会选择将基座纳入 LyCORIS 内核的算子（`lycoris.kernels.autograd` 的 `base=` / `delta=` / `diff=False`），使加法在内核存储时完成，梯度可传递到基座；compile 候选则在编译后的计算图内部执行加法：

- 旁路（LoRA、LoHa、LoKr）：`forwards.with_base(linear, op)` 计算 `op((x, base(x)))`；
- 乘数为 1 的合并：`forwards.merged_with_base` 计算 `lin(x, merge(W))`，使用 fp32 基座权重作为基座（记录中的 `kernel_merged_weight`）；
- DoRA：`forwards.dora_with_delta` 分解 `(W, dW)`，在分解算子内读取 W + dW（`apply_dora(..., delta=)`）；
- OFT / BOFT 旁路：`forwards.rotated(linear, rotate)` 直接计算旋转后的基座输出（`diff=False`），乘数为 1 的 OFT 合并则计算旋转后的权重。BOFT 合并保留单独的加法：`boft_diff_weight` 没有 `diff=False`。

乘数不为 1 和可训练标量会保留单独的加法（缩放仅作用于 dW）。

<a id="quantized-and-streamed-bases"></a>
## 量化和流式传输的基座

非普通 16/32 位稠密参数的基座（`Target.stored`：int8 / fp8 仅权重存储、fp8 计算副本、环形 / swap 流式传输槽位）会保留其存储留下的空 `weight` 占位符。LyCORIS 通过 `in_features` / `out_features` 确定每个模块的大小，并在遇到非原生 linear 时以旁路运行；LyCORIS 在构建时需要从基座权重读取的状态（DoRA 幅值、T-LoRA SVD、OFT / BOFT 重缩放）由 `network.Deferred` 使用每层基座的一份临时副本构建。该层持有其存储的基座和适配器因子：

- 训练只绑定旁路（这类基座上使用 `ADAPTER_MODE='merge'` 会产生一个指出基座的验证错误）；旁路内核算子接收 16 位激活并加到基座自身的输出上（fp8 GEMM、量化内核或流式传输槽位）；
- 乘数不为 1 时的 OFT、模块路径上的 BOFT、DyLoRA 和 GLoRA 运行 LyCORIS 的旁路前向函数（`eager.module_rotation`，通过模块的 `org_forward` 执行一次基座前向；`eager.dylora_train`；`eager.glora_bypass`，扣回 LyCORIS 的 GLoRA 增量重复添加的基座偏置）；
- DoRA（`bind.dora.bypass_out` / `bypass_out_biased` / `bypass_in`）读取 W 的一份临时副本来计算 W + dW 的范数，然后根据基座前向和旁路增量计算 `y + m ((s - 1) W x + s dW x)`（输出轴）或 `y + m (W' (c x) - y)`（输入轴）；
- 推理（`adapters.infer.stored`）无论模式如何，都会将每种类型以旁路链附加到这类基座上（请求合并时会回退并记录一行日志）；在 fp16 折叠下，旋转或通道类型通过 `org_forward` 读取未折叠的基座输出，DoRA 比率在附加时计算一次，稠密差分文件则会被拒绝。

<a id="tests"></a>
## 测试

| 测试 | 检查内容 |
| --- | --- |
| `tests/adapters/test_kernel_ops.py` | 单个 linear 的 LyCORIS 模块上的每个内核算子：绑定可运行的每个候选（适用时的融合内核、主体候选、eager 主体）与 eager 主体进行比较，分别在 autocast 到测试 dtype（与训练运行方式一致）和 fp64 下检查前向与反向，覆盖 fp16 和 bf16。候选的输出 dtype 为 eager 的 dtype 或操作数 dtype 的提升结果（内核以操作数的自然 dtype 存储） |
| `tests/adapters/test_body_candidate.py` | LoHa 算子绑定内核与 eager 进行比较，没有 compile 候选 |
| `tests/adapters/test_stored_bypass.py` | 每种类型（LoRA、LoHa、全矩阵因子为 16 的 LoKr、OFT、BOFT、IA3、GLoRA、DoRA、T-LoRA）在 int8 / fp8 仅权重基座上的一个 linear：通过误差统计，将绑定前向和推理附加（旁路和合并模式）与独立 fp64 公式比较，以稠密基座上的相同适配器为基线；旁路前向中不读取已存储的权重（DoRA：恰好读取一次）；合并错误；稠密差分拒绝 |
| `tests/training/prewarm/test_adapter_prewarm*.py` | 预热的桶、因子上不遗留梯度，以及重启时不进行计时 |
| `tests/integration/test_adapters_real.py` | 真实骨干网络上，基座分别为 16 位、int8 和 fp8 计算的 LoRA 与 LoKr：一个训练步、适配层与 fp64 比较、重新导入 LyCORIS 导出结果 |
