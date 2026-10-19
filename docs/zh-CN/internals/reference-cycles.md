[English](../../internals/reference-cycles.md)

<a id="reference-cycles"></a>
# 引用环

设备内存通过引用计数释放：模型、优化器或训练器状态在其最后一个名字消失时释放自己的张量。处于
引用环中的对象只能由 Python 的循环回收器释放，而它按对象分配计数触发，不按设备内存触发：
在它运行之前，从该环可达的每个张量都保持已分配。处于引用环中的骨干在 `del` 之后仍持有其整套
权重；此时在同一进程中加载第二个模型就会耗尽设备内存。因此源码树中不存在任何经过持有张量、
模块、流或缓冲区的对象的引用环。

<a id="the-rule"></a>
## 规则

对象绝不保存指回自身的可调用对象；对象在它所持有的东西上安装的钩子或回调，只能通过弱引用
到达该对象。

| 模式 | 改用 |
| --- | --- |
| `self._impl = self._dense`（绑定在 self 上的方法） | 类函数，`self._impl = MossLinear._dense`，以 `self._impl(self, x)` 调用 |
| `partial(self._run_block, i)` 保存在 self 上 | 对不含 self 的模块（或类）函数取 partial，`model.backbone` 中的 `partial(_run_block, i)`，调用时传入模型 |
| 保存在 self 上、捕获 self 的闭包或 lambda | 对以该对象为参数的模块函数取 partial |
| `p.register_post_accumulate_grad_hook(partial(self._hook, b))`，且 `p` 由 self 持有 | `weak_call(Owner._hook, self, b)`（`kohakumoss_common.weak`） |
| `child.on_grad = self._on_grad`，且 `child` 由 self 持有 | `weak_method(self._on_grad)` |
| 从类外部调用的属性（`base.linear(x, bias)`） | 调用所存类函数的方法：`def linear(self, x, bias): return self._linear(self, x, bias)` |

`nothing`（`kohakumoss_common.calls`）接受任意参数，因此可以原样填入类函数槽位。类函数槽位
是首选；只有在对象确实被指回时才使用弱引用（它持有的张量或模块上的钩子、其子对象调用的
回调）。对象消失后才被调用的弱回调会抛出 ReferenceError：静默什么都不做的钩子会丢掉梯度或
优化器步骤。FusedStep 和 LayerStreamer 不持有它们挂钩的参数 / 层的引用，因此它们的钩子强持有
这些对象，不需要弱引用。每个被弱持有的对象在其整个工作期内都有强所有者（见下表）。

| 弱边 | 被弱持有者 | 强所有者 |
| --- | --- | --- |
| 每个参数上的 GradOffload 梯度累加后钩子 | GradOffload | 内存引擎（`engines.grads`）和 bundle 的 `hooks` |
| 每个参数上的 GradReducer 梯度累加后钩子 | GradReducer | `AsyncReducer.inner`（会话的 `hooks.ddp`），或其创建者 |
| `AsyncReducer.reduced` 钩子主体（FusedStep / GradOffload 的 `final`） | AsyncReducer | 会话的 `hooks.ddp` |
| FusedStep、GradOffload 的 `final`、AsyncReducer slab 的逐参数步骤（`step_param`） | bundle | 训练器状态（`bundle`） |
| StreamedAdamW 块变换 | bundle | 训练器状态（`bundle`） |
| 权重流 `on_grad` / `on_block_backward`、`WeightStream.optimizer` | StreamedAdamW | bundle 的 `blocks` |
| 流水线调度的 `loss_fn`、阶段的 `on_chunk` | PipelineEngine | 流水线驱动（`build_engine` 的调用者） |
| `Target.layer`、绑定的适配器前向的层、推理链的基座前向、LyCORIS 的 `org_module` / `org_forward` | 被适配的线性层 | 模型 |
| LyCORIS 状态字典钩子 | LyCORIS 模块 | 其层（`_modules["lyco"]`）或 AdapterStack 的 `live` 列表 |
| 预热孪生的携带梯度前向 | 孪生的线性层 | 孪生模型 |

源码树之外持有栈帧的引用环会持有它上方每一层调用者栈帧，而活得比其调用更久的栈帧会保留它返回
时带着的局部变量。导入 `torch._dynamo` 会留下这样一个环（`torch.fx.wrap` 保留自己的栈帧）；
首次内核构建会在 `load_backbone` 下导入它，因此 `registry.load_backends` 最后会执行一次循环
gc（[kernel-dev.md](kernel-dev.md#lazy-backends)）。

数据集会被 pickle 进数据加载器工作进程，因此它不持有弱引用：其分辨率规划器保留数据集的
`text_len` 方法（只经过主机侧目录列的环，不含设备张量）。在对象的方法上启动的线程会持有该
对象，直到线程返回（在线缓存写入器的线程在 `close` 时返回）。

选择留在绑定时：所存的类函数就是被选中的实现，直接调用。调用成本与绑定方法相同（一次属性
加载，一次显式传入 self 的调用）。弱回调每次调用多一次弱引用解引用和一次 partial 调用；
这些调用每个参数每步执行一次。

<a id="optimiser-wiring"></a>
## 优化器接线

bundle（`FusedStepBundle` / `Bundle`）、内存引擎和 DDP 归约器是训练器持有的根；从它们所
拥有的对象回到它们的每条边都是弱边：

- FusedStep、GradOffload（`final`）或 AsyncReducer 交付到的逐参数步骤是
  `weak_method(bundle.step_param)`；
- `AsyncReducer.reduced` 返回的钩子主体基于对归约器的弱引用；`block_reduce` 返回强引用的
  主体（归约器持有的任何东西都不会保留它，而调用者可能只持有那个可调用对象）；
- StreamedAdamW 以弱引用在权重流上安装其梯度和块反向回调，`WeightStream.optimizer` 是指向
  它的弱引用；bundle 对 StreamedAdamW 的块变换也是弱引用；
- GradReducer 钩子携带其桶的索引，而不是桶：桶持有其参数，因此挂在其中一个参数上并持有桶的
  钩子会经过该参数构成一个环；
- 流的 BlockReducer 从所有者列表读取块归属，而不是从流读取；
- CPUAdamW 某条 lane 的复制和步骤函数把 lane（以及优化器）作为调用参数传入，而不是持有它们。

<a id="adapters"></a>
## 适配器

绑定的适配器前向安装在所在层的实例字典中，LyCORIS 模块注册为该层的子对象，因此前向和模块
引用的一切都不得指回该层：

- `Target` 弱持有其层（`Target.linear` 是基于弱引用的属性）；`Target.layer` 就是绑定的前向
  所捕获的弱引用，每次调用读取一次。
- 构建模块之后，LyCORIS 的 `org_module` / `org_forward` 会（通过
  `adapters.build.target.weak_org`）重新弱指向该层，训练和推理加载都如此；模块自身的状态
  字典钩子（LyCORIS 以模块的绑定方法注册）也被弱持有（`weak_self_hooks`）。
- `LycorisNetwork.__init__` 会让该网络陷入它自己的一个环（一个引用该网络的递归嵌套函数），
  因此 `build_targets` 会把模块从网络的 `loras` 列表中取出；此后网络本身不持有任何张量。
- 训练 / 评估重新绑定钩子挂在被适配的层上（其 `train`），持有绑定设置和目标，而不是适配器
  集合：目标引用其 LyCORIS 模块，因此挂在模块自身上的钩子会经过它构成一个环。
- 推理链（`AdapterStack`）从层的类前向出发，经过一个弱引用；预热孪生的携带梯度前向是基于其
  线性层的 `weak_call`。

<a id="verification"></a>
## 验证

`tests/_util/release.py`（`check_released`）构建一个对象，在关闭循环回收器的情况下丢弃它，
并检查指向该对象及其拥有的部分（一个权重、一个缓冲区）的弱引用都已失效。各区域的测试会在把
某个环放回的变体上运行它，并要求检查失败。
