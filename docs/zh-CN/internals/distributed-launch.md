[English](../../internals/distributed-launch.md)

<a id="distributed-launch"></a>
# 分布式启动

`kohakumoss.parallel.launch.spawn(fn, plan, *args)` 使用 `torch.multiprocessing.spawn` 为每个设备启动一个进程，并自行初始化进程组。

多 GPU 训练（`PARALLEL` 不为 `"none"`）按设计仅支持 Linux，配置项验证会在 Windows 上拒绝该模式。在 Windows 上，启动器用于 CPU gloo [自检](#self-checks)。

<a id="knobs"></a>
## 配置项

| 配置项 | 取值（默认值在前） | 效果 |
| --- | --- | --- |
| `DIST_BACKEND` | `auto`, `nccl`, `gloo` | `auto` 在 Linux CUDA 上使用 nccl，在 XPU 可用时使用 xccl（仅 `auto` 会选择它），其他情况使用 gloo（Windows、CPU）；gloo 通过主机内存对 CUDA 张量进行 all-reduce |
| `DIST_BIND` | `lazy`, `device` | 进程组设备绑定（[见下文](#device-binding)） |
| `DIST_TIMEOUT` | 600 s | 每个集合通信操作和存储的等待时间 |
| `DIST_GATE_TIMEOUT` | 7200 s（`launch.RANK0_WAIT_S`） | 存储门控等待仅 rank 0 执行或各 rank 分别执行的启动工作的时间（[见下文](#timeouts-and-gates)） |

<a id="process-group"></a>
## 进程组

- **存储**：`init="tcp"` 显式构建 `TCPStore`。父进程在操作系统分配的端口上持有存储主端，工作进程作为客户端连接，因此端口会在任何工作进程使用之前绑定。
  - Windows wheel 不含 libuv，因此该平台的存储使用 `use_libuv=False`，并在每个工作进程中设置 `USE_LIBUV=0`。
  - Windows 上的存储地址为 `localhost`：字面量 `127.0.0.1` 会在非 libuv 存储内部触发 torch 的格式化错误。
- **文件初始化**：`init="file"` 在新的临时目录中使用 `file://` 会合，该目录在作业结束时
  删除（正常退出、某个 rank 失败，或 launcher 收到 SIGTERM，后者随后终止各 rank 进程）。
  tcp 初始化不会创建目录。
- **工作进程**：`fn` 必须是模块级函数，因为生成的进程需要导入它。

<a id="device-binding"></a>
### 设备绑定

`DIST_BIND`（`ParallelPlan.bind`）：

| 值 | 绑定方式 |
| --- | --- |
| `"lazy"`（默认） | 进程组初始化时不传入 `device_id`，因此 NCCL 在首次使用时为每一对点对点通信创建一个 2-rank 通信器，而不是让流水线的每次发送 / 接收都经过同一个共享通信器 |
| `"device"` | nccl / xccl 在初始化时绑定到该 rank 的设备（`device_id`）：在该设备上创建通信器，并在该设备上运行 `dist.barrier()` |

无论取哪个值，流水线都保持惰性；gloo 从不绑定设备。

<a id="timeouts-and-gates"></a>
## 超时和门控

`DIST_TIMEOUT`（`ParallelPlan.timeout_s`，默认 600 s）：每个集合通信操作和存储都会为每个 rank 等待这么长时间，因此挂起的 rank 会在这段时间内使任务停止。

各 rank 耗时不同的启动工作在存储上会合，而非在集合通信操作中会合，最多等待 `DIST_GATE_TIMEOUT`（默认 2 h）：

- **Rank-0 门控。** 不在集合通信操作中等待 rank 0 在步循环之前的一次性工作：其他 rank 在存储上等待（`launch.wait_for_rank0`），直到 rank 0 设置其键（`release_ranks`），然后在之后的集合通信操作中会合。这包括 `writer_first` 中的目录扫描、内核预热测量，以及在 rank 0 上规划并交给其他 rank 的 `MEMORY_PROFILE='auto'`（`launch.rank0_value`）。
- **存储屏障。** 各 rank 耗时不同的启动工作以存储屏障结束（`launch.ranks_meet` / `startup_barrier`）：每个 rank 将 `kohakumoss/<gate>/arrived` 加一，最后一个到达者设置 `kohakumoss/<gate>`，每个 rank 都等待这个键。门控如下：

  | 门控 | 位置 |
  | --- | --- |
  | `data` | 每个 rank 构建加载器之后、构建模型之前 |
  | `prewarmed` | 每个 rank 预热之后、阶段划分和梯度同步 hooks 之前 |
  | `checkpoint` | rank 0 写入检查点的前后 |
  | `sampled` | rank 0 生成采样网格之后 |

在 `data` 和 `prewarmed` 门控之间，构建中的集合通信操作（DDP `swap` 块的共享主机区域）等待 `DIST_TIMEOUT`，因此如果不同 rank 的权重加载耗时之差超过该值，就需要更大的 `DIST_TIMEOUT`。

<a id="failing-rank"></a>
## 失败的 rank

某个 rank 的 `fn` 抛出异常时，会打印回溯、终止自己的 multiprocessing 子进程，然后通过 `os._exit(1)` 退出，不执行进程组销毁；`spawn` 随后检测到退出并终止其他 rank。

- **不销毁。** 在 nccl 上，当失败 rank 已提交发送或接收（流水线阶段处于调度中途）时，销毁会阻塞，因此若先销毁，就会让失败 rank 保持存活，而其对端在接收中等待 `DIST_TIMEOUT`。
- **终止子进程。** `os._exit` 会跳过 DataLoader 关闭：由 forkserver 启动的加载器工作进程监视的是 forkserver 而不是 rank，并且各自持有 forkserver 的保活管道，否则工作进程和 forkserver 都会在 rank 退出后继续存活。
- `tests/parallel/test_launch_failfast.py` 检查退出行为（gloo，以及双 GPU 上的 nccl，其中执行销毁的变体会挂起）和子进程（gloo、Linux）。

<a id="self-checks"></a>
## 自检

`kogine run scripts/dist/check_comm.py --set DEVICES=0,1` 在生成的工作进程中、不使用模型地运行：

- 将分桶梯度归约器与所有 rank 和微批次的梯度总和进行比较（小桶、两个微批次；同时检查梯度是否为桶缓冲区的视图）；
- 对 all-reduce、all-gather / reduce-scatter（FSDP2 的集合通信操作，bf16）以及沿 rank 环的点对点跳转（流水线的链路）进行扫描测试。

引擎自身在 `tests/integration/test_multi_gpu_real.py` 中使用真实模型检查。

实测 all-reduce 带宽（算法 GB/s，64 MB）：

| 设备对 | 链路 | GB/s |
| --- | --- | --- |
| 2x V100 16 GB | NVLink | 90.6 |
| V100 32 GB + V100 16 GB | PCIe，不同根复合体 | 1.07 |
