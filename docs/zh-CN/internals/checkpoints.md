[English](../../internals/checkpoints.md)

<a id="checkpoint-writes"></a>
# 检查点写入

检查点如何写入磁盘，以及主机内存和磁盘耗时方面的成本。

| 文件 | 写入器 | 峰值时的额外主机内存 |
| --- | --- | --- |
| `trainable.safetensors`、`ema.safetensors`、EMA 快照 | `kohakumoss.training.checkpoint.stream.save_stream` | 一个张量 |
| `optim.pt` | `torch.save` | 每次一个存储；FSDP2 下为完整主机状态，在写入前汇集（[见下文](#optimizer-state)） |

<a id="where-to-write"></a>
## 写到哪里

写入速度由目标磁盘决定。5.1 B 参数的全量微调（fp32 主权重、两个矩状态、EMA）会产生数十 GB 数据，因此检查点目录应位于本地 NVMe 上。对于慢速或共享磁盘，请先写入快速临时目录，再进行复制。

<a id="streaming-safetensors-writes"></a>
## 流式 safetensors 写入

`save_stream` 分两遍写入：

1. 仅根据元数据构建 safetensors 头部（按状态顺序排列的每个张量的 dtype、形状和字节偏移）。
2. 随后将每个张量复制到主机并单独写入。CPU 上的连续张量原位写入，不进行复制。

峰值时的额外主机内存为一个张量；先将整个状态复制到主机的写入器则会保留第二份完整副本。`tests/training/checkpoint/test_checkpoint_stream.py` 在新进程中测量两种写入器的峰值 RSS 增长，并检查 safetensors（Rust）读取器能否精确加载写入的文件。

<a id="optimizer-state"></a>
## 优化器状态

`optim.pt` 通过 `torch.save` 写入，每次写入一个存储。FSDP2 下，写入前会在每个 rank 上将各分片汇集为完整主机张量（[multi-gpu.md](../guides/multi-gpu.md)）。
