[English](../../internals/encode-feed.md)

<a id="encode-feed"></a>
# 编码供给（Encode feed）

`ENCODE_DEVICES` 为文本编码器和 VAE 分配专属显卡：`DEVICES` 只训练，`ENCODE_DEVICES`
只编码。一次运行会拆分为三类进程：

| 进程 | 运行内容 | 模块 |
| --- | --- | --- |
| main | 每个数据 rank 的数据加载器、编码供给、批次流 | `training.encode.feed` |
| encoder（每张 `ENCODE_DEVICES` 显卡一个） | 文本编码器和 VAE | `training.encode.service` |
| training（DDP / FSDP2 / pipeline 下每张 `DEVICES` 显卡一个，否则一个） | 仅 DiT 步骤 | `training.encode.streamed` |

```python
DEVICES = (0, 1, 2, 3)      # 4x RTX 3090: training
ENCODE_DEVICES = (4, 5)     # 2x RTX 2080 Ti: text encoder + VAE
ENCODE_DEPTH = 32           # micro-batches kept encoded ahead per data stream
ENCODE_CHUNK = 8            # micro-batches per encoder request
```

所有布局都能在编码供给下工作；单进程布局（`none`、`threaded`）在主进程的一个子进程中训练。

<a id="data-flow"></a>
## 数据流

1. 主进程为每个数据 rank 构建一个加载器（它写出目录，训练进程读取该目录），并在
   `VALIDATE_EVERY` > 0 时为每个 rank 构建一个验证加载器。
2. 每条数据流有一个生产者线程，它加载批次的进度最多超前其训练 rank 已取用的批次
   `ENCODE_DEPTH` 个。
3. 调度器把最多 `ENCODE_CHUNK` 个待处理批次（来自所有数据流）作为一块送入一个请求队列，
   所有编码器进程都从该队列取用，因此无论 rank 数多少，编码器显卡都共享这些工作。某个请求
   槽位空出时（每个编码器进程两个）就收集一块，因此该块包含其间到达的所有内容。
4. 编码器进程用一次文本编码器调用编码一块（执行前先把提示词截断到各自的 `text_limit` 个
   token，每次内核调用打包到 `TEXT_ENCODE['max_tokens']`）以及一次 VAE 调用（图像按尺寸
   排序，相同尺寸归为一批）。
5. 主进程在缓存命中把批次带回来时完成每个批次：文本按在线缓存存储的方式舍入并截断到
   `text_limit`，潜变量舍入并收集到保留的 token 上；在线缓存在这里拿到新编码；像素被丢弃。
6. 批次按加载器顺序送往各自的训练 rank（一个释放缓冲区会把各编码器进程完成的块重新排序）；
   一个数据 rank 服务多个 rank 的布局会把每个批次放入每个 rank 的队列。

训练进程从自己的队列取用就绪批次并移到自己的显卡上；它从不与编码器进程通信。它不持有文本
编码器；采样 rank 持有一个用于解码样本的 VAE，并在构建时从编码供给获取样本提示词的嵌入。

<a id="state-and-resume"></a>
## 状态与续训

流式批次会携带其之后的加载器状态；训练进程的加载器状态就是它取用的最后一个批次的状态，
因此检查点会从第一个尚未训练过的批次恢复。续训时训练进程请求编码供给执行 seek：一旦该流的
所有 rank 都发出请求，该流就在那里以新一代重新开始，旧一代的批次在两端都被丢弃。

<a id="failure"></a>
## 失败

编码器进程、加载器或编码供给线程失败时，会在主进程中记录其回溯并设置编码供给的失败事件；
每个训练进程在下次等待时抛出 `FeedError`，启动器随后结束本次运行。主进程消失后，编码器
进程也会结束。

<a id="metrics"></a>
## 指标

- `feed_wait`：训练进程每步等待就绪批次的秒数；编码供给保持超前时为 0。
- `feed_encode`：每步其批次占用的编码器进程秒数（文本编码器 + VAE，每块的时间按 token /
  图像分摊）。低于步时乘以每张编码器显卡所服务的 rank 数时，编码器显卡就能跟上。
- `te_*` / `vae_*`：文本编码器和 VAE 的占比，与进程内编码器相同。

<a id="process-setup"></a>
## 进程设置

编码器进程像训练进程一样设置日志、分配器、编译池和 TUNE，把自己所在的显卡设为当前设备，
并在第一轮执行前预热其文本编码器。张量通过共享内存跨越队列（`file_system` 共享）。
