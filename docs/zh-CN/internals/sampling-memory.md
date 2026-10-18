[English](../../internals/sampling-memory.md)

<a id="sampling-memory"></a>
# 采样内存

一次 `MossPipeline.generate` 调用在各阶段在设备上保留的内容，以及使同一张显卡上采样显存占用不高于训练的放置方案。

- `kohakumoss.inference.peaks.estimated` 估算每个阶段，流水线会在采样前记录这些估算值。
- `scripts/bench/infer/memory.py` 在真实模型上测量这些值（每个阶段一行）。

<a id="phases"></a>
## 阶段

| 阶段 | 设备上的内容 |
| --- | --- |
| 文本编码 | DiT 已放置的权重（每个常驻块、流式传输 DiT 的环形槽位、非块层）+ 文本编码器单次执行的字节数（常驻时为全部，流式传输时为层环，卸载时为整个模型） |
| 采样 | DiT 已放置的权重 + 最大打包批次的一次前向计算（不保存任何激活）+ 文本编码器和 VAE 的常驻字节数 |
| 解码 | DiT 已放置的权重 + VAE 权重 + 对图像的一次 VAE 执行（每像素 `ACT_BYTES_PER_PIXEL`，分块时为一个 512 x 512 图块） |

一次调用的峰值是占用最大的阶段，而不是采样前向计算。

测量示例：`balanced`，fp16，1024^2，RTX 3090，文本编码器采用流式传输：

- 每次调用峰值为 7.24 GiB；
- 其中 4.14 GiB 是已放置的 DiT 权重：42 个块中的 16 个常驻块，加上环形槽位和非块层；
- 其余部分是占用最大的阶段在此基础上增加的内存。根据估算，该阶段为文本编码或解码；在这个尺寸下，采样前向计算的工作集更小。

`balanced` 保持 16 个块常驻，因此不应将其放置方案与小显存显卡的训练配置档比较：训练的 8 GB / 6 GB 层级只保留两个块环形槽位。

<a id="keeping-sampling-at-or-below-training"></a>
## 使采样显存占用不高于训练

- **DiT** 采用该显卡的训练放置方案。`kohakumoss-config new sample` 写入该显卡对应训练显存层级的 `MEMORY_PROFILE`（低于 6 GiB 时采用最小层级；高于最大层级时采用常驻），因此 DiT 权重与训练时保留的权重相同。
- **文本编码器和 VAE** 默认为 `"auto"`，在可用内存中根据 DiT 前向计算的工作集（单个 1024^2 CFG 样本时的 `peaks.working_set`）确定大小（VAE 使用 `offload.encoders.vae.choose_vae_plan`）：

  | 编码器 | `auto` 采用第一个能放下的方案 |
  | --- | --- |
  | 文本编码器 | `resident`，否则为 `ring` |
  | VAE | 常驻 / 卸载中最快的方案；然后是相同方案的分块执行；最后是流式传输的分块执行 |

  训练的小显存显卡层级也以相同方式流式传输文本编码器，并对 VAE 采用卸载和分块执行。
- **在 `TEXT_ENCODE` / `VAE_ENCODE` 中显式设置的任何放置方案**均按原样使用。
