[英文原文](../../concepts/model.md)

<a id="model"></a>
# 模型

`kohakumoss.model.MossBackbone` 是 MOSS-Image transformer 的重新实现，
模块名称与官方 state-dict 键一致，因此 HF 和 ComfyUI 检查点可以直接加载。

<a id="shape-base--sft"></a>
## 结构（base / sft）

| 部分 | 结构 |
| --- | --- |
| `video_embedder` | Linear 128 -> 3072（/16 网格中每个潜变量像素对应一个词元） |
| `text_embedder` | `proj` Linear 4096 -> 3072，然后是 2 个 `TextBlock`（带偏置的融合 `qkv`、qk-norm、RoPE、SiTU FFN，无调制） |
| `time_embedding` | sinusoid(1000 sigma, 256) -> Linear -> SiLU -> Linear |
| `shared_block_modulation` | 所有 42 个块共用一个 Linear 3072 -> 6 x 3072：shift_a、scale_a、gate_a、shift_f、scale_f、gate_f |
| `layers.{0..41}` | 作用于 `[image ; text]` 的联合块：RMSNorm + 调制、q/k/v、逐头 qk RMSNorm + 4 轴 RoPE、注意力、门控残差；RMSNorm + 调制、`ffn_up_gate`（前半部分为 up，后半部分为 gate）、SiTU、`ffn_down`、门控残差 |
| `final_norm_video`, `video_head_modulation`, `final_linear_video` | 图像词元上的调制 RMSNorm 和 Linear 3072 -> 128 |

SiTU 为 `(4 tanh(g/4) sigmoid(g)) * (25 tanh(u/25))`。模型在 [0, 1] 范围内的
sigma 处预测流速度 `noise - x0`（sigma = 1 表示纯噪声）。

<a id="calling-convention"></a>
## 调用约定

```python
v = backbone(img, text, sigma, seq)                       # every block runs

skip = backbone.bind_forward(skip_layers=(22,))           # bound once
v = skip(img, text, sigma, seq)                           # block 22 skipped for all samples

cfg = backbone.bind_forward(segment_skip_layers=(22,))    # bound once
v = cfg(img, text, sigma, seq, mask)                      # skipped where mask[b] is True
```

- `img` [N_img, 128] 和 `text` [N_txt, 4096] 都采用打包形式（见
  [seqinfo.md](seqinfo.md)）；`sigma` 为 fp32 [B]。
- `bind_forward` 一次性解析块执行计划（折叠或 fp16 修复发生变化后需要重新绑定）。
  使用 `segment_skip_layers` 时，一个 bool 掩码 [B] 应用于列出的所有块：
  每个指定块都在整个打包批次上运行，但只保留未标记样本的词元输出
  （`torch.where`）；标记样本保留跳过该块时得到的流（仅应用折叠的重缩放 `r`），
  无论该块对它们计算出了什么，因此其中的 inf 或 NaN 都无法传递给它们。
  打包 CFG 使用这一机制跳过负分支的块 22。
  `sample_latents` 和采样配置项都会将超出 `[0, depth)` 的跳过索引视为错误。
- 时间嵌入、共享调制和输出头调制按样本计算；这些模块存储 fp32 权重
  （`KEEP_FP32`）。它们的线性层是普通的 PyTorch 线性层，在调用方的 autocast 下运行，
  输出转换为 fp32 后用于调制（[numerics.md](numerics.md#conditioning)）。
  时间步正弦嵌入使用 fp32。

<a id="checkpoints"></a>
## 检查点

| 输入 | 处理方式 |
| --- | --- |
| HF `transformer/` 目录（3 个分片 + 索引） | 原样加载 |
| ComfyUI 单文件，裸键或 `model.diffusion_model.` 键 | 去除前缀，将融合的 `proj.qkv` 拆分为 q / k / v |
| KohakuMoss 单文件保存结果 | 原样加载 |

- 加载是严格的：必须填充每个模型键，并使用每个源键
  （`rope.bands` 为派生值，可选）。
- 保存器写出一个使用裸键的单文件（拆分或融合 qkv），或带有
  `config.json` 的 diffusers 分片；写出前先撤销残差折叠。

<a id="ops-and-runners"></a>
## 算子与运行器

- 块通过按 (device, dtype) 一次性解析的 `OpSet` 调用算子。
- `bind_ops()` 绑定每层的融合 linear+epilogue 可调用对象；任何结构性变化
  （附加适配器、更换基座存储）之后都要运行它。
- 块循环调用 `runner.run(i, block, x, ctx)`；`PlainRunner` 直接调用该块，
  其他运行器由内存引擎提供。
