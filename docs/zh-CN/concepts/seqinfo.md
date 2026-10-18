[英文原文](../../concepts/seqinfo.md)

<a id="seqinfo-and-packing"></a>
# SeqInfo 与打包

每个批次都采用打包形式：模型中的每个张量都有一个展平的词元轴，而 `SeqInfo`
携带样本边界。

- 所有样本的图像词元组成一个轴 `[N_img, C]`，以样本优先排列，每个样本的潜变量网格
  `(h, w)` 内按行优先排列。
- 文本词元组成第二个轴 `[N_txt, D]`。
- 样本 i 的联合序列为 `[img_i ; txt_i]`；联合张量以样本优先、按此顺序排列。
  `img_in_joint` / `txt_in_joint` 将这两个轴的内容散布到联合序列中。

<a id="building-it"></a>
## 构建

- `build_seqinfo(hw, text_len, img_keep=None)` 在主机上运行，每个批次一次。
- `img_keep[i]`（样本 i 的行优先网格中的已排序索引，或用 None 表示全部）
  只保留这些图像词元（随机词元丢弃）：`n_img`、`cu_img` 和每个图像轴
  都只计入保留的词元，每个保留的词元都保有其原始网格单元的 RoPE 位置。
- `SeqInfo.to(device)` 按数据类型将主机张量合并为各自一个锁页缓冲区后传输
  （在 CUDA 上每种数据类型各复制一次）；设备上的字段都是该缓冲区的视图。

<a id="fields"></a>
## 字段

| 字段 | 含义 |
| --- | --- |
| `cu_img`, `cu_txt`, `attn_joint.cu_seqlens`, `attn_text.cu_seqlens` | 分段边界 |
| `seg_img`, `seg_txt`, `seg_joint` | 每个词元的样本索引（用于选择逐样本调制） |
| `pos_joint`, `pos_txt` | fp32 的 RoPE 坐标 `(r, f, h, w)` |

<a id="positions"></a>
## 位置

位置遵循官方布局：图像词元的 `f = text_len`，`h, w` 居中；
文本词元 j 的 `f = j`。因此图像位置取决于各样本的真实文本长度，
这也是文本从不填充、注意力从不跨越样本边界的原因。

一个批次可以打包不同分辨率和文本长度的样本，代价只取决于它们的词元数量：
注意力按分段运行，其他所有算子按词元运行。
