[English](../../performance/README.md)

<a id="performance"></a>
# 性能

| 页面 | 内容 |
| --- | --- |
| [attention.md](attention.md) | 各 GPU 上变长注意力与 PyTorch SDPA 的对比 |
| [block-ops.md](block-ops.md) | 各 GPU 上归一化 / 残差 / qk-rope / SiTU / 损失 / 融合线性层与 PyTorch eager 及 compile 的对比 |
| [offload-synthetic.md](offload-synthetic.md) | 在 MOSS 规模的合成块上检查卸载选项及硬件极限，涵盖 V100 和 RTX 4090 |
| [vae.md](vae.md) | 各 GPU 上按 dtype、compile、分块、批量划分的 VAE 编码 / 解码每张图像耗时（ms）及峰值显存 |
| [bench-matrix.md](bench-matrix.md) | 真实 GPU 基准测试矩阵：每张显卡的速度 / 显存 / 质量 / 推理运行、功能覆盖情况及报告 |
