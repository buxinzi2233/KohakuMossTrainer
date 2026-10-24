[English](../README.md)

<a id="kohakumoss-docs"></a>
# KohakuMoss 文档

集成入口：[外部接口参考](guides/interfaces.md) — CLI 默认值、run 文件、指标、续训/停止与可复用 Python 入口。

快速入口：[安装](guides/install.md) · [第一个 LoRA](guides/first-lora.md) · [配置与预设](guides/configs.md) · [RTX 5090 训练基准](performance/training-5090.md)

| 章节 | 内容 |
| --- | --- |
| [guides/](guides/README.md) | 安装（Windows / Linux / Intel Arc / Colab）、配置与预设、主机 RAM、数据集与缓存、第一个 LoRA、全量微调、推理与对比网格、导出到 ComfyUI、日志、多 GPU、脚本目录 |
| [concepts/](concepts/README.md) | 模型、SeqInfo 与打包、数值精度、PyTorch 精度默认值与 fp16 修复、适配器、训练、显存配置档、数据管线 |
| [internals/](internals/README.md) | 内核开发、调优、fp16 块运算、量化与 fp8 内核、卸载引擎、采样显存、文本编码器、分布式启动、适配器记录与绑定、检查点写入 |
| [performance/](performance/README.md) | 按 GPU 实测的表格：注意力、块运算、VAE、真实 GPU 基准测试矩阵 |

新用户请先读[安装](guides/install.md)，再读[你的第一个 LoRA](guides/first-lora.md)；
若想通过少量选项启动一次运行，请参阅[示例脚本](../../scripts/examples/README.md)。架构图
（包结构、训练步骤、显存放置、内核选择、多 GPU）位于 [README](../../README.md#architecture)。
