[English](../../guides/scripts.md)

<a id="scripts"></a>
# 脚本

所有脚本都从仓库根目录运行，使用以下三种方式之一（“运行方式”列）：

| 运行方式 | 命令 | 设置 |
| --- | --- | --- |
| kogine | `kogine run <script> --config <file> --set KEY=VALUE` | 模块级配置项（[configs.md](configs.md)） |
| python | `python <script> --flag` | 命令行标志 |
| shell | `.\<script>.ps1`（PowerShell）或 `bash <script>.sh` | 脚本参数 |

每个脚本的文档字符串（或文件头）都包含完整用法。

<a id="train-sample-export"></a>
## 训练、采样、导出

| 脚本 | 运行方式 | 作用 |
| --- | --- | --- |
| `scripts/train/moss.py` | kogine | 训练：LyCORIS 适配器或全量微调，支持所有配置项（[first-lora.md](first-lora.md)、[training.md](../concepts/training.md)） |
| `scripts/sample/sample.py` | kogine | 使用基础权重和适配器生成图像，每次运行生成一个元数据文件（[inference.md](inference.md#cli)） |
| `scripts/sample/compare_grid.py` | kogine | 对比网格：针对提示词列表，先展示基础模型，再展示每个检查点（[inference.md](inference.md#comparison-grid)） |
| `scripts/export/export.py` | kogine | 将训练状态中的适配器导出为 ComfyUI / LyCORIS 文件，或导出合并后的 / diffusers transformer（[export-to-comfy.md](export-to-comfy.md)） |
| `scripts/export/merge.py` | kogine | 将 LyCORIS、ComfyUI 或训练状态适配器文件合并到基础 transformer 中 |
| `scripts/export/fp16_fix.py` | kogine | fp16 修正检查点：将残差折叠写入权重（[numerics.md](../concepts/numerics.md#the-fp16-fix-checkpoint)） |
| `scripts/export/fp8_quant.py` | kogine | 预量化的 fp8 / MXFP8 transformer（训练和采样中的 `TRANSFORMER=<file>`） |
| `scripts/export/text_quant.py` | kogine | 预量化的文本编码器（`TEXT_ENCODE` 中的 `stored`） |

<a id="examples"></a>
## 示例

| 脚本 | 运行方式 | 作用 |
| --- | --- | --- |
| `scripts/examples/train.ps1`、`train.sh` | shell | 通过少量选项进行短程训练：写入配置、检查配置、运行 `scripts/train/moss.py`（[scripts/examples/README.md](../../../scripts/examples/README.md)） |
| `scripts/examples/sample.ps1`、`sample.sh` | shell | 对 `scripts/sample/sample.py` 执行同样的操作：基础权重、适配器、fp16 修正 transformer、放置方式 |
| `scripts/examples/grid.ps1`、`grid.sh` | shell | 对 `scripts/sample/compare_grid.py` 执行同样的操作：适配器、训练示例运行的导出文件或检查点、按画师划分的数据集描述文本（`grid_prompts.py`） |

<a id="data"></a>
## 数据

| 脚本 | 运行方式 | 作用 |
| --- | --- | --- |
| `scripts/data/inspect_data.py` | kogine | 目录与第 0 轮计划的统计信息、示例描述文本（[datasets.md](datasets.md)） |
| `scripts/data/catalog_build.py` | kogine | 扫描数据集一次并写入其目录，或将其导入 `moss_shard` |
| `scripts/data/cache_text.py` | kogine | 离线文本嵌入缓存 |
| `scripts/data/cache_latents.py` | kogine | 离线潜变量缓存 |
| `scripts/data/cache_merge.py` | kogine | 将在线缓存的逐 rank 分片并入主文件 |
| `scripts/data/catalog_cache.py` | kogine | 显示、删除或压缩目录缓存文件 |

<a id="setup"></a>
## 设置

| 脚本 | 运行方式 | 作用 |
| --- | --- | --- |
| `scripts/install/install.sh`、`install.ps1` | shell | 从与当前 GPU / 驱动匹配的索引安装 torch、带默认可选依赖组的 KohakuMoss，并运行诊断工具（[install.md](install.md)） |
| `scripts/colab/setup.py` | python | Colab venv，使用与虚拟机驱动匹配的 torch 构建；`kohakumoss_colab.ipynb` 是 notebook |
| `scripts/doctor/doctor.py` | python | 环境报告（`kohakumoss-doctor`） |
| `scripts/bench/tune/tune.py` | kogine | 一次性填充各显卡的内核调优表（`kohakumoss-tune`、[logging.md](logging.md#kernel-tuning)） |

<a id="multi-gpu-linux"></a>
## 多 GPU（Linux）

| 脚本 | 运行方式 | 作用 |
| --- | --- | --- |
| `scripts/dist/check_comm.py` | kogine | 检查进程组并扫描集合通信 / p2p 链路（[multi-gpu.md](multi-gpu.md)） |
| `scripts/dist/scaling_report.py` | kogine | 一组运行的扩展性表格：samples/s、效率、步骤分解、损失一致性 |

<a id="benches"></a>
## 基准测试

每个基准测试都会在 `runs/bench/<area>/` 下为每张显卡写入一个 JSON；
[performance/](../performance/README.md) 和 [internals/](../internals/README.md)
中的表格来自这些文件。

| 脚本 | 测量内容 |
| --- | --- |
| `scripts/bench/kernel/attention.py` | 变长注意力，各后端与 PyTorch 基线的对比 |
| `scripts/bench/kernel/norm_mod.py`、`qk_rope.py`、`residual.py`、`ffn.py`、`fused_linear.py` | 块算子与 PyTorch eager 及 `torch.compile` 的对比 |
| `scripts/bench/kernel/quant.py` | int8 / fp8 仅权重量化的冻结基础模型线性层，与反量化 + 矩阵乘法的对比 |
| `scripts/bench/kernel/fp8.py` | fp8 / MXFP8 计算线性层与 16 位 `F.linear` 的对比 |
| `scripts/bench/kernel/optim.py` | 优化器更新步骤内核与 `torch.optim.AdamW` 的对比 |
| `scripts/bench/adapters/fold_base.py` | 将基础分支加法折叠进 LyCORIS 内核的 LoRA 旁路 |
| `scripts/bench/textenc/textenc.py`、`ops.py`、`quant.py` | 文本编码器：各放置方式和布局下的 tokens/s、其算子、其量化格式 |
| `scripts/bench/vae/vae.py` | 各 dtype 下 VAE 编码 / 解码的精度、速度与内存 |
| `scripts/bench/offload/stream.py`、`knobs.py`、`limits.py` | 复制带宽、各内存配置项的独立表现、配置项组合相对于链路的表现 |
| `scripts/bench/step/step.py` | 通过训练器测量各内存配置档的训练步骤 |
| `scripts/bench/train/memory.py` | 训练器启动和最初几个步骤的分阶段设备内存 |
| `scripts/bench/infer/memory.py` | 一次 generate 调用的分阶段设备内存 |
| `scripts/bench/data/loader.py`、`catalog.py` | 加载器吞吐量；目录扫描、打开和 token 预算填充 |
| `scripts/bench/matrix/launch.py`、`host.py` | 真实 GPU 基准测试矩阵（开发包 `kohakumoss_bench`、`uv pip install -e scripts/bench/matrix`）：每张显卡的速度 / 显存 / 质量 / 推理运行（[bench-matrix.md](../performance/bench-matrix.md)） |

<a id="probes"></a>
## 探测

使用真实权重检查模型本身；每项都会写入一份 JSON 报告。

| 脚本 | 检查内容 |
| --- | --- |
| `scripts/probe/embed.py` | 将探测提示词编码一次，供其他探测使用 |
| `scripts/probe/parity.py` | 模型和流水线与官方实现的对比 |
| `scripts/probe/trajectory.py` | 一条采样轨迹，使用官方循环或本项目循环，以及两者的对比 |
| `scripts/probe/activations.py` | fp32 下逐层的数值幅度，以及 fp16 的折叠因子 |
| `scripts/probe/backward.py` | fp16 下逐模块的前向 / 反向数值幅度 |
| `scripts/probe/adapter_e2e.py` | 运行中模型上的适配器，与由 `MossPipeline` 加载的导出文件的对比 |
| `scripts/probe/fp8_inference.py` | 文本编码器和 DiT 的 fp8 / MXFP8 推理，与 bf16 和 fp64 的对比 |
