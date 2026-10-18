[English](../../guides/install-windows.md)

<a id="install-on-windows"></a>
# 在 Windows 上安装

所有单 GPU 功能都可在 Windows 上运行（[功能矩阵](#feature-matrix)）。多 GPU
训练（DDP、FSDP2、流水线、线程式）按设计仅支持 Linux。

<a id="requirements"></a>
## 要求

| 要求 | 用途 |
| --- | --- |
| Python 3.10+（已测试 3.12 / 3.13；助手及下方命令会创建 3.12 venv） | 所有功能 |
| 足够新、能够运行所安装 torch CUDA 构建的 NVIDIA 驱动，或 Intel Arc 驱动 | GPU |
| 安装了“使用 C++ 的桌面开发”工作负载的 Visual Studio Build Tools（MSVC x64 工具 + Windows SDK） | 从任意控制台编译内核（[编译器设置](#compiler-setup)） |
| CUDA 工具包，TileLang 的 JIT 会调用其中的 `nvcc` | NVIDIA sm70 / sm75（V100、2080 Ti、T4），它们使用 TileLang 内核；缺失时诊断工具会将 `nvcc` 列在 `missing:` 下 |
| `triton-windows`（Triton 后端，sm80+）和 `tilelang`（TileLang 后端，sm70 / sm75 及其他架构） | 内核；两者都有 Windows wheel |

<a id="install"></a>
## 安装

在克隆的仓库中执行以下一条命令，安装助手就会完成下文全部操作
（[install.md](install.md#install-helper)）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1
```

Windows PowerShell 的默认执行策略拒绝运行脚本，因此激活
venv（`.venv\Scripts\Activate.ps1`）时，每个用户需要先执行一次
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`；在 `cmd` 中，
`.venv\Scripts\activate.bat` 可直接运行。

NVIDIA GPU 的手动安装方式（一个环境；[install.md](install.md) 列出了可选依赖组）：

```bash
uv venv -p 3.12 .venv
uv pip install --python .venv/Scripts/python.exe "torch>=2.13,<2.15" torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe -e ".[cuda]" "torch>=2.13,<2.15" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
.venv/Scripts/kohakumoss-doctor
```

在同时配有 NVIDIA GPU 的机器上，Intel Arc（XPU）需要单独的环境，
因为 XPU 版 torch 会替换 CUDA 版；`xpu` 可选依赖组不包含仅限 CUDA 的
`kernels` 可选依赖组（XPU torch wheel 自带 Intel 的 Triton）：

```bash
uv venv -p 3.12 .venv-xpu
uv pip install --python .venv-xpu/Scripts/python.exe "torch>=2.13,<2.15" torchvision --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv-xpu/Scripts/python.exe -e ".[xpu]" "torch>=2.13,<2.15" --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
.venv-xpu/Scripts/kohakumoss-doctor
```

- 文本编码器的 Gated DeltaNet、卷积和注意力层运行在
  `kohakumoss_kernels` 算子上（[text-encoder.md](../internals/text-encoder.md)）。
- `baselines` 可选依赖组（flash-linear-attention、causal-conv1d）用于对比
  基准测试。

`kohakumoss-doctor` 会打印 GPU 架构、工具链及软件包版本、每个内核后端使用的
主机编译器、当前机器上为各算子提供服务的内核后端、先前运行记录的后端失败，
以及缺失项（软件包、仅支持 CPU 的 torch、C++ 构建工具）及其修复命令
（[install.md](install.md#the-doctor)）。

<a id="compiler-setup"></a>
## 编译器设置

用户侧只需安装 Build Tools；内核会自行配置编译器。

进程第一次构建内核时（通常是模型第一次解析算子时），会一次性加载后端
（`registry.load_backends()`）：设置 torch 的编译缓存键标签，并导入
 eager、compile、Triton 和 TileLang 后端。Triton 和 TileLang 的导入，
以及 compile 后端的第一次 CPU 探测，都会调用
`kohakumoss_kernels.hardware.toolchain.ensure_toolchain()`：

- **MSVC 环境。** 在 Visual Studio 开发人员 shell 之外，它会通过 `vswhere` 查找
  安装了 x64 C++ 工具的最新 Visual Studio，执行一次其 `vcvars64.bat`，
  并将它设置的环境（`PATH`、`INCLUDE`、`LIB`、`LIBPATH`、
  `VCINSTALLDIR`、`VCToolsVersion`、Windows SDK 变量）导入进程。结果
  会缓存到内核缓存中（`toolchain/vcvars.json`），Visual Studio 变化时会重新运行。
  首个进程承担 `vcvars64.bat` 的执行开销（在 VS 18 /
  MSVC 14.51 上实测为 0.8 s）；后续进程读取缓存。
- **`CXX`。** 将其设为上述 `cl`，写作 `cl.EXE`，这是 triton-xpu 据以
  选择 MSVC 标志的名称。如果 `CXX` 指向其他编译器，则保留该值。
- **nvcc 的主机编译器** 对 TileLang 而言也是该 `cl`：LyCORIS 的内核导入会设置
  `TILELANG_DISABLE_CLANG_CL=1`，因此即使第一次编译发生在工具链设置之前，
  且 `PATH` 上存在 clang-cl，TileLang 也会解析到 MSVC `cl`
  （`tests/adapters/test_lycoris_host_compiler.py` 会测试两种顺序）。
- **控制台代码页。** Intel Triton 使用控制台代码页读取其 UTF-8 C 驱动源码；
  工具链会使这些读写使用 UTF-8，因此 XPU 驱动和 XPU 上的 inductor
  可在任何控制台代码页下初始化，包括 cp950 / cp936 / cp932。

<a id="when-a-backend-fails"></a>
### 后端失败时

如果后端仍然无法运行某个算子（缺少编译器、驱动 bug、任何 RuntimeError），
该算子会转到架构优先级中的下一个后端（先 compile，再 eager），并发出一条
WARNING：`<op> on <arch>/<dtype>: backend <failed> failed, using <next> (<n> bound ops):
<error>`。

| 失败情况 | 保留方式 |
| --- | --- |
| 构建时遇到的工具链失败（后端探测、预热） | 记录到磁盘；后续运行会跳过该算子的该后端，并发出 WARNING（`backend <b> skipped, failed here (<error>)`）；诊断工具会列出记录，`--clear-failures` 可将其清除 |
| 步骤或采样期间的失败 | 仅在该进程中保留 |

参见 [kernel-dev.md](../internals/kernel-dev.md#fallback)。

<a id="feature-matrix"></a>
## 功能矩阵

| 功能 | Windows | 说明 |
| --- | --- | --- |
| 模型、加载器、保存器、fp16 修正 | 可用 | |
| 推理（MossPipeline、sample CLI） | 可用 | RTX 4090 上 512^2、20 步、bf16：6.1 s，已分配显存峰值 15.7 GiB |
| 训练：原生和 Lightning 驱动、恢复、采样、验证、日志记录器 | 可用 | |
| LyCORIS 适配器、所有算法、融合内核 | 可用 | |
| 导出为 ComfyUI / LyCORIS / 合并权重 / diffusers | 可用 | |
| 数据集软件包、缓存、加载器 worker | 可用 | worker 使用 spawn（Linux：forkserver）；每个 worker 会在第一个批次前依次导入其模块（[data.md](../concepts/data.md#loader)） |
| 注意力内核 | 可用，快于 SDPA | RTX 4090 1 MP x2 前向+反向 13.0 ms，而 SDPA 为 26.9 ms |
| 块算子内核（Triton） | 可用 | |
| 量化基础模型（int8、sm89+ 上的 fp8） | 可用 | |
| 内存配置档、自动规划器 | 可用 | |
| 激活卸载、权重流式传输 | 可用 | WDDM 下 RTX 4090 的页锁定内存 H2D 19.7 / D2H 24.3 GB/s；可分页内存为 8.3 / 13.5 |
| 梯度卸载、CPU AdamW、融合的逐参数更新步骤 | 可用 | |
| 文本编码器：内核、批量 / 打包编码、resident / offload / ring / cpu 放置、提前编码 | 可用 | sm80+ 上使用与 Linux 相同的 Triton 内核；sm70 / sm75 上使用 TileLang |
| Intel Arc（XPU）内核 | 可用 | Triton（Intel 后端）可在普通控制台中使用上述工具链构建；注册表会针对每个算子和形状，对 Triton 内核与 `torch.compile` 计时并绑定更快者（见[性能](../performance/README.md)）；fp16 `kahan` 优化器回写在 Arc 上会漂移（以 1e-5 更新 200 步后为 2e-3，而 CUDA 上低于 1e-5）；Arc 上请使用 bf16 参数或 `nearest` 回写 |
| 多 GPU（DDP / FSDP2 / 流水线 / 线程式） | 不支持 | 按设计仅限 Linux |

<a id="notes"></a>
## 说明

- 在多 GPU 机器上设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，使设备索引与
  `nvidia-smi` 一致。
- 同时驱动显示器的 GPU 可用显存更少，计时噪声也更大。
- Windows 上的 CUDA 分配器没有可扩展段；训练和采样会改为设置
  `roundup_power2_divisions:4`，使预留显存在不同大小的批次间保持平稳
  （运行日志行会显示 `allocator roundup_power2_divisions:4`；
  [offload.md](../internals/offload.md#allocator)）。
