[English](../../guides/install.md)

<a id="install"></a>
# 安装

torch 必须来自与 GPU 匹配的 PyTorch 索引：NVIDIA 驱动能够运行的 CUDA 构建（cu126 /
cu128 / cu130），或面向 Intel Arc 的 XPU 构建。下述每种方式都会先从该索引安装
torch，再使用加速器的默认可选依赖组（`cuda` 或 `xpu`）安装 KohakuMoss，
最后运行 `kohakumoss-doctor`。

| 方式 | 适用情况 |
| --- | --- |
| [从克隆的仓库安装](#from-a-clone)，使用[安装助手](#install-helper) | 常规安装：Linux、Windows、Google Colab |
| [从 PyPI 安装](#from-pypi) | 无需克隆；wheel 包含配置和脚本 |
| [手动安装](#manual-install) | 自行选择 venv 或索引 |

<a id="from-a-clone"></a>
## 从克隆的仓库安装

```bash
git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer && cd KohakuMossTrainer
```

| 平台 | 命令 |
| --- | --- |
| Linux | `bash scripts/install/install.sh` |
| Windows | `powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1` |
| Google Colab | `!git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer /content/KohakuMossTrainer && python /content/KohakuMossTrainer/scripts/colab/setup.py` |

助手会以可编辑模式安装克隆的仓库；下方使用 `-e ".[cuda]"` 的手动步骤
也是如此。

Colab：`scripts/colab/setup.py` 会在 `/content/km-venv` 创建 uv venv，使用与
虚拟机驱动匹配的 torch 构建，连同可选依赖组一起安装软件包并运行诊断工具；
`scripts/colab/kohakumoss_colab.ipynb` 是完整的 notebook。

<a id="install-helper"></a>
## 安装助手

```bash
bash scripts/install/install.sh                                         # Linux
powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1    # Windows
```

助手会：

1. 检测 GPU 和驱动（`nvidia-smi` 的 CUDA 版本；Intel Arc 适配器），
   如果无法从 `nvidia-smi` 读取 CUDA 版本，则显示提示并停止；
2. 选择索引：驱动 CUDA >= 13.0：cu130，>= 12.8：cu128，>= 12.6：cu126；Intel：
   xpu；两者都没有：cpu；
3. 创建 Python 3.12 venv `.venv`：有 uv 时使用 `uv venv -p 3.12`；没有时，
   使用 `python -m venv` + pip，其中 Linux 在安装了 `python3.12` 时使用它（否则
   使用 `python3`），Windows 使用 python.org 安装版本的 `py` 启动器（单独的 `python`
   可能是 Microsoft Store 别名）；
4. 从该索引安装 `torch>=2.13,<2.15`（`KM_TORCH` 可覆盖 torch
   版本要求）；
5. 安装 KohakuMoss（从克隆的仓库运行时采用可编辑安装，否则从 PyPI 安装
   `kohakumoss`），并运行诊断工具。

激活 venv：

| shell | 命令 |
| --- | --- |
| bash | `source .venv/bin/activate` |
| PowerShell | `.venv\Scripts\Activate.ps1`（每个用户需要执行一次 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`） |
| `cmd` | `.venv\Scripts\activate.bat` |

<a id="from-pypi"></a>
## 从 PyPI 安装

在克隆的仓库外运行助手，或执行下方的手动步骤，都会从 PyPI 安装 `kohakumoss`。
wheel 将 `configs/` 和 `scripts/` 作为包数据（`kohakumoss/bundle/`）携带，
因此 `kohakumoss-config new` 和配方无需克隆仓库即可使用：

- 它打印的命令会通过绝对路径指定内置脚本（`kogine run
  <site-packages>/kohakumoss/bundle/scripts/train/moss.py --config my_run.py`）；
- 它的配置会写入当前目录。

<a id="manual-install"></a>
## 手动安装

先从与你的驱动相符的索引安装 torch（`nvidia-smi` 会打印其 CUDA 版本：
>= 13.0 cu130，>= 12.8 cu128，>= 12.6 cu126；Intel Arc：xpu），然后安装软件包，
同时将该索引设为附加索引，以保留该 torch 构建：

```bash
uv venv -p 3.12
uv pip install "torch>=2.13,<2.15" torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install "kohakumoss[cuda]" "torch>=2.13,<2.15" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```

- 从克隆的仓库安装时，将 `"kohakumoss[cuda]"` 替换为 `-e ".[cuda]"`。
- Intel Arc 使用 `xpu` 索引和 `[xpu]`。
- 没有 uv 时，用 `pip install` 执行相同的两步（使用 `--extra-index-url`，不使用
  `--index-strategy`）。

LyCORIS（`lycoris-lora>=4.1.2`）会作为依赖从 PyPI 安装。在安装软件包后，
可通过 `uv pip install -e <path to LyCORIS>` 将其替换为检出的 LyCORIS 仓库
（Colab 安装程序的 `--lycoris <path>` 作用相同）。

<a id="the-doctor"></a>
## 诊断工具

```bash
kohakumoss-doctor
```

它会打印操作系统、Python、torch 构建、每张 GPU 及其架构、软件包版本、
各内核后端的主机编译器、各后端在各设备上的冒烟构建结果
（`status triton@cuda:0 ok` 或失败信息）、各算子使用的后端，以及
`missing:` 部分，其中包含每一项的修复命令：

- 基础软件包或加速器默认可选依赖组中缺失或版本不符的依赖；
- NVIDIA GPU 环境中安装了仅支持 CPU 的 torch，或驱动无法运行的 CUDA torch；
- Windows：MSVC `cl.exe`（带 C++ 工作负载的 VS Build Tools 2022；工具链会自行
  从任意控制台导入 `vcvars64`）和 Windows SDK；
- CUDA：工具包的 `nvcc`（TileLang JIT）；
- XPU：缺少 `pytorch-triton-xpu`，或同时安装了 `triton-windows`；
- 冒烟构建失败的每个内核后端，以及可能的修复办法。

| 标志 | 效果 |
| --- | --- |
| `--json report.json` | 写出相同的报告（不包含路径、主机名或用户名） |
| `--memory` | 添加针对本地设备解析后的所有内存配置档 |
| `--clear-failures` | 清除已记录的构建期工具链失败（记录的失败会使后续运行跳过该算子的该后端，并发出 WARNING） |

Windows 工具链和功能矩阵见 [install-windows.md](install-windows.md)。
多 GPU 训练仅限 Linux。

<a id="triton-and-tilelang"></a>
## Triton 和 TileLang

| 平台 | Triton | TileLang |
| --- | --- | --- |
| Linux、CUDA | 随 torch wheel 提供 | `tilelang`（可选依赖组 `kernels-cuda`）；JIT 需要 `nvcc` |
| Windows、CUDA | `triton-windows`（可选依赖组 `kernels-cuda`） | `tilelang`；JIT 需要 MSVC `cl.exe`、Windows SDK 和 `nvcc` |
| Intel Arc | `pytorch-triton-xpu`，与 torch 一起来自 XPU torch 索引 | - |

- `triton-windows` 3.7 对应 torch 2.13，3.8 对应 torch 2.14；可选依赖组将版本限定为
  `>=3.7,<3.9`，软件包将版本限定为 `torch>=2.13,<2.15`，以保持配对（助手
  和此处命令也重复了该版本限制；`tests/rules/test_torch_pin.py` 确保它们
  一致）。
- `xpu` 可选依赖组使用 `kernels-xpu`（空：Intel 的 Triton 随 torch 提供）。

<a id="where-caches-go"></a>
## 缓存位置

首次在某台机器上运行时，会构建并调优内核；结果保存在磁盘上，之后的运行可跳过这些工作。
每种缓存都有自己的位置，且每个位置都可通过环境变量移动（适用于 home 或系统盘较小，或使用
共享盘的情况）：

| 缓存 | 默认位置 | 移动方式 |
| --- | --- | --- |
| KohakuMoss 内核缓存：调优表（`tuning/`）、设备探测（`devices/`）、记录的后端失败（`failures/`）、Windows 工具链（`toolchain/`） | `~/.cache/kohakumoss_kernels` | `KOHAKUMOSS_KERNEL_CACHE` |
| LyCORIS 融合内核调优 | `~/.cache/lycoris_kernels` | `LYCORIS_KERNEL_CACHE_DIR` |
| Triton 编译内核 | `~/.triton/cache` | `TRITON_CACHE_DIR`（或 `TRITON_HOME`） |
| `torch.compile`（TorchInductor）图及其 Triton 内核 | `<系统临时目录>/torchinductor_<user>` | `TORCHINDUCTOR_CACHE_DIR`（设置后，其 Triton 内核跟随 `TRITON_CACHE_DIR`） |
| TileLang 编译内核（sm70 / sm75） | `~/.tilelang/cache` | `TILELANG_CACHE_DIR` |
| 数据集缓存：latents、文本嵌入、目录文件 | `<第一个数据集路径>/cache` | `CACHE_DIR` 配置项（[data.md](../concepts/data.md)） |

- 删除内核缓存是安全的：下次运行会重新构建并调优。
- 调优表按显卡、CUDA 和 torch 版本分别保存，因此一个缓存目录可被多张 GPU 和多个安装共享。

<a id="extras"></a>
## 可选依赖组

| 可选依赖组 | 软件包 | 用途 |
| --- | --- | --- |
| `cuda` | `kernels-cuda` + `lightning` + `quant` | NVIDIA GPU 上的默认组合 |
| `xpu` | `kernels-xpu` + `lightning` + `quant` | Intel Arc 上的默认组合 |
| `kernels-cuda`（`kernels`） | `tilelang`、`triton-windows`（Windows） | TileLang 内核（sm70 / sm75）、Windows 上的 Triton |
| `kernels-xpu` | 无 | Intel 的 Triton 随 XPU torch 提供 |
| `lightning` | `lightning` | `DRIVER="lightning"` |
| `quant` | `bitsandbytes` | `OPTIMIZER="adamw8bit"`（bitsandbytes 8 位 AdamW） |
| `optim` | `prodigyopt`、`schedulefree` | `OPTIMIZER="prodigy"` / `"schedule_free"` |
| `parquet` | `pyarrow` | `parquet` 数据集来源 |
| `wandb` | `wandb` | `LOGGER="wandb"` |
| `dev` | pytest、ruff、black、lpips + `bench` | 测试和 lint（脚本测试会导入每个脚本） |
| `bench` | matplotlib、pandas | 基准测试、fp16 修复图表 |
| `baselines` | flash-linear-attention、causal-conv1d | 仅用于对比基准测试 |

真实 GPU 基准测试矩阵（`scripts/bench/matrix/`，
[bench-matrix.md](../performance/bench-matrix.md)）有自己的开发包，
即 `kohakumoss-bench`：从克隆的仓库安装时，在安装软件包后执行
`uv pip install -e scripts/bench/matrix`。

<a id="prebuilt-wheels"></a>
## 预构建 wheel

- `kohakuvault`（依赖）从 wheel 安装：对 CPython 3.10 - 3.13 上的 win_amd64 和
  manylinux_2_28 x86_64 执行仅二进制解析（`uv pip compile --only-binary :all:`），
  可以找到最新发布的 `kohakuvault 0.9.0`（及其依赖 numpy）。
