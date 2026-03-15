# EngineeringPractice

## 项目题目

基于 Triton 的深度学习算子优化与实现研究

## 课题说明

随着深度学习的发展，矩阵乘法、卷积、注意力机制、Fused MoE 等计算密集型算子对 GPU 硬件利用率提出了更高要求。虽然 PyTorch、TensorFlow 等主流框架已经提供了丰富的算子库，但在新型算子、特定数据形态以及硬件定制优化场景下，通用实现往往无法达到最优性能。

Triton 作为一个基于 Python 的 GPU 编程语言与编译器，能够用较少的代码编写高性能 GPU Kernel，并在许多场景下达到接近甚至超过手写 CUDA Kernel 的性能。本项目围绕 Triton 的编程模型、算子实现方法和性能优化策略，构建一个可扩展的研究与 Benchmark 框架。

## 项目目标

本项目聚焦以下几个研究方向：

1. 学习 Triton 的编程模型、Kernel 编写方式与编译优化机制。
2. 实现典型算子的 Triton 版本，并与 CUDA 或框架内核进行性能对比。
3. 探索内存访问优化、线程块划分、Kernel 融合等策略的性能收益。
4. 在统一 Benchmark 框架中自动生成测试数据、运行对比实验并输出可视化结果。

## 基础要求

- 了解深度学习基本原理与常见算子。
- 了解 GPU 编程基础。
- 对编译器、高性能计算和算子优化有兴趣。

## 当前项目内容

当前版本已经搭建了一个可直接扩展的项目骨架，重点完成了以下内容：

- 提供 `GEMM` 的 `CUDA` 基线版本与 `Triton` 版本。
- 提供 `FP16` 与 `FP8(E4M3)` 两种精度下的 `GEMM` Benchmark。
- 提供 `Fused MoE` 的研究模板目录，便于后续继续补充实现。
- 提供统一的 Benchmark 入口，可自动构造多组测试数据。
- 支持对同一算子的不同实现进行耗时统计、TFLOPS 统计、正确性校验与画图。
- 默认 `GEMM` Benchmark 会自动构造多组连续 sweep，用于生成更有信息量的性能曲线。
- Benchmark 输出结果保存在 `output/` 目录中。

## 项目结构

```text
EngineeringPractice/
├── benchmarks/
│   ├── __init__.py
│   ├── common.py
│   └── run_benchmark.py
├── operators/
│   ├── __init__.py
│   ├── base.py
│   ├── fused_moe/
│   │   ├── __init__.py
│   │   ├── cuda_impl.py
│   │   └── triton_impl.py
│   └── gemm/
│       ├── __init__.py
│       ├── cuda_impl.py
│       └── triton_impl.py
├── output/
├── .gitignore
├── requirements.txt
└── README.md
```

## Benchmark 设计

本项目的 Benchmark 流程如下：

1. 从 `operators/<operator_name>/` 中读取同一算子的不同实现。
2. 根据算子定义自动生成多组适合测试的数据规模。
3. 对每种实现执行预热与重复测试，统计平均耗时与吞吐结果。
4. 对不同实现的输出进行误差比对，确认数值结果可接受。
5. 将统计结果保存为 `csv`，并将性能对比图保存到 `output/` 目录。

当前 `GEMM` 默认会构造三类数据族：

- `square`：方阵尺寸 sweep，观察规模增长后的性能变化。
- `mlp`：模拟 Transformer/MLP 中常见的 `M x 4096` 与 `4096 x 11008` 乘法。
- `attn`：模拟注意力模块中较常见的 `4096 x 4096` 投影形态。

当前默认优先演示 `GEMM`：

- `operators/gemm/cuda_impl.py`
- `operators/gemm/triton_impl.py`

说明：

- 这里的 `CUDA` 基线当前通过 PyTorch CUDA Kernel 进行封装，便于快速建立对照组。
- 对于 `FP8 GEMM`，当前项目使用 `float8_e4m3fn` 作为前向精度，并优先使用 PyTorch 原生的 `torch._scaled_mm` 路径，不再通过回退到 `FP16 matmul` 冒充 `FP8`。
- 当前项目不再提供 `e5m2` 前向 GEMM case，避免进入 PyTorch 原生接口不支持的路径。
- 原生 `FP8 GEMM` 对硬件和软件有要求：通常需要 Hopper 及以上 GPU，且当前 PyTorch 构建需要提供 `_scaled_mm`。
- 如果后续你要换成自己写的 `.cu` / `cpp_extension` 版本，只需要替换对应实现文件，Benchmark 主流程无需重写。

## 环境准备

建议环境：

- Python 3.10+
- NVIDIA GPU
- CUDA 可用
- PyTorch
- Triton

安装依赖：

```bash
pip3 install -r requirements.txt
```

## 运行方式

### 1. 运行 GEMM Benchmark

```bash
cd EngineeringPractice
python3 benchmarks/run_benchmark.py --operator gemm --plot
```

### 2. 指定测试轮数

```bash
python3 benchmarks/run_benchmark.py --operator gemm --warmup 20 --repeat 100 --plot
```

### 3. 使用轻量级 smoke 配置快速验证

```bash
python3 benchmarks/run_benchmark.py --operator gemm --profile smoke --warmup 5 --repeat 20 --plot
```

### 4. 运行 Fused MoE 模板

```bash
python3 benchmarks/run_benchmark.py --operator fused_moe --plot
```

说明：

- `GEMM` 当前可直接作为示例运行。
- `Fused MoE` 当前提供的是研究模板与目录结构，便于继续补充 Triton Kernel。

## 输出内容

Benchmark 运行结束后，会在 `output/` 中生成类似文件：

- `gemm_benchmark.csv`
- `gemm_latency.png`
- `gemm_tflops.png`
- `gemm_speedup.png`

其中：

- `csv` 用于记录不同实现、不同 shape 下的耗时、TFLOPS、相对 CUDA 的加速比与误差。
- `png` 用于展示按 `family` 分组的时延、吞吐率与加速比曲线，便于在课程设计或论文中直接引用。

## 后续建议

建议你下一步沿着下面的顺序继续完善：

1. 补充 `Fused MoE` 的 Triton Kernel。
2. 增加更多 Benchmark 维度，例如吞吐率、TFLOPS、显存占用。
3. 增加块大小、分组策略、数据类型等超参数搜索。
4. 增加卷积、Attention 等更多算子。
5. 将实验结果整理成课程设计报告或论文实验章节。
