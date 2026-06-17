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
- **新增**: 提供 `Fused KV Materialize` 的 `CUDA`、`PyTorch` 和 `Triton` 三种实现。
- **新增**: 提供 `Rotate Input IDs` 的 `CUDA`、`PyTorch` 和 `Triton` 三种实现。
- 提供统一的 Benchmark 入口，可自动构造多组测试数据。
- 支持对同一算子的不同实现进行耗时统计、TFLOPS 统计、正确性校验与画图。
- 默认 `GEMM` Benchmark 会自动构造多组连续 sweep，用于生成更有信息量的性能曲线。
- `Triton GEMM` 当前已经接入 autotune 配置搜索，用于在不同矩阵规模下自动选择更优的块划分参数。
- Benchmark 输出结果保存在 `output/` 目录中。

## 新增算子说明

### Fused KV Materialize

`Fused KV Materialize` 是 LLM 推理中用于高效 KV Cache 构建的关键算子，参考自 SGLang 的实现。该算子融合了以下操作：

1. **KV 投影输出处理**: 从投影结果中分离 K 和 V
2. **RMSNorm**: 对 K 应用 RMS 归一化
3. **RoPE**: 对 K 应用旋转位置嵌入 (Neox-style)
4. **V 直通**: V 不经处理直接输出

该算子在 speculative decoding 和 multi-layer Eagle 模型中被广泛使用。

**文件结构**:
```
operators/fused_kv_materialize/
├── __init__.py        # Benchmark case 定义
├── cuda_impl.py       # CUDA kernel 实现
├── torch_impl.py      # PyTorch 参考实现
└── triton_impl.py     # Triton kernel 实现
```

### Rotate Input IDs

`Rotate Input IDs` 是 speculative decoding 中用于序列旋转的关键算子，同样参考自 SGLang。该算子的功能是：

1. 将每个序列中的元素向左移动一位（位置 i+1 的元素移到位置 i）
2. 在序列末尾插入新的 token

这个操作常用于 Eagle 模型的多轮预测场景，无需完整重处理序列即可更新 token。

**文件结构**:
```
operators/rotate_input_ids/
├── __init__.py        # Benchmark case 定义
├── cuda_impl.py       # CUDA kernel 实现
├── torch_impl.py      # PyTorch 参考实现
└── triton_impl.py     # Triton kernel 实现
```

## 项目结构

```text
EngineeringPractice/
├── benchmarks/
│   ├── __init__.py
│   ├── common.py
│   ├── run_benchmark.py
│   └── benchmark_all.py           # 新增：批量运行所有算子
├── operators/
│   ├── __init__.py
│   ├── base.py
│   ├── fused_moe/
│   │   ├── __init__.py
│   │   ├── cuda_impl.py
│   │   └── triton_impl.py
│   ├── gemm/
│   │   ├── __init__.py
│   │   ├── cuda_impl.py
│   │   └── triton_impl.py
│   ├── fused_kv_materialize/      # 新增
│   │   ├── __init__.py
│   │   ├── cuda_impl.py
│   │   ├── torch_impl.py
│   │   └── triton_impl.py
│   └── rotate_input_ids/          # 新增
│       ├── __init__.py
│       ├── cuda_impl.py
│       ├── torch_impl.py
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

### 5. 运行 Fused KV Materialize Benchmark

```bash
# 运行规整版测试
python3 benchmarks/run_benchmark.py --operator fused_kv_materialize --profile regular --plot

# 运行不规整版测试（模拟 padding 等场景）
python3 benchmarks/run_benchmark.py --operator fused_kv_materialize --profile irregular --plot

# 运行默认测试
python3 benchmarks/run_benchmark.py --operator fused_kv_materialize --plot
```

### 6. 运行 Rotate Input IDs Benchmark

```bash
# 运行规整版测试
python3 benchmarks/run_benchmark.py --operator rotate_input_ids --profile regular --plot

# 运行不规整版测试（模拟变长序列、padding 等场景）
python3 benchmarks/run_benchmark.py --operator rotate_input_ids --profile irregular --plot

# 运行默认测试
python3 benchmarks/run_benchmark.py --operator rotate_input_ids --plot
```

### 7. 批量运行所有算子的 Benchmark

```bash
# 运行所有算子的规整版和不规整版测试
python3 benchmarks/benchmark_all.py --plot

# 指定预热和重复次数
python3 benchmarks/benchmark_all.py --warmup 20 --repeat 100 --plot
```

说明：

- `GEMM` 当前可直接作为示例运行。
- `Fused MoE` 当前提供的是研究模板与目录结构，便于继续补充 Triton Kernel。
- `Fused KV Materialize` 和 `Rotate Input IDs` 提供了完整的 CUDA、PyTorch 和 Triton 三种实现。
- 所有新增算子支持 `regular`（规整版）和 `irregular`（不规整版）两种 profile，便于对比不同场景下的性能表现。

## 输出内容

Benchmark 运行结束后，会在 `output/` 中生成类似文件：

- `gemm_benchmark.csv`
- `gemm_latency.png`
- `gemm_tflops.png`
- `gemm_speedup.png`
- `fused_kv_materialize_benchmark.csv`    # 新增
- `fused_kv_materialize_latency.png`      # 新增
- `fused_kv_materialize_speedup.png`      # 新增
- `rotate_input_ids_benchmark.csv`        # 新增
- `rotate_input_ids_latency.png`          # 新增
- `rotate_input_ids_speedup.png`          # 新增
- `combined_benchmark.csv`                # 新增：所有算子的汇总结果

其中：

- `csv` 用于记录不同实现、不同 shape 下的耗时、TFLOPS、相对 CUDA 的加速比与误差。
- `png` 用于展示按 `family` 分组的时延、吞吐率与加速比曲线，便于在课程设计或论文中直接引用。

## Benchmark 测试场景设计

### 规整版测试 (Regular Profile)

规整版测试模拟理想场景，使用均匀分布的数据规模：

**Fused KV Materialize Regular**:
- 固定 `num_kv_heads=8`, `head_dim=128`, `rotary_dim=128`
- Sweep `total_ctx`: 64, 128, 256, 512, 1024, 2048
- Sweep `head_dim`: 64, 96, 128, 160, 192, 256

**Rotate Input IDs Regular**:
- Sweep `batch_size`: 1, 2, 4, 8, 16, 32, 64, 128
- Sweep `seq_len`: 8, 16, 32, 64, 128, 256, 512, 1024

### 不规整版测试 (Irregular Profile)

不规整版测试模拟真实场景，包含变长序列、padding 等情况：

**Fused KV Materialize Irregular**:
- **prefill_batch**: 模拟 prefill 批处理，序列长度从 1 到最大值不等
- **decode_batch**: 模拟 decode 阶段，大量小型序列
- **mixed_padding**: 模拟不同 padding 比率 (10%, 30%, 50%, 70%) 的批处理

**Rotate Input IDs Irregular**:
- **variable_lengths**: 序列长度从 1 到 max_seq_len 随机分布
- **power2_lengths**: 幂次序列长度（常见于 padding 批处理）
- **padding_simulation**: 模拟不同 padding overhead 的场景
- **decode_tiny**: Decode 场景，大量极小序列（1-3 tokens）

## 后续建议

建议你下一步沿着下面的顺序继续完善：

1. 补充 `Fused MoE` 的 Triton Kernel。
2. 增加更多 Benchmark 维度，例如吞吐率、TFLOPS、显存占用。
3. 增加块大小、分组策略、数据类型等超参数搜索。
4. 增加卷积、Attention 等更多算子。
5. 将实验结果整理成课程设计报告或论文实验章节。
6. 对 `Fused KV Materialize` 和 `Rotate Input IDs` 的 CUDA kernel 进行更细粒度的优化（如 shared memory tiling、warp-level reduce 等）。
