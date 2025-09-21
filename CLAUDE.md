# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是一个 CUDA 编程教程仓库，包含结构化的课程，涵盖 CUDA 基础知识，从基本概念到高级内存优化技术。仓库采用带编号课程目录的渐进式学习结构。

## 仓库结构

### 课程组织
- **课程 1.x**: GPU/CUDA 概念介绍、安装和基础工具
- **课程 2.x**: 核心 CUDA 编程 - 内核函数、线程模型、编译
- **课程 3.x**: 实用 CUDA 编程 - 矩阵操作、错误处理、计时、GPU 查询
- **课程 4.x**: 高级内存管理 - 寄存器、全局内存、共享内存、常量内存、优化

### 课程目录模式
- 简单课程：单个 `.cu` 文件直接编译
- 复杂课程：基于 CMake 的项目，源文件/头文件分离组织
- 现代课程 (4.4+)：包含 `CMakeLists.txt`、`common.cuh` 和构建目录
- 高级课程 (4.10)：完整项目结构，包含 `src/`、`inc/` 目录

### 通用文件
- `tools/common.cuh`: 共享的 CUDA 实用函数，包括错误检查和 GPU 设置
- 独立的 `common.cuh`: 课程专用实用程序（课程 4.4-4.8）
- `README.md`: 中文版课程概述和课程索引

## 构建命令

### 对于基于 CMake 的课程 (4.4, 4.5, 4.6, 4.7, 4.8, 4.10)
```bash
cd [课程目录]
mkdir -p build
cd build
cmake ..
make
```

### 对于简单的 CUDA 文件
```bash
nvcc -o [输出名称] [源文件].cu
# 示例: nvcc -o hello hello.cu
```

### 通用 CMake 配置
- 需要 CUDA 11.6+ （各课程有所不同）
- CUDA 和 C++ 均采用 C++14 标准
- 使用 `-O3` 优化构建
- GPU 架构：`sm_86`（根据目标硬件调整）
- 调试标志可用但已注释

## 开发工作流程

### 构建项目
1. 始终使用现有的 CMakeLists.txt 文件作为新 CMake 项目的模板
2. 检查 CMakeLists.txt 中的 CUDA 版本要求（范围从 11.6 到 11.7）
3. 根据需要基于目标硬件修改 GPU 架构（`-arch=sm_86`）

### 错误处理
- 对所有 CUDA API 调用使用 `common.cuh` 中的 `ErrorCheck()` 函数
- 函数签名：`cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)`
- 使用模式：`ErrorCheck(cudaFunction(), __FILE__, __LINE__)`

### GPU 设置
- 使用 `common.cuh` 中的 `setGPU()` 函数初始化和设置 GPU 设备
- 自动检测 GPU 数量并设置设备 0 用于计算

## 代码架构

### 内存管理模式
- **全局内存**：使用 `cudaMalloc`/`cudaFree` 的基本 GPU 内存分配
- **共享内存**：课程 4.5 中的静态和动态分配示例
- **常量内存**：课程 4.6 中的只读内存优化
- **缓存优化**：课程 4.7 中的 L1/L2 缓存使用模式

### 线程模型
- 早期课程中的 1D 网格 + 1D 块模式
- 课程 3.5 中的 2D 网格 + 1D/2D 块组合
- 不同网格/块配置的线程索引计算

### 项目结构（高级课程）
```
lesson/
├── CMakeLists.txt
├── src/           # 源文件 (.cu, .cpp)
├── inc/           # 头文件 (.hpp, .cuh)
├── build/         # 生成的构建文件
└── common.cuh     # 课程专用实用程序
```

## 测试和验证

### 性能测量
- 使用 CUDA 事件为 GPU 操作计时（课程 3.3）
- 比较 CPU 与 GPU 性能实现
- 使用示例中展示的 nvprof 工具进行性能分析

### 硬件查询
- 使用 GPU 属性查询来调整代码以适应硬件能力（课程 3.4）
- 检查计算能力和内存规格

## 重要说明

- 课程材料为中文，但代码注释和结构具有自解释性
- 仓库包含教育示例和实用实现
- 构建目录已添加到 gitignore，但可能包含编译的二进制文件
- 一些课程目录名有拼写错误（例如 "2.2lession" 而不是 "2.2lesson"）
- GPU 架构设置为 sm_86 - 构建前请验证与目标硬件的兼容性