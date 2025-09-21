/**
 * 课程 4.10: 高级 CUDA 项目结构 - 并行归约算法
 * 文件名: CMakeLists.txt
 * 作者: 权 双
 * 日期: 2023-12-31
 * 功能: 高级项目构建配置文件
 *
 * 项目特点：
 * 1. 模块化设计 - 分离头文件和源文件
 * 2. 混合编程 - C++ 和 CUDA 混合编译
 * 3. 性能优化 - CPU vs GPU 性能对比
 * 4. 算法演示 - 并行归约算法的优化
 * 5. 项目规范 - 企业级项目的标准结构
 *
 * 学习目标：
 * - 掌握大型 CUDA 项目的组织结构
 * - 理解模块化编程的优势
 * - 学习 CMake 构建系统的高级用法
 * - 体验软件工程的最佳实践
 */

# ====================================================================================
# 基础构建配置
# ====================================================================================

# 最低版本要求
# 3.20版本提供了更好的CUDA支持和现代CMake特性
cmake_minimum_required(VERSION 3.20)

# 项目信息定义
# ReductionComputation: 项目名称，专注于归约计算
# LANGUAGES: 指定项目使用的编程语言
# - CXX: C++，用于主程序逻辑和工具函数
# - CUDA: CUDA C++，用于GPU内核函数
project(ReductionComputation LANGUAGES CXX CUDA)

# 生成编译命令数据库文件
# 用于IDE和静态分析工具的代码索引
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ====================================================================================
# 编程语言标准设置
# ====================================================================================

# CUDA C++标准版本设置
# C++14提供了足够的现代特性，同时保持兼容性
set(CMAKE_CUDA_STANDARD 14) # 用于指定CUDA编译器应该使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON)    # 表明如果找不到指定版本的CUDA编译器，将发出错误

# 主机端C++标准版本设置
# 保持与CUDA端一致的C++标准
set(CMAKE_CXX_STANDARD 14)  # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的 C++ 编译器，将发出错误

# ====================================================================================
# CUDA 工具包检测和配置
# ====================================================================================

# 查找 CUDA 工具包
# 要求最低版本 11.7，支持现代 GPU 架构和最新特性
find_package(CUDA 11.7 REQUIRED)

# CUDA 工具包检测结果处理
if (CUDA_FOUND)
    # 输出找到的 CUDA 版本信息
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
    # 如果未找到 CUDA，终止构建过程
    message(FATAL_ERROR "Cannot find CUDA")
endif()

# ====================================================================================
# 编译优化标志设置
# ====================================================================================

# CUDA 编译优化标志
# -O3: 最高级别的编译优化，提升运行时性能
# -arch=sm_86: 针对 Ampere 架构（如 RTX 30 系列，A100 等）
# 注意：根据目标 GPU 架构调整 sm_XX 版本
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_86")

# C++ 编译优化标志
# -O3: 与 CUDA 端保持一致的优化级别
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 调试模式编译标志（已注释）
# 开发调试时可以取消注释这些行
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0 -arch=sm_86")   # CUDA调试
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")                    # C++调试

# ====================================================================================
# 源文件和头文件配置
# ====================================================================================

# 递归搜索所有源文件
# 在 src/ 目录下查找所有 .cpp 和 .cu 文件
# GLOB_RECURSE: 递归搜索子目录
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

# 设置头文件包含目录
# inc/ 目录包含所有项目头文件
set(INC "${PROJECT_SOURCE_DIR}/inc")

# ====================================================================================
# 可执行文件构建和链接配置
# ====================================================================================

# 创建可执行文件目标
# reduction: 可执行文件名称
# ${SRCS}: 包含所有源文件的变量
add_executable(reduction ${SRCS})

# 设置头文件包含路径
# PRIVATE: 仅对当前目标可见，不传播给依赖者
target_include_directories(reduction PRIVATE ${INC})          # 项目头文件目录
target_include_directories(reduction PRIVATE ${CUDA_INCLUDE_DIRS})  # CUDA 头文件目录

# 链接 CUDA 运行时库
# 提供 CUDA API 函数的实现
target_link_libraries(reduction PRIVATE ${CUDA_LIBRARIES})

# ====================================================================================
# 项目结构说明
# ====================================================================================
#
# 项目目录结构：
# 4.10lesson/
# ├── CMakeLists.txt        # 本构建配置文件
# ├── inc/                  # 头文件目录
# │   ├── utils.hpp         # 工具函数声明
# │   ├── timer.hpp         # 性能计时器类
# │   └── reduce.hpp        # 归约算法声明
# ├── src/                  # 源文件目录
# │   ├── main.cpp          # 主程序入口
# │   ├── utils.cpp         # 工具函数实现
# │   ├── timer.cpp         # 计时器实现
# │   ├── reduce_cpu.cpp    # CPU 归约实现
# │   └── matmul_gpu_basic.cu # GPU 归约实现
# └── build/                # 构建输出目录（生成）
#
# 构建命令：
# mkdir -p build && cd build
# cmake ..
# make
# ./reduction [size] [blockSize]
#
# 参数说明：
# - size: 数组大小，建议使用2的幂次方
# - blockSize: 线程块大小，通常为64、128、256等
#
# 示例运行：
# ./reduction 1024 256
# ./reduction 4096 512
# ./reduction 16384 1024
#
# ====================================================================================
