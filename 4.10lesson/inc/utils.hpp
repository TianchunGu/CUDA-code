/**
 * 课程 4.10: 高级 CUDA 项目结构 - 工具函数头文件
 * 文件名: utils.hpp
 * 作者: 权 双
 * 日期: 2023-12-31
 * 功能: 通用工具函数和宏定义
 *
 * 本文件提供：
 * 1. CUDA 错误检查宏
 * 2. 内核错误检查机制
 * 3. 日志输出功能
 * 4. 数据初始化和验证函数
 * 5. 矩阵操作工具函数
 *
 * 使用说明：
 * - CUDA_CHECK: 包装 CUDA API 调用，自动检查错误
 * - CUDA_KERNEL_CHECK: 检查内核执行后的错误
 * - LOG: 格式化日志输出
 */

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>
#include <system_error>
#include <stdarg.h>

/**
 * CUDA 错误检查宏定义
 *
 * CUDA_CHECK(call):
 * - 执行 CUDA API 调用并检查返回值
 * - 如果出错，打印错误信息和位置并退出
 * - 使用示例: CUDA_CHECK(cudaMalloc(&d_ptr, size));
 *
 * CUDA_KERNEL_CHECK():
 * - 检查最近一次内核执行是否有错误
 * - 通常在 cudaDeviceSynchronize() 之后调用
 * - 使用示例: kernel<<<grid, block>>>(); CUDA_KERNEL_CHECK();
 *
 * LOG(...):
 * - 格式化日志输出，支持可变参数
 * - 使用示例: LOG("Value: %d", value);
 */
#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK()          __kernelCheck(__FILE__, __LINE__)
#define LOG(...)                     __log_info(__VA_ARGS__)

/**
 * 默认线程块大小定义
 * 16x16 = 256 线程，适合大多数 GPU 架构
 * 可根据具体硬件和算法需求调整
 */
#define BLOCKSIZE 16

/**
 * CUDA API 错误检查函数
 *
 * @param err: CUDA API 返回的错误码
 * @param file: 调用位置的文件名（通过 __FILE__ 宏获取）
 * @param line: 调用位置的行号（通过 __LINE__ 宏获取）
 *
 * 功能：
 * - 检查 CUDA API 调用是否成功
 * - 如果失败，打印详细错误信息
 * - 包括错误位置、错误代码和错误描述
 * - 出错后立即终止程序
 *
 * inline static 说明：
 * - inline: 建议编译器内联，减少函数调用开销
 * - static: 限制函数作用域在当前编译单元
 */
inline static void __cudaCheck(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

/**
 * CUDA 内核执行错误检查函数
 *
 * @param file: 调用位置的文件名
 * @param line: 调用位置的行号
 *
 * 功能：
 * - 检查最近一次内核启动是否有错误
 * - 使用 cudaPeekAtLastError() 获取错误状态
 * - 不会清除错误状态，只是查看
 * - 适合在内核调用后立即检查
 *
 * 使用场景：
 * - 内核启动语法错误
 * - 内核配置参数错误
 * - 内核执行时的非法内存访问
 * - 设备断言失败
 */
inline static void __kernelCheck(const char* file, const int line)
{
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

/**
 * 格式化日志输出函数
 *
 * @param format: printf 风格的格式字符串
 * @param ...: 可变参数列表
 *
 * 功能：
 * - 提供类似 printf 的格式化输出
 * - 自动添加换行符
 * - 使用固定大小缓冲区避免内存分配
 * - 线程安全的输出到 stdout
 *
 * 实现细节：
 * - va_list: 处理可变参数
 * - vsnprintf: 安全的格式化字符串
 * - 1000 字节缓冲区足够大多数日志消息
 *
 * 使用示例：
 * LOG("GPU kernel completed in %.2f ms", time);
 * LOG("Processing %d elements", count);
 */
static void __log_info(const char* format, ...)
{
    char msg[1000];     // 固定大小缓冲区
    va_list args;       // 可变参数列表
    va_start(args, format);  // 初始化参数列表

    // 安全的格式化，防止缓冲区溢出
    vsnprintf(msg, sizeof(msg), format, args);

    // 输出到标准输出，自动换行
    fprintf(stdout, "%s\n", msg);
    va_end(args);       // 清理参数列表
}

/**
 * 矩阵/数组初始化函数
 *
 * @param data: 要初始化的数组指针
 * @param size: 数组元素个数
 * @param seed: 随机数种子，确保可重现性
 *
 * 功能：使用随机数初始化数组，值范围 [0, 1]
 */
void initMatrix(float* data, int size, int seed);

/**
 * 矩阵/数组打印函数
 *
 * @param data: 要打印的数组指针
 * @param size: 数组元素个数
 *
 * 功能：格式化打印数组内容，便于调试
 */
void printMat(float* data, int size);

/**
 * CPU 和 GPU 结果比较函数
 *
 * @param h_data: CPU 计算结果（参考答案）
 * @param d_data: GPU 计算结果（待验证）
 * @param size: 数组元素个数
 *
 * 功能：
 * - 逐元素比较两个数组
 * - 使用浮点数容差判断相等
 * - 发现差异时报告位置和数值
 */
void compareMat(float* h_data, float* d_data, int size);

#endif //__UTILS_HPP__//
