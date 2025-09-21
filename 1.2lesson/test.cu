/**
 * 课程 1.2: CUDA 基础 Hello World 程序
 *
 * 本程序演示了最基本的 CUDA 编程概念：
 * 1. 定义和调用 GPU 内核函数
 * 2. 线程网格和线程块的基本配置
 * 3. CPU 与 GPU 之间的同步
 */

#include <stdio.h>

/**
 * GPU 内核函数（Kernel Function）
 *
 * __global__ 关键字标识：
 * - 这是一个在 GPU 上执行的函数
 * - 可以从 CPU 代码调用
 * - 函数返回类型必须是 void
 *
 * 当内核被调用时，GPU 会启动大量并行线程来执行这个函数
 */
__global__ void hello_from_gpu()
{
    // printf 可以在 GPU 内核中使用（需要 CUDA 2.0+ 和 Compute Capability 2.0+）
    // 每个线程都会执行这行代码，因此会打印多次
    printf("Hello World from the the GPU\n");
}

/**
 * 主函数 - 在 CPU 上执行
 */
int main(void)
{
    /**
     * 内核调用语法：function_name<<<grid_size, block_size>>>()
     *
     * <<<4, 4>>> 配置说明：
     * - 第一个参数 (4): 网格大小 (Grid Size) - 启动 4 个线程块
     * - 第二个参数 (4): 线程块大小 (Block Size) - 每个线程块包含 4 个线程
     *
     * 总线程数 = 网格大小 × 线程块大小 = 4 × 4 = 16 个线程
     * 因此会看到 16 行 "Hello World from the the GPU" 输出
     */
    hello_from_gpu<<<4, 4>>>();

    /**
     * GPU 内核调用是异步的，CPU 不会等待 GPU 完成
     * cudaDeviceSynchronize() 强制 CPU 等待所有 GPU 操作完成
     * 这确保在程序结束前所有 printf 输出都已完成
     */
    cudaDeviceSynchronize();

    return 0;
}
