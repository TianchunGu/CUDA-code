/**
 * 课程 2.3 实例 1: CUDA 基础内核调用
 *
 * 本程序演示：
 * 1. 基本的 GPU 内核函数定义
 * 2. 不同的网格和线程块配置 (2×4)
 * 3. 与课程 1.2 的区别在于配置参数不同
 */

#include <stdio.h>

/**
 * GPU 内核函数
 *
 * 每个线程都会执行此函数
 * 与课程 1.2 相同的函数，但调用配置不同
 */
__global__ void hello_from_gpu()
{
    // 每个线程打印相同的消息
    printf("Hello World from the the GPU\n");
}

/**
 * 主函数
 */
int main(void)
{
    /**
     * 内核调用配置：<<<2, 4>>>
     * - 网格大小: 2 个线程块
     * - 线程块大小: 每个线程块 4 个线程
     * - 总线程数: 2 × 4 = 8 个线程
     *
     * 因此会看到 8 行输出
     */
    hello_from_gpu<<<2, 4>>>();

    // 等待 GPU 完成所有操作
    cudaDeviceSynchronize();

    return 0;
}