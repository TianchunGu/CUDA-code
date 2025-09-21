/**
 * 课程 2.6: CPU 与 GPU 混合编程
 *
 * 本程序演示：
 * 1. CPU 和 GPU 代码的混合使用
 * 2. 程序执行顺序：CPU -> GPU -> CPU
 * 3. 线程索引的详细使用
 */

#include <stdio.h>

/**
 * GPU 内核函数 - 显示详细的线程信息
 */
__global__ void hello_from_gpu()
{
    // 获取线程标识符
    const int bid = blockIdx.x;     // 线程块索引
    const int tid = threadIdx.x;    // 线程在块内的索引

    // 计算全局线程索引
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    // 每个线程打印自己的标识信息
    printf("Hello World from block %d and thread %d, global id %d\n", bid, tid, id);
}

/**
 * 主函数 - 演示 CPU 与 GPU 的协作
 */
int main(void)
{
    /**
     * 第一步：CPU 执行
     * 在启动 GPU 内核之前，CPU 先打印欢迎信息
     */
    printf("Hello World from CPU!\n");

    /**
     * 第二步：启动 GPU 内核
     * 配置：<<<2, 2>>>
     * - 2 个线程块
     * - 每个线程块 2 个线程
     * - 总共 4 个线程
     *
     * 预期输出：
     * Block 0, Thread 0, Global ID 0
     * Block 0, Thread 1, Global ID 1
     * Block 1, Thread 0, Global ID 2
     * Block 1, Thread 1, Global ID 3
     */
    hello_from_gpu<<<2, 2>>>();

    /**
     * 第三步：CPU 等待 GPU 完成
     * cudaDeviceSynchronize() 确保所有 GPU 操作完成后
     * CPU 才继续执行后续代码
     */
    cudaDeviceSynchronize();

    /**
     * 第四步：程序结束
     * 所有 CPU 和 GPU 操作都已完成
     */
    return 0;
}