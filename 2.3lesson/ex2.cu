/**
 * 课程 2.3 实例 2: CUDA 线程标识和索引
 *
 * 本程序演示：
 * 1. 如何获取线程的块索引 (blockIdx)
 * 2. 如何获取线程在块内的索引 (threadIdx)
 * 3. 如何计算全局线程索引
 * 4. CUDA 的分层线程模型
 */

#include <stdio.h>

/**
 * GPU 内核函数 - 演示线程索引系统
 */
__global__ void hello_from_gpu()
{
    /**
     * CUDA 内置变量：
     * - blockIdx.x: 当前线程块在网格中的索引 (0-based)
     * - threadIdx.x: 当前线程在线程块中的索引 (0-based)
     * - blockDim.x: 线程块的大小（每个块的线程数）
     */
    const int bid = blockIdx.x;     // 线程块索引
    const int tid = threadIdx.x;    // 线程在块内的索引

    /**
     * 全局线程索引计算公式：
     * global_id = threadIdx.x + blockIdx.x * blockDim.x
     *
     * 这个公式将二维的 (block, thread) 坐标映射为一维的全局索引
     */
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    // 打印每个线程的详细信息
    printf("Hello World from block %d and thread %d, global id %d\n", bid, tid, id);
}

/**
 * 主函数
 */
int main(void)
{
    /**
     * 内核调用配置：<<<2, 4>>>
     *
     * 线程组织结构：
     * Block 0: Thread 0,1,2,3 -> Global IDs: 0,1,2,3
     * Block 1: Thread 0,1,2,3 -> Global IDs: 4,5,6,7
     *
     * 输出顺序可能不确定，因为线程是并行执行的
     */
    hello_from_gpu<<<2, 4>>>();

    // 同步等待所有 GPU 操作完成
    cudaDeviceSynchronize();

    return 0;
}