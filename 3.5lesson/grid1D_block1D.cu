/**
 * 课程 3.5: CUDA 线程组织模型详解 - 一维网格一维线程块
 * 文件名: grid1D_block1D.cu
 * 作者: 权 双
 * 日期: 2023-08-14
 * 功能: 使用一维网格和一维线程块组织模型计算二维矩阵加法
 *
 * 线程组织模型特点：
 * 1. 一维网格 (1D Grid) - 线程块在网格中按一维排列
 * 2. 一维线程块 (1D Block) - 线程在线程块中按一维排列
 * 3. 列导向处理 - 每个线程处理一整列
 * 4. 串行循环 - 在内核内部使用循环处理第二维
 *
 * 与其他模型的区别：
 * - 最简单的线程组织方式
 * - 使用最少的并行线程
 * - 结合并行和串行的处理方式
 * - 适合特定的计算模式
 *
 * 学习目标：
 * - 理解最基础的线程组织
 * - 掌握并行与串行的结合
 * - 分析不同组织模式的权衡
 * - 选择合适的处理策略
 */

#include <stdio.h>
#include "../tools/common.cuh"

/**
 * GPU 内核函数 - 一维网格一维线程块矩阵加法
 *
 * @param A: 输入矩阵 A
 * @param B: 输入矩阵 B
 * @param C: 输出矩阵 C (C = A + B)
 * @param nx: 矩阵宽度（列数）
 * @param ny: 矩阵高度（行数）
 *
 * 处理策略：
 * - 每个线程负责处理矩阵的一列
 * - 在线程内部循环处理该列的所有行
 * - 结合并行（列间）和串行（列内）处理
 */
__global__ void addMatrix(int *A, int *B, int *C, const int nx, const int ny)
{
    /**
     * 计算当前线程对应的列索引
     *
     * 一维线程组织的索引计算：
     * - threadIdx.x: 线程在块内的位置 (0 到 blockDim.x-1)
     * - blockIdx.x: 块在网格中的位置 (0 到 gridDim.x-1)
     * - blockDim.x: 线程块大小
     * - ix = 当前线程处理的列号
     *
     * 示例：矩阵宽度 16，线程块大小 4
     * - Block 0: 线程处理列 0, 1, 2, 3
     * - Block 1: 线程处理列 4, 5, 6, 7
     * - Block 2: 线程处理列 8, 9, 10, 11
     * - Block 3: 线程处理列 12, 13, 14, 15
     */
    int ix = threadIdx.x + blockIdx.x * blockDim.x;

    /**
     * 边界检查 - 确保不超出矩阵宽度
     *
     * 只有有效的列索引才进行处理
     */
    if (ix < nx)
    {
        /**
         * 串行处理该列的所有行
         *
         * 处理模式：
         * - 每个线程处理一整列（ny 行）
         * - 在线程内部使用 for 循环串行处理
         * - iy 从 0 到 ny-1，遍历所有行
         *
         * 内存访问模式：
         * - 跨行访问（stride access）
         * - 可能不是最优的缓存利用
         * - 但简化了线程组织
         *
         * 计算负载：
         * - 每个线程的工作量：ny 个元素
         * - 总线程数：nx 个线程
         * - 工作均衡：每个线程工作量相同
         */
        for (int iy = 0; iy < ny; iy++)
        {
            /**
             * 计算当前元素的一维索引
             *
             * 行优先存储映射：
             * idx = iy * nx + ix
             *
             * 对于线程 ix，处理的元素索引为：
             * - (0, ix) → 0 * nx + ix
             * - (1, ix) → 1 * nx + ix
             * - (2, ix) → 2 * nx + ix
             * - ...
             * - (ny-1, ix) → (ny-1) * nx + ix
             *
             * 内存访问特点：
             * - 同一线程的访问间隔为 nx（步长访问）
             * - 不同线程访问相邻内存位置
             * - 可能导致缓存未命中
             */
            int idx = iy * nx + ix;

            /**
             * 执行矩阵加法
             * C[idx] = A[idx] + B[idx]
             *
             * 每个线程串行执行 ny 次加法操作
             */
            C[idx] = A[idx] + B[idx];
        }
    }
}

/**
 * 主函数 - 一维网格一维线程块演示
 */
int main(void)
{
    /**
     * 设置GPU设备
     */
    setGPU();

    /**
     * 矩阵参数配置
     */
    int nx = 16;        // 矩阵宽度（列数）
    int ny = 8;         // 矩阵高度（行数）
    int nxy = nx * ny;  // 总元素数
    size_t stBytesCount = nxy * sizeof(int);

    /**
     * 分配和初始化主机内存
     */
    int *ipHost_A, *ipHost_B, *ipHost_C;
    ipHost_A = (int *)malloc(stBytesCount);
    ipHost_B = (int *)malloc(stBytesCount);
    ipHost_C = (int *)malloc(stBytesCount);

    if (ipHost_A != NULL && ipHost_B != NULL && ipHost_C != NULL)
    {
        // 初始化测试数据
        for (int i = 0; i < nxy; i++)
        {
            ipHost_A[i] = i;
            ipHost_B[i] = i + 1;
        }
        memset(ipHost_C, 0, stBytesCount);
    }
    else
    {
        printf("主机内存分配失败!\n");
        exit(-1);
    }

    /**
     * 分配和初始化设备内存
     */
    int *ipDevice_A, *ipDevice_B, *ipDevice_C;
    ErrorCheck(cudaMalloc((int**)&ipDevice_A, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int**)&ipDevice_B, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int**)&ipDevice_C, stBytesCount), __FILE__, __LINE__);

    if (ipDevice_A != NULL && ipDevice_B != NULL && ipDevice_C != NULL)
    {
        ErrorCheck(cudaMemcpy(ipDevice_A, ipHost_A, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpy(ipDevice_B, ipHost_B, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpy(ipDevice_C, ipHost_C, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    }
    else
    {
        printf("设备内存分配失败\n");
        free(ipHost_A);
        free(ipHost_B);
        free(ipHost_C);
        exit(1);
    }

    /**
     * 配置内核启动参数 - 一维组织
     *
     * 线程块配置：block(4, 1)
     * - blockDim.x = 4: X 方向 4 个线程
     * - blockDim.y = 1: Y 方向 1 个线程（实际是一维）
     * - 每个线程块包含 4 个线程
     *
     * 网格配置：grid(4, 1)
     * - gridDim.x = (16+4-1)/4 = 4: X 方向 4 个线程块
     * - gridDim.y = 1: Y 方向 1 个线程块（实际是一维）
     * - 网格包含 4 个线程块
     *
     * 线程分布特点：
     * - 总线程数：4 个线程块 × 4 个线程/块 = 16 个线程
     * - 每个线程处理 8 个元素（一列）
     * - 总计算量：16 个线程 × 8 个元素/线程 = 128 个元素
     *
     * 与其他模型的对比：
     * - grid2D_block2D: 128 个线程，每线程 1 个元素
     * - grid2D_block1D: 128 个线程，每线程 1 个元素
     * - grid1D_block1D: 16 个线程，每线程 8 个元素
     */
    dim3 block(4, 1);   // 一维线程块：4×1
    dim3 grid((nx + block.x - 1) / block.x, 1);  // 一维网格：4×1

    printf("线程配置：网格:<%d, %d>, 线程块:<%d, %d>\n", grid.x, grid.y, block.x, block.y);
    printf("线程组织模式：一维网格 + 一维线程块\n");
    printf("处理方式：%d 个线程，每线程处理 %d 行（一列）\n", grid.x * block.x, ny);
    printf("总线程数：%d 个线程块 × %d 个线程/块 = %d 个线程\n",
           grid.x * grid.y, block.x * block.y, grid.x * grid.y * block.x * block.y);
    printf("每线程工作量：%d 个元素\n", ny);

    /**
     * 启动内核
     */
    addMatrix<<<grid, block>>>(ipDevice_A, ipDevice_B, ipDevice_C, nx, ny);

    /**
     * 获取结果并验证
     */
    ErrorCheck(cudaMemcpy(ipHost_C, ipDevice_C, stBytesCount, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    printf("\n计算结果验证（前10个元素）：\n");
    for (int i = 0; i < 10; i++)
    {
        printf("id=%d, A=%d, B=%d, C=%d (预期:%d)\n",
               i + 1, ipHost_A[i], ipHost_B[i], ipHost_C[i], 2 * i + 1);
    }

    /**
     * 清理资源
     */
    free(ipHost_A);
    free(ipHost_B);
    free(ipHost_C);

    ErrorCheck(cudaFree(ipDevice_A), __FILE__, __LINE__);
    ErrorCheck(cudaFree(ipDevice_B), __FILE__, __LINE__);
    ErrorCheck(cudaFree(ipDevice_C), __FILE__, __LINE__);

    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    printf("\n程序执行完成\n");
    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. 一维线程组织特点：
 *    - 最简单的线程组织方式
 *    - 索引计算最直观
 *    - 结合并行和串行处理
 *    - 线程数量最少
 *
 * 2. 处理模式对比：
 *    grid2D_block2D: 完全并行
 *    - 128 线程并行处理 128 个元素
 *    - 每线程工作量：1 个元素
 *
 *    grid2D_block1D: 部分并行
 *    - 128 线程并行处理 128 个元素
 *    - 每线程工作量：1 个元素
 *
 *    grid1D_block1D: 混合处理
 *    - 16 线程并行处理 16 列
 *    - 每线程串行处理 8 个元素
 *
 * 3. 性能特点分析：
 *    优势：
 *    - 线程数少，启动开销小
 *    - 索引计算简单
 *    - 适合计算密集型任务
 *    - 减少线程调度开销
 *
 *    劣势：
 *    - 并行度不够充分
 *    - 内存访问模式不理想（跨行访问）
 *    - 可能无法充分利用 GPU 资源
 *    - 缓存效率可能较低
 *
 * 4. 适用场景：
 *    - 数据集较小的情况
 *    - 计算密集型操作
 *    - 列优先的处理算法
 *    - 简单的原型验证
 *
 * 5. 内存访问模式：
 *    - 跨行访问（stride access）
 *    - 访问间隔为 nx
 *    - 可能导致缓存未命中
 *    - 不是最优的访问模式
 *
 * 6. 线程组织选择指导：
 *    - 完全并行：适合简单元素级操作
 *    - 混合处理：适合复杂计算逻辑
 *    - 数据大小：影响线程组织选择
 *    - 硬件特性：考虑 GPU 架构特点
 *
 * 7. 优化建议：
 *    - 增加每线程的工作量
 *    - 优化内存访问模式
 *    - 考虑使用共享内存
 *    - 根据数据大小调整策略
 *
 * 8. 实际应用考虑：
 *    - 这种模式在实际中较少使用
 *    - 主要用于学习和理解
 *    - 为更复杂的组织提供基础
 *    - 在特定场景下可能有优势
 */
