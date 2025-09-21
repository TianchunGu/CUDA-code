/**
 * 课程 3.5: CUDA 线程组织模型详解 - 二维网格一维线程块
 * 文件名: grid2D_block1D.cu
 * 作者: 权 双
 * 日期: 2023-08-14
 * 功能: 使用二维网格和一维线程块组织模型计算二维矩阵加法
 *
 * 线程组织模型特点：
 * 1. 二维网格 (2D Grid) - 线程块在网格中按二维排列
 * 2. 一维线程块 (1D Block) - 线程在线程块中按一维排列
 * 3. 混合维度映射 - 结合两种组织方式的优势
 *
 * 与 grid2D_block2D 的区别：
 * - 线程块内部使用一维组织，更简单的索引计算
 * - Y 方向通过 blockIdx.y 直接映射
 * - 适合某些特定的内存访问模式
 *
 * 学习目标：
 * - 理解混合维度线程组织
 * - 掌握不同维度间的索引转换
 * - 比较不同组织模式的特点
 * - 选择适合的线程组织策略
 */

#include <stdio.h>
#include "../tools/common.cuh"

/**
 * GPU 内核函数 - 二维网格一维线程块矩阵加法
 *
 * @param A: 输入矩阵 A
 * @param B: 输入矩阵 B
 * @param C: 输出矩阵 C (C = A + B)
 * @param nx: 矩阵宽度（列数）
 * @param ny: 矩阵高度（行数）
 *
 * 索引计算特点：
 * - X 坐标：通过一维线程块计算
 * - Y 坐标：直接使用 blockIdx.y
 * - 每行由多个一维线程块处理
 */
__global__ void addMatrix(int *A, int *B, int *C, const int nx, const int ny)
{
    /**
     * X 坐标计算（水平方向）
     *
     * 一维线程块中的坐标计算：
     * - threadIdx.x: 线程在块内的位置 (0 到 blockDim.x-1)
     * - blockIdx.x: 块在网格 X 方向的位置
     * - blockDim.x: 线程块在 X 方向的大小
     *
     * 示例：矩阵宽度 16，线程块大小 4
     * - Block(0,y): 处理列 0-3
     * - Block(1,y): 处理列 4-7
     * - Block(2,y): 处理列 8-11
     * - Block(3,y): 处理列 12-15
     */
    int ix = threadIdx.x + blockIdx.x * blockDim.x;

    /**
     * Y 坐标计算（垂直方向）
     *
     * 直接映射策略：
     * - iy = blockIdx.y: 每个线程块处理一行
     * - 简化了 Y 方向的索引计算
     * - 每行数据由一个或多个线程块并行处理
     *
     * 优势：
     * - 索引计算简单
     * - 适合行优先的访问模式
     * - 减少了 threadIdx.y 的复杂性
     */
    int iy = blockIdx.y;

    /**
     * 二维到一维索引转换
     *
     * 行优先存储映射：
     * idx = iy * nx + ix
     *
     * 访问模式分析：
     * - 同一行的相邻元素由相邻线程处理
     * - 有利于内存合并访问
     * - 缓存友好的访问模式
     */
    unsigned int idx = iy * nx + ix;

    /**
     * 边界检查
     *
     * 检查条件：
     * - ix < nx: 确保不超出矩阵宽度
     * - iy < ny: 确保不超出矩阵高度
     *
     * 边界情况：
     * - 当矩阵宽度不是线程块大小的整数倍时
     * - 最后一个线程块可能有部分线程超出边界
     */
    if (ix < nx && iy < ny)
    {
        /**
         * 执行矩阵加法
         * 只有有效范围内的线程参与计算
         */
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * 主函数 - 二维网格一维线程块演示
 */
int main(void)
{
    /**
     * 设置GPU设备
     */
    setGPU();

    /**
     * 矩阵参数配置（与其他版本保持一致）
     */
    int nx = 16;        // 矩阵宽度
    int ny = 8;         // 矩阵高度
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
     * 配置内核启动参数 - 关键区别
     *
     * 线程块配置：block(4, 1)
     * - blockDim.x = 4: X 方向 4 个线程
     * - blockDim.y = 1: Y 方向 1 个线程
     * - 每个线程块包含 4×1 = 4 个线程（一维排列）
     *
     * 网格配置：grid(4, 8)
     * - gridDim.x = (16+4-1)/4 = 4: X 方向 4 个线程块
     * - gridDim.y = 8: Y 方向 8 个线程块（每行一个）
     * - 网格包含 4×8 = 32 个线程块
     *
     * 线程分布特点：
     * - 每行由 4 个线程块处理
     * - 每个线程块处理连续的 4 个元素
     * - 总线程数：32 个线程块 × 4 个线程/块 = 128 个线程
     *
     * 与 grid2D_block2D 的对比：
     * grid2D_block2D: 8 个线程块 × 16 个线程/块 = 128 个线程
     * grid2D_block1D: 32 个线程块 × 4 个线程/块 = 128 个线程
     */
    dim3 block(4, 1);   // 一维线程块：4×1
    dim3 grid((nx + block.x - 1) / block.x, ny);  // 二维网格：4×8

    printf("线程配置：网格:<%d, %d>, 线程块:<%d, %d>\n", grid.x, grid.y, block.x, block.y);
    printf("线程组织模式：二维网格 + 一维线程块\n");
    printf("每行处理：%d 个线程块，每块 %d 个线程\n", grid.x, block.x);
    printf("总线程数：%d 个线程块 × %d 个线程/块 = %d 个线程\n",
           grid.x * grid.y, block.x * block.y, grid.x * grid.y * block.x * block.y);

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
 * 1. 混合维度组织的特点：
 *    - 网格：二维组织，适合二维数据结构
 *    - 线程块：一维组织，简化线程索引
 *    - 灵活组合不同维度的优势
 *
 * 2. 索引计算对比：
 *    grid2D_block2D:
 *    - ix = threadIdx.x + blockIdx.x * blockDim.x
 *    - iy = threadIdx.y + blockIdx.y * blockDim.y
 *
 *    grid2D_block1D:
 *    - ix = threadIdx.x + blockIdx.x * blockDim.x
 *    - iy = blockIdx.y
 *
 * 3. 性能特点分析：
 *    优势：
 *    - 索引计算更简单
 *    - 适合行优先访问模式
 *    - 较少的线程间通信
 *
 *    考虑：
 *    - 线程块数量增加
 *    - 可能影响缓存利用
 *    - 需要根据具体应用选择
 *
 * 4. 适用场景：
 *    - 行优先的矩阵操作
 *    - 图像的行扫描处理
 *    - 一维卷积操作
 *    - 简单的并行规约
 *
 * 5. 线程组织选择原则：
 *    - 数据访问模式
 *    - 计算复杂度
 *    - 硬件资源利用
 *    - 代码可维护性
 *
 * 6. 调试和优化：
 *    - 验证索引计算正确性
 *    - 检查内存访问模式
 *    - 分析线程利用率
 *    - 测试不同配置的性能
 */
