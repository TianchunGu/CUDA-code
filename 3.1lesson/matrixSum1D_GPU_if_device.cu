/**
 * 课程 3.1: 增强版 GPU 矩阵加法程序
 * 文件名: matrixSum1D_GPU_if_device.cu
 * 作者: 权 双
 * 日期: 2023-08-04
 * 功能: 矩阵求和程序，通过调用核函数在 GPU 执行
 *       新增功能：1、if 判断条件边界检查；2、调用设备函数
 *
 * 本程序相比基础版本的改进：
 * 1. 添加了边界检查，防止数组越界访问
 * 2. 使用了设备函数 (__device__) 进行代码模块化
 * 3. 演示了不整除情况下的网格大小计算
 * 4. 更完善的错误处理和程序结构
 */

#include <stdio.h>
#include "../tools/common.cuh"

/**
 * 设备函数 (__device__) - GPU 上的辅助函数
 *
 * @param x: 第一个操作数
 * @param y: 第二个操作数
 * @return: 两数之和
 *
 * 设备函数特点：
 * - 只能在 GPU 上运行，由 GPU 内核或其他设备函数调用
 * - 不能从主机 (CPU) 代码直接调用
 * - 支持函数重载和递归 (计算能力 >= 2.0)
 * - 编译时内联优化，性能开销小
 */
__device__ float add(const float x, const float y)
{
    return x + y;
}

/**
 * GPU 内核函数 - 改进版并行向量加法
 *
 * @param A: 输入向量 A (GPU 内存)
 * @param B: 输入向量 B (GPU 内存)
 * @param C: 输出向量 C (GPU 内存)
 * @param N: 向量元素数量
 *
 * 关键改进：
 * 1. 添加边界检查 (id >= N)
 * 2. 使用设备函数进行计算
 * 3. 支持任意大小的数组 (不必是线程数的整数倍)
 */
__global__ void addFromGPU(float *A, float *B, float *C, const int N)
{
    // 计算当前线程的全局索引
    const int bid = blockIdx.x;      // 线程块索引
    const int tid = threadIdx.x;     // 线程在块内的索引
    const int id = tid + bid * blockDim.x;

    /**
     * 关键的边界检查
     *
     * 为什么需要边界检查？
     * - 当数组大小不是线程数的整数倍时，部分线程的索引会超出数组范围
     * - 例如：数组大小 513，线程块大小 32，需要 17 个线程块 (32*17=544)
     * - 最后 31 个线程 (索引 513-543) 需要被忽略，避免越界访问
     */
    if (id >= N) return;

    /**
     * 调用设备函数进行计算
     * 这里演示了设备函数的使用，实际上可以直接写 C[id] = A[id] + B[id]
     * 但使用设备函数有助于代码模块化和复用
     */
    C[id] = add(A[id], B[id]);
}

/**
 * 数据初始化函数 (与基础版本相同)
 */
void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

/**
 * 主函数 - 演示改进版 GPU 计算流程
 */
int main(void)
{
    /**
     * 第一步：设置 GPU 设备
     */
    setGPU();

    /**
     * 第二步：分配内存并初始化
     *
     * 注意：这里使用 513 个元素，故意选择一个不能被 32 整除的数
     * 用于演示边界检查的重要性
     */
    int iElemCount = 513;                                    // 设置元素数量（故意不整除）
    size_t stBytesCount = iElemCount * sizeof(float);       // 计算字节数

    /**
     * 子步骤 2.1：分配主机内存
     */
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);

    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        memset(fpHost_A, 0, stBytesCount);  // 主机内存初始化为 0
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    /**
     * 子步骤 2.2：分配设备内存
     */
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float**)&fpDevice_A, stBytesCount);
    cudaMalloc((float**)&fpDevice_B, stBytesCount);
    cudaMalloc((float**)&fpDevice_C, stBytesCount);

    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
    {
        cudaMemset(fpDevice_A, 0, stBytesCount);  // 设备内存初始化为 0
        cudaMemset(fpDevice_B, 0, stBytesCount);
        cudaMemset(fpDevice_C, 0, stBytesCount);
    }
    else
    {
        printf("fail to allocate memory\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

    /**
     * 第三步：初始化主机数据
     */
    srand(666);  // 设置随机种子
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);

    /**
     * 第四步：数据传输到设备
     */
    cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);

    /**
     * 第五步：计算网格配置并启动内核
     *
     * 关键改进：向上取整的网格大小计算
     *
     * 旧方法（可能导致数据丢失）：
     * dim3 grid(iElemCount / 32);  // 513 / 32 = 16（整数除法截断）
     *
     * 新方法（向上取整）：
     * dim3 grid((iElemCount + block.x - 1) / 32);  // (513 + 32 - 1) / 32 = 17
     *
     * 计算原理：
     * - 当 iElemCount = 513, block.x = 32
     * - (513 + 32 - 1) / 32 = 544 / 32 = 17
     * - 确保所有 513 个元素都有对应的线程处理
     */
    dim3 block(32);
    dim3 grid((iElemCount + block.x - 1) / 32);  // 向上取整：17 个线程块

    printf("网格配置：%d 个线程块，每个线程块 %d 个线程，总线程数：%d\n",
           grid.x, block.x, grid.x * block.x);
    printf("数组元素数量：%d，多余线程数：%d\n",
           iElemCount, grid.x * block.x - iElemCount);

    // 启动内核函数
    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);

    /**
     * 同步等待（这里取消了注释，确保计算完成）
     */
    cudaDeviceSynchronize();

    /**
     * 第六步：将结果传回主机
     */
    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);

    /**
     * 第七步：验证结果
     */
    printf("\n验证计算结果（前 10 个元素）：\n");
    for (int i = 0; i < 10; i++)
    {
        printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n",
               i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    /**
     * 第八步：释放内存资源
     */
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    cudaDeviceReset();
    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. 边界检查的重要性：
 *    - 防止数组越界访问
 *    - 支持任意大小的数据集
 *    - 避免内存错误和未定义行为
 *
 * 2. 设备函数的使用：
 *    - 代码模块化和复用
 *    - 编译器内联优化
 *    - 不增加函数调用开销
 *
 * 3. 网格大小的正确计算：
 *    - 向上取整公式：(N + blockSize - 1) / blockSize
 *    - 确保所有数据元素都有对应线程
 *    - 配合边界检查使用
 *
 * 4. 实际应用考虑：
 *    - 数据大小通常不是线程块大小的整数倍
 *    - 边界检查是 CUDA 编程的标准实践
 *    - 设备函数有助于复杂算法的实现
 *
 * 5. 性能影响：
 *    - 边界检查的分支可能影响性能
 *    - 设备函数通常被编译器内联
 *    - 多余的线程会浪费计算资源
 */

