/**
 * 课程 3.1: GPU 矩阵加法程序
 * 文件名: matrixSum1D_GPU.cu
 * 作者: 权 双
 * 日期: 2023-08-04
 * 功能: 矩阵求和程序，通过调用核函数在 GPU 执行
 *
 * 本程序演示：
 * 1. GPU 内核函数的编写和调用
 * 2. CPU-GPU 内存管理和数据传输
 * 3. CUDA 编程的完整工作流程
 * 4. 并行计算的基本实现
 */

#include <stdio.h>
#include "../tools/common.cuh"

/**
 * GPU 内核函数 - 并行向量加法
 *
 * @param A: 输入向量 A (GPU 内存)
 * @param B: 输入向量 B (GPU 内存)
 * @param C: 输出向量 C (GPU 内存)
 * @param N: 向量元素数量
 *
 * 核心概念：
 * - 每个线程处理一个数组元素
 * - 线程并行执行，大大提高计算效率
 * - 通过线程索引确定处理的数据位置
 */
__global__ void addFromGPU(float *A, float *B, float *C, const int N)
{
    // 获取当前线程的标识
    const int bid = blockIdx.x;      // 线程块索引
    const int tid = threadIdx.x;     // 线程在块内的索引

    /**
     * 计算全局线程索引
     * 这是 CUDA 编程的核心公式，将二维线程组织映射为一维数组索引
     */
    const int id = tid + bid * blockDim.x;

    /**
     * 并行计算：每个线程处理一个元素
     * 注意：这里没有边界检查，假设线程数 == 数组大小
     * 在实际应用中应添加边界检查：if (id < N)
     */
    C[id] = A[id] + B[id];
}

/**
 * 数据初始化函数（与 CPU 版本相同）
 *
 * @param addr: 要初始化的内存地址
 * @param elemCount: 元素数量
 */
void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        // 生成 0.0-25.5 范围的随机浮点数
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

/**
 * 主函数 - 演示完整的 GPU 计算流程
 */
int main(void)
{
    /**
     * 第一步：设置 GPU 设备
     * 使用 common.cuh 中的 setGPU() 函数初始化 GPU
     */
    setGPU();

    /**
     * 第二步：分配主机内存和设备内存，并初始化
     */
    int iElemCount = 512;                               // 设置元素数量
    size_t stBytesCount = iElemCount * sizeof(float);   // 计算字节数

    /**
     * 子步骤 2.1：分配主机（CPU）内存
     */
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);

    // 主机内存分配检查
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
     * 子步骤 2.2：分配设备（GPU）内存
     * cudaMalloc() 类似于 malloc()，但分配的是 GPU 内存
     */
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float**)&fpDevice_A, stBytesCount);
    cudaMalloc((float**)&fpDevice_B, stBytesCount);
    cudaMalloc((float**)&fpDevice_C, stBytesCount);

    // 设备内存分配检查
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
    srand(666);  // 设置随机种子，确保与 CPU 版本结果一致
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);

    /**
     * 第四步：数据从主机复制到设备
     * 这是 CUDA 编程的关键步骤：CPU -> GPU 数据传输
     */
    cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);

    /**
     * 第五步：调用核函数在设备中进行计算
     *
     * 线程配置解析：
     * - dim3 block(32): 每个线程块包含 32 个线程
     * - dim3 grid(iElemCount / 32): 网格包含 512/32 = 16 个线程块
     * - 总线程数: 16 × 32 = 512，正好等于数组大小
     *
     * 这种配置确保每个数组元素由一个线程处理
     */
    dim3 block(32);
    dim3 grid(iElemCount / 32);

    // 启动内核函数
    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);
    // 注意：cudaDeviceSynchronize() 被注释掉了
    // 在实际应用中，建议取消注释以确保计算完成

    /**
     * 第六步：将计算结果从设备传回主机
     * GPU -> CPU 数据传输
     */
    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);

    /**
     * 第七步：显示部分结果
     * 打印前 10 个元素验证计算正确性
     */
    for (int i = 0; i < 10; i++)
    {
        printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n",
               i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    /**
     * 第八步：释放内存资源
     * 需要分别释放主机内存和设备内存
     */
    // 释放主机内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);

    // 释放设备内存
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    /**
     * 重置 CUDA 设备
     * 清理 CUDA 运行时状态，释放设备资源
     */
    cudaDeviceReset();
    return 0;
}

