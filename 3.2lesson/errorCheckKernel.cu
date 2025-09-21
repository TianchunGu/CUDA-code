/**
 * 课程 3.2: CUDA 内核函数错误检查详解
 * 文件名: errorCheckKernel.cu
 * 作者: 权 双
 * 日期: 2023-08-05
 * 功能: 捕捉核函数错误
 *
 * 本程序演示：
 * 1. 如何检测内核启动时的错误
 * 2. 如何检测内核执行过程中的错误
 * 3. cudaGetLastError() 和 cudaDeviceSynchronize() 的使用
 * 4. 故意创建错误场景来演示错误检测机制
 */

#include <stdio.h>
#include "../tools/common.cuh"

/**
 * 设备函数 - 简单的加法运算
 */
__device__ float add(const float x, const float y)
{
    return x + y;
}

/**
 * GPU 内核函数 - 并行向量加法
 *
 * @param A: 输入向量 A
 * @param B: 输入向量 B
 * @param C: 输出向量 C
 * @param N: 向量元素数量
 */
__global__ void addFromGPU(float *A, float *B, float *C, const int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;

    // 边界检查，防止数组越界
    if (id >= N) return;

    C[id] = add(A[id], B[id]);
}

/**
 * 数据初始化函数
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
 * 主函数 - 演示内核错误检查的完整流程
 */
int main(void)
{
    /**
     * 第一步：设置 GPU 设备
     */
    setGPU();

    /**
     * 第二步：分配内存并初始化
     */
    int iElemCount = 4096;                                   // 设置元素数量
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
     * 第五步：配置内核启动参数并进行错误检查
     *
     * 注意：这里故意使用了很大的线程块大小 (2048)
     * 这可能会导致错误，因为：
     * 1. 大多数 GPU 的最大线程块大小限制在 1024
     * 2. 这是一个故意的错误配置，用于演示错误检测
     */
    dim3 block(2048);  // 故意设置过大的线程块大小
    dim3 grid((iElemCount + block.x - 1) / 2048);

    printf("尝试启动内核：网格大小 = %d, 线程块大小 = %d\n", grid.x, block.x);

    /**
     * 启动内核函数
     *
     * 重要：内核启动是异步的，即使参数错误，这行代码也可能不会立即报错
     * 错误通常在后续的 CUDA API 调用中被检测到
     */
    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);

    /**
     * 第六步：内核错误检查 - 关键步骤
     *
     * cudaGetLastError() 的作用：
     * - 获取最近一次 CUDA API 调用产生的错误
     * - 检查内核启动时的参数错误
     * - 不等待内核执行完成
     * - 调用后会清除错误状态
     */
    ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);

    /**
     * cudaDeviceSynchronize() 的作用：
     * - 等待所有之前启动的内核执行完成
     * - 检查内核执行过程中的错误
     * - 阻塞 CPU 线程直到 GPU 操作完成
     * - 如果内核执行过程中出错，这里会检测到
     */
    ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    printf("内核执行完成，未发现错误\n");

    /**
     * 第七步：将结果传回主机
     */
    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);

    /**
     * 第八步：验证部分结果
     */
    printf("\n验证计算结果（前 10 个元素）：\n");
    for (int i = 0; i < 10; i++)
    {
        printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n",
               i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    /**
     * 第九步：释放内存资源
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
 * 1. 内核错误检查的两个阶段：
 *    a) 启动时错误检查：cudaGetLastError()
 *       - 检查内核启动参数是否有效
 *       - 检查资源是否足够
 *       - 检查内核函数是否存在
 *
 *    b) 执行时错误检查：cudaDeviceSynchronize()
 *       - 等待内核执行完成
 *       - 检查执行过程中的错误
 *       - 检查内存访问错误
 *
 * 2. 常见的内核启动错误：
 *    - 线程块大小超过硬件限制
 *    - 网格大小超过硬件限制
 *    - 共享内存使用超过限制
 *    - 寄存器使用超过限制
 *
 * 3. 常见的内核执行错误：
 *    - 数组越界访问
 *    - 空指针访问
 *    - 除零错误
 *    - 非法内存访问
 *
 * 4. 错误检查的最佳实践：
 *    - 在开发阶段对所有内核启动进行错误检查
 *    - 使用 cudaGetLastError() 检查启动参数
 *    - 使用 cudaDeviceSynchronize() 检查执行结果
 *    - 在关键代码段后添加错误检查
 *
 * 5. 性能考虑：
 *    - cudaDeviceSynchronize() 会阻塞 CPU
 *    - 在生产环境中可以考虑减少同步调用
 *    - 错误检查有助于调试但会影响性能
 *
 * 6. 预期行为：
 *    - 如果 GPU 支持 2048 线程/块：程序正常运行
 *    - 如果 GPU 不支持：ErrorCheck() 会报告错误并退出
 *    - 错误信息会显示具体的错误代码和描述
 */

