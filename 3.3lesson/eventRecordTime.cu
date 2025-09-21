/**
 * 课程 3.3: CUDA 事件计时详解
 * 文件名: eventRecordTime.cu
 * 作者: 权 双
 * 日期: 2023-08-05
 * 功能: 核函数记时
 *
 * 本程序演示：
 * 1. CUDA 事件 (cudaEvent_t) 的创建和使用
 * 2. 高精度 GPU 内核执行时间测量
 * 3. 多次运行求平均值以获得稳定的性能数据
 * 4. GPU 计时的最佳实践和注意事项
 *
 * CUDA 事件计时的优势：
 * - 高精度：微秒级精度
 * - GPU 原生：直接在 GPU 上记录时间
 * - 异步友好：不影响 GPU-CPU 重叠执行
 * - 硬件支持：利用 GPU 硬件计时器
 */

#include <stdio.h>
#include "../tools/common.cuh"

/**
 * 重复测试次数
 * 多次运行可以减少测量误差，获得更稳定的性能数据
 */
#define NUM_REPEATS 10

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

    // 边界检查
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
 * 主函数 - 演示 CUDA 事件计时的完整流程
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
     * 子步骤 2.2：分配设备内存（使用错误检查）
     */
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    ErrorCheck(cudaMalloc((float**)&fpDevice_A, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((float**)&fpDevice_B, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((float**)&fpDevice_C, stBytesCount), __FILE__, __LINE__);

    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
    {
        ErrorCheck(cudaMemset(fpDevice_A, 0, stBytesCount), __FILE__, __LINE__);
        ErrorCheck(cudaMemset(fpDevice_B, 0, stBytesCount), __FILE__, __LINE__);
        ErrorCheck(cudaMemset(fpDevice_C, 0, stBytesCount), __FILE__, __LINE__);
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
     * 第四步：数据传输到设备（使用错误检查）
     */
    ErrorCheck(cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    /**
     * 第五步：配置内核启动参数
     */
    dim3 block(32);
    dim3 grid((iElemCount + block.x - 1) / 32);

    printf("开始性能测试：\n");
    printf("数组大小：%d 个元素\n", iElemCount);
    printf("网格配置：%d 个线程块，每个线程块 %d 个线程\n", grid.x, block.x);
    printf("测试次数：%d 次（第一次为预热，不计入平均值）\n\n", NUM_REPEATS + 1);

    /**
     * 第六步：CUDA 事件计时循环
     *
     * 关键概念：
     * 1. 预热 (Warm-up)：第一次运行用于预热 GPU，不计入平均时间
     * 2. 多次测量：减少测量误差，获得稳定的性能数据
     * 3. 事件同步：确保准确的时间测量
     */
    float t_sum = 0;  // 总时间累加器

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        /**
         * 子步骤 6.1：创建 CUDA 事件
         *
         * CUDA 事件特点：
         * - 轻量级对象，创建开销小
         * - 硬件支持的高精度计时
         * - 可以异步记录时间点
         */
        cudaEvent_t start, stop;
        ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
        ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);

        /**
         * 子步骤 6.2：记录开始时间
         *
         * cudaEventRecord() 的作用：
         * - 在当前 CUDA 流中插入一个时间标记
         * - 异步操作，不阻塞 CPU
         * - 记录的是 GPU 时间，不是 CPU 时间
         */
        ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);

        /**
         * 特殊的事件查询
         *
         * cudaEventQuery() 的作用：
         * - 检查事件是否已完成
         * - 这里用于确保事件已正确记录
         * - 注释说明：此处不可用错误检测函数
         *   因为 cudaEventQuery() 在事件未完成时返回 cudaErrorNotReady
         *   这不是真正的错误，而是正常的状态查询结果
         */
        cudaEventQuery(start);  // 此处不可用错误检测函数

        /**
         * 子步骤 6.3：执行内核函数
         *
         * 重要：内核启动是异步的
         * 时间测量包含内核的完整执行时间
         */
        addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);

        /**
         * 子步骤 6.4：记录结束时间
         */
        ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);

        /**
         * 子步骤 6.5：等待事件完成并计算时间
         *
         * cudaEventSynchronize() 的作用：
         * - 阻塞 CPU 直到事件完成
         * - 确保内核执行完毕
         * - 类似于 cudaDeviceSynchronize()，但更精确
         */
        ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);

        /**
         * 计算经过的时间
         *
         * cudaEventElapsedTime() 的特点：
         * - 返回毫秒为单位的时间差
         * - 高精度（通常亚毫秒级）
         * - 基于 GPU 硬件计时器
         */
        float elapsed_time;
        ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);

        // 显示每次测试的时间（可选）
        printf("第 %2d 次测试：%.4f ms", repeat + 1, elapsed_time);
        if (repeat == 0) {
            printf(" (预热，不计入平均值)\n");
        } else {
            printf("\n");
        }

        /**
         * 累加时间（跳过第一次预热）
         */
        if (repeat > 0)
        {
            t_sum += elapsed_time;
        }

        /**
         * 子步骤 6.6：销毁事件对象
         *
         * 重要：每次循环都要销毁事件，避免资源泄露
         */
        ErrorCheck(cudaEventDestroy(start), __FILE__, __LINE__);
        ErrorCheck(cudaEventDestroy(stop), __FILE__, __LINE__);
    }

    /**
     * 第七步：计算并显示平均时间
     */
    const float t_ave = t_sum / NUM_REPEATS;
    printf("\n性能测试结果：\n");
    printf("平均执行时间：%.4f ms\n", t_ave);
    printf("数据传输量：%.2f MB\n", (stBytesCount * 3) / 1024.0 / 1024.0);  // 3个数组的总大小
    printf("有效带宽：%.2f GB/s\n", (stBytesCount * 3) / (t_ave / 1000.0) / 1024.0 / 1024.0 / 1024.0);

    /**
     * 第八步：将结果传回主机并验证
     */
    ErrorCheck(cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    printf("\n验证计算结果（前 5 个元素）：\n");
    for (int i = 0; i < 5; i++)
    {
        printf("A[%d]=%.2f + B[%d]=%.2f = C[%d]=%.2f\n",
               i, fpHost_A[i], i, fpHost_B[i], i, fpHost_C[i]);
    }

    /**
     * 第九步：释放内存资源
     */
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    ErrorCheck(cudaFree(fpDevice_A), __FILE__, __LINE__);
    ErrorCheck(cudaFree(fpDevice_B), __FILE__, __LINE__);
    ErrorCheck(cudaFree(fpDevice_C), __FILE__, __LINE__);

    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);
    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. CUDA 事件计时的核心步骤：
 *    a) 创建事件：cudaEventCreate()
 *    b) 记录开始：cudaEventRecord(start)
 *    c) 执行内核：kernel<<<>>>()
 *    d) 记录结束：cudaEventRecord(stop)
 *    e) 等待完成：cudaEventSynchronize(stop)
 *    f) 计算时间：cudaEventElapsedTime()
 *    g) 销毁事件：cudaEventDestroy()
 *
 * 2. 与其他计时方法的对比：
 *    - CPU 计时（time(), clock()）：可能不准确，受 CPU-GPU 异步影响
 *    - cudaDeviceSynchronize() + CPU计时：准确但阻塞，影响性能
 *    - CUDA 事件：GPU 原生，高精度，异步友好
 *
 * 3. 性能测试最佳实践：
 *    - 预热：第一次运行用于预热，消除冷启动影响
 *    - 多次测量：减少噪音，获得稳定结果
 *    - 统计分析：计算平均值、标准差等
 *    - 环境控制：固定频率、关闭节能等
 *
 * 4. 注意事项：
 *    - 事件记录是异步的，需要同步等待
 *    - cudaEventQuery() 的返回值有特殊含义
 *    - 事件对象需要手动销毁，避免资源泄露
 *    - 时间单位是毫秒 (ms)
 *
 * 5. 应用场景：
 *    - 内核性能分析
 *    - 算法优化验证
 *    - 性能基准测试
 *    - 瓶颈识别
 *
 * 6. 扩展应用：
 *    - 可以测量多个内核的总时间
 *    - 可以测量内存传输时间
 *    - 可以在多个流中使用
 *    - 可以用于异步操作的时序分析
 */

