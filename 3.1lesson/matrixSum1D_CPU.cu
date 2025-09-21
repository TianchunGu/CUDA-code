/**
 * 课程 3.1: CPU 矩阵加法程序
 * 文件名: matrixSum1D_CPU.cu
 * 作者: 权 双
 * 日期: 2023-08-04
 * 功能: 矩阵求和程序，通过 CPU 计算
 *
 * 本程序演示：
 * 1. CPU 上的向量/矩阵加法实现
 * 2. 动态内存分配和管理
 * 3. 数据初始化和随机数生成
 * 4. 为后续 GPU 版本提供性能对比基准
 */

#include <stdio.h>
#include "../tools/common.cuh"

/**
 * CPU 版本的向量加法函数
 *
 * @param A: 输入向量 A
 * @param B: 输入向量 B
 * @param C: 输出向量 C (C = A + B)
 * @param N: 向量元素数量
 *
 * 特点：
 * - 顺序执行，单线程处理
 * - 时间复杂度 O(N)
 * - 适合小规模数据或作为正确性验证的参考
 */
void addFromCPU(float *A, float *B, float *C, const int N)
{
    // 使用 for 循环顺序处理每个元素
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];  // 逐个元素相加
    }
}

/**
 * 数据初始化函数
 *
 * @param addr: 要初始化的内存地址
 * @param elemCount: 元素数量
 *
 * 功能：
 * - 用随机数填充数组
 * - 随机数范围：0.0 ~ 25.5 (通过位运算和除法实现)
 */
void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        // rand() & 0xFF: 生成 0-255 的随机整数
        // 除以 10.0f: 得到 0.0-25.5 的浮点数
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

/**
 * 主函数 - 演示完整的 CPU 矩阵加法流程
 */
int main(void)
{
    /**
     * 第一步：分配主机内存并初始化
     */
    int iElemCount = 512;                                    // 设置元素数量（向量长度）
    size_t stBytesCount = iElemCount * sizeof(float);        // 计算所需字节数

    // 分配三个浮点数组：A、B（输入），C（输出）
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);

    /**
     * 内存分配检查
     * 确保所有内存分配都成功，否则程序无法正常运行
     */
    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        // 将分配的内存初始化为 0
        memset(fpHost_A, 0, stBytesCount);
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    /**
     * 第二步：初始化主机数据
     */
    srand(666);  // 设置随机种子，确保结果可重现
    initialData(fpHost_A, iElemCount);  // 用随机数填充向量 A
    initialData(fpHost_B, iElemCount);  // 用随机数填充向量 B

    /**
     * 第三步：执行 CPU 计算
     * 这是核心计算步骤，在 CPU 上完成向量加法
     */
    addFromCPU(fpHost_A, fpHost_B, fpHost_C, iElemCount);

    /**
     * 第四步：显示部分结果
     * 打印前 10 个元素的计算结果，用于验证正确性
     */
    for (int i = 0; i < 10; i++)
    {
        printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n",
               i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    /**
     * 第五步：释放内存资源
     * 防止内存泄漏，良好的编程习惯
     */
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);

    return 0;
}

