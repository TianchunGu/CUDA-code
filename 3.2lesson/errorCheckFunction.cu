/**
 * 课程 3.2: CUDA 错误检查函数使用示例
 * 文件名: errorCheckFunction.cu
 * 作者: 权 双
 * 日期: 2023-08-04
 * 功能: 错误检查函数的使用例子
 *
 * 本程序演示：
 * 1. 如何使用 ErrorCheck() 函数进行 CUDA 错误检查
 * 2. CUDA API 调用的正确错误处理流程
 * 3. 调试和开发阶段的最佳实践
 * 4. 内存操作中的错误检测
 */

#include <stdio.h>
#include "../tools/common.cuh"

/**
 * 主函数 - 演示 CUDA 错误检查的使用
 *
 * 注意：本程序故意包含一个错误，用于演示错误检查功能
 */
int main(void)
{
    /**
     * 第一步：分配主机内存并初始化
     * 这部分使用标准 C 库函数，不需要 CUDA 错误检查
     */
    float *fpHost_A;
    fpHost_A = (float *)malloc(4);  // 分配 4 字节内存
    memset(fpHost_A, 0, 4);         // 主机内存初始化为 0

    /**
     * 第二步：分配设备内存并使用错误检查
     *
     * ErrorCheck() 函数的作用：
     * - 检查 CUDA API 调用是否成功
     * - 如果出错，打印错误信息和位置
     * - 提供 __FILE__ 和 __LINE__ 宏用于定位错误
     */
    float *fpDevice_A;
    cudaError_t error = ErrorCheck(cudaMalloc((float**)&fpDevice_A, 4), __FILE__, __LINE__);
    cudaMemset(fpDevice_A, 0, 4);   // 设备内存初始化为 0

    /**
     * 第三步：数据传输 - 包含故意的错误
     *
     * 错误说明：
     * 这里使用了 cudaMemcpyDeviceToHost，但实际上应该是 cudaMemcpyHostToDevice
     * 因为我们是从主机复制到设备，不是从设备复制到主机
     *
     * 正确的应该是：
     * ErrorCheck(cudaMemcpy(fpDevice_A, fpHost_A, 4, cudaMemcpyHostToDevice), __FILE__, __LINE__);
     */
    ErrorCheck(cudaMemcpy(fpDevice_A, fpHost_A, 4, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    /**
     * 第四步：释放内存资源
     * 展示如何对内存释放操作也进行错误检查
     */
    free(fpHost_A);  // 释放主机内存（标准 C 函数，无需错误检查）

    // 释放设备内存，使用错误检查
    ErrorCheck(cudaFree(fpDevice_A), __FILE__, __LINE__);

    /**
     * 第五步：设备重置
     * 在程序结束时重置 CUDA 设备，也应该进行错误检查
     */
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 0;
}

/**
 * 学习要点：
 *
 * 1. ErrorCheck() 函数的重要性：
 *    - 帮助快速定位 CUDA 程序中的错误
 *    - 提供详细的错误信息和代码位置
 *    - 在开发和调试阶段非常有用
 *
 * 2. 应该检查的 CUDA API：
 *    - 内存分配：cudaMalloc(), cudaFree()
 *    - 内存传输：cudaMemcpy(), cudaMemset()
 *    - 设备管理：cudaSetDevice(), cudaDeviceReset()
 *    - 内核启动后：cudaGetLastError(), cudaDeviceSynchronize()
 *
 * 3. 最佳实践：
 *    - 在开发阶段，对所有 CUDA API 调用进行错误检查
 *    - 在发布版本中，可以考虑移除一些错误检查以提高性能
 *    - 使用宏定义可以方便地开启/关闭错误检查
 *
 * 4. 本程序的预期行为：
 *    - 运行时会在第三步报告内存拷贝方向错误
 *    - ErrorCheck() 函数会显示错误信息和发生错误的文件行号
 */

