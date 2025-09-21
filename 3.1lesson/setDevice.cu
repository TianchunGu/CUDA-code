/**
 * 课程 3.1: GPU 设备检测和设置
 *
 * 本程序演示：
 * 1. 如何检测系统中可用的 GPU 数量
 * 2. 如何设置特定的 GPU 设备用于计算
 * 3. CUDA 错误处理的基本方法
 * 4. GPU 设备管理的基础概念
 */

#include <stdio.h>

/**
 * 主函数 - 演示 GPU 设备的检测和设置
 */
int main(void)
{
    /**
     * 第一步：检测系统中可用的 GPU 设备数量
     *
     * cudaGetDeviceCount() 函数用于获取系统中 CUDA 兼容的 GPU 数量
     * 这是编写 CUDA 程序时的重要步骤，确保系统有可用的 GPU
     */
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);

    /**
     * 错误检查：验证 GPU 检测是否成功
     *
     * 可能的失败原因：
     * - 系统没有安装 NVIDIA GPU
     * - GPU 驱动程序未正确安装
     * - CUDA 运行时环境问题
     */
    if (error != cudaSuccess || iDeviceCount == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    else
    {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }

    /**
     * 第二步：设置要使用的 GPU 设备
     *
     * 在多 GPU 系统中，需要指定使用哪个 GPU 进行计算
     * GPU 设备编号从 0 开始，到 (iDeviceCount - 1)
     */
    int iDev = 0;  // 选择第一个 GPU 设备（设备 0）
    error = cudaSetDevice(iDev);

    /**
     * 错误检查：验证设备设置是否成功
     *
     * 可能的失败原因：
     * - 指定的设备编号无效
     * - 设备已被其他进程占用
     * - 设备处于独占模式
     */
    if (error != cudaSuccess)
    {
        printf("fail to set GPU 0 for computing.\n");
        exit(-1);
    }
    else
    {
        printf("set GPU 0 for computing.\n");
    }

    /**
     * 程序成功完成
     *
     * 此时 GPU 0 已被设置为当前设备，后续的 CUDA 操作
     * （如内存分配、内核启动等）将在这个设备上执行
     */
    return 0;
}

