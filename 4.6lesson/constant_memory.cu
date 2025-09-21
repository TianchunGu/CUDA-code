/**
 * 课程 4.6: CUDA 常量内存使用详解
 * 文件名: constant_memory.cu
 * 作者: 权 双
 * 日期: 2023-12-26
 * 功能: 演示常量内存的声明、初始化和使用
 *
 * 常量内存特点：
 * 1. 只读访问 - 内核中不能修改常量内存的值
 * 2. 缓存优化 - 具有专用的常量内存缓存，访问速度快
 * 3. 广播效率 - 当所有线程访问相同地址时效率最高
 * 4. 大小限制 - 每个 GPU 的常量内存大小有限（通常 64KB）
 * 5. 生命周期 - 在整个程序执行期间有效
 */

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

/**
 * 常量内存声明
 *
 * __constant__ 修饰符声明常量内存变量：
 * - 必须在全局作用域声明
 * - 可以在声明时初始化
 * - 所有内核函数都可以访问
 */
__constant__ float c_data;          // 声明常量内存变量，未初始化
__constant__ float c_data2 = 6.6f;  // 声明并初始化常量内存变量

/**
 * 内核函数 1 - 演示常量内存的读取
 *
 * 本内核展示如何在 GPU 内核中访问常量内存
 */
__global__ void kernel_1(void)
{
    /**
     * 直接访问常量内存变量
     *
     * 特点：
     * - 所有线程都会读取相同的值
     * - 由于是广播访问，效率很高
     * - 常量内存缓存会加速访问
     */
    printf("Constant data c_data = %.2f.\n", c_data);
}

/**
 * 内核函数 2 - 预留用于扩展功能
 *
 * @param N: 线程数量限制
 *
 * 当前为空实现，可用于演示更复杂的常量内存使用场景
 */
__global__ void kernel_2(int N)
{
    int idx = threadIdx.x;
    if (idx < N)
    {
        // 这里可以添加使用常量内存的计算逻辑
        // 例如：float result = c_data * idx + c_data2;
    }
}

/**
 * 主函数 - 演示常量内存的完整使用流程
 */
int main(int argc, char **argv)
{
    /**
     * 第一步：获取和显示 GPU 设备信息
     * 了解当前使用的 GPU 设备特性
     */
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    /**
     * 第二步：设置常量内存的值
     *
     * cudaMemcpyToSymbol() 函数用法：
     * - 将主机内存的值复制到常量内存
     * - 第一个参数：常量内存变量名（符号）
     * - 第二个参数：主机内存地址
     * - 第三个参数：复制的字节数
     */
    float h_data = 8.8f;  // 主机端的数据
    CUDA_CHECK(cudaMemcpyToSymbol(c_data, &h_data, sizeof(float)));

    /**
     * 第三步：启动内核函数
     *
     * 线程配置：
     * - 1 个线程块，每个块 1 个线程
     * - 用于演示常量内存的基本访问
     */
    dim3 block(1);
    dim3 grid(1);
    kernel_1<<<grid, block>>>();

    /**
     * 第四步：同步等待内核完成
     * 确保内核执行完毕，printf 输出可以显示
     */
    CUDA_CHECK(cudaDeviceSynchronize());

    /**
     * 第五步：从常量内存读取数据到主机
     *
     * cudaMemcpyFromSymbol() 函数用法：
     * - 将常量内存的值复制到主机内存
     * - 第一个参数：主机内存地址
     * - 第二个参数：常量内存变量名（符号）
     * - 第三个参数：复制的字节数
     */
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_data, c_data2, sizeof(float)));
    printf("Constant data h_data = %.2f.\n", h_data);

    /**
     * 第六步：清理设备资源
     */
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. 常量内存的优势：
 *    - 访问延迟低，有专用缓存
 *    - 适合所有线程访问相同数据的场景
 *    - 不需要显式分配和释放
 *
 * 2. 使用场景：
 *    - 数学常数（π、e 等）
 *    - 查找表和系数
 *    - 配置参数
 *    - 不变的输入数据
 *
 * 3. 限制和注意事项：
 *    - 大小限制（通常 64KB）
 *    - 只读访问
 *    - 必须在编译时确定大小
 *    - 当线程访问不同地址时性能下降
 *
 * 4. 与其他内存类型的对比：
 *    - 全局内存：大容量，但访问较慢
 *    - 共享内存：快速，但需要手动管理
 *    - 常量内存：中等容量，自动缓存，只读
 *
 * 5. 程序输出预期：
 *    - "Constant data c_data = 8.80"（来自 kernel_1）
 *    - "Constant data h_data = 6.60"（来自 c_data2）
 */