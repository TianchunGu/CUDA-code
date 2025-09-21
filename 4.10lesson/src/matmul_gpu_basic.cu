/**
 * 课程 4.10: 高级 CUDA 项目结构 - GPU 归约算法实现
 * 文件名: matmul_gpu_basic.cu
 * 作者: 权 双
 * 日期: 2023-12-31
 * 功能: 并行归约算法的 GPU 实现
 *
 * 本文件实现两种 GPU 归约算法：
 * 1. 带分支版本 - 演示线程束分支的影响
 * 2. 无分支版本 - 优化版本，提高执行效率
 *
 * 归约算法的核心思想：
 * - 树形归约：每一步将数据量减半
 * - 并行处理：多个线程同时参与计算
 * - 同步协调：确保数据一致性
 *
 * 性能优化要点：
 * - 线程束分支是 GPU 性能的主要瓶颈
 * - 合并内存访问提高带宽利用率
 * - 减少同步次数提高效率
 */

#include "cuda_runtime.h"
#include "cuda.h"
#include "stdio.h"
#include "utils.hpp"

/**
 * GPU 内核函数 - 带分支的归约算法
 *
 * @param d_idata: 输入数据（设备全局内存）
 * @param d_odata: 输出数据（每个线程块一个结果）
 * @param size: 数据总大小
 *
 * 算法流程：
 * 1. 每个线程负责一个数据元素
 * 2. 通过步长(stride)递增进行树形归约
 * 3. 使用模运算判断线程是否参与
 * 4. 线程0将最终结果写入输出数组
 *
 * 性能问题：
 * - if ((tid % (2 * stride)) == 0) 导致线程束分支
 * - 随着stride增大，越来越多的线程空闲
 * - SIMD 执行效率低下
 */
__global__ void ReduceNeighboredWithDivergence(float *d_idata, float *d_odata, int size){
    /**
     * 线程索引计算
     */
    unsigned int tid = threadIdx.x;     // 线程在块内的索引
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程索引

    /**
     * 将全局内存指针转换为当前线程块的局部指针
     * 每个线程块处理 blockDim.x 个连续元素
     */
    float *idata = d_idata + blockIdx.x * blockDim.x;

    /**
     * 边界检查 - 防止越界访问
     */
    if (idx >= size) return;

    /**
     * 归约主循环 - 在全局内存中原位归约
     *
     * 步长变化：1, 2, 4, 8, 16, ...
     * 每次迭代后活跃线程数减半
     *
     * 示例（blockDim.x = 8）：
     * 迭代0: stride=1, 线程0,2,4,6参与
     * 迭代1: stride=2, 线程0,4参与
     * 迭代2: stride=4, 线程0参与
     */
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        /**
         * 线程束分支点！
         * 使用模运算判断线程是否参与计算
         * 这导致同一个 warp 内的线程执行不同路径
         */
        if ((tid % (2 * stride)) == 0)
        {
            // 将相邻的两个数据相加
            idata[tid] += idata[tid + stride];
        }

        /**
         * 线程同步 - 确保所有线程完成当前步骤
         * 在进入下一步之前必须同步
         */
        __syncthreads();
    }

    /**
     * 线程0将最终结果写入输出数组
     * idata[0] 现在包含该线程块的归约结果
     */
    if (tid == 0) d_odata[blockIdx.x] = idata[0];
}

/**
 * GPU 内核函数 - 无分支优化的归约算法
 *
 * @param d_idata: 输入数据（设备全局内存）
 * @param d_odata: 输出数据（每个线程块一个结果）
 * @param n: 数据总大小
 *
 * 优化策略：
 * 1. 通过索引计算替代模运算
 * 2. 连续的线程参与模式
 * 3. 避免线程束分支
 * 4. 更好的内存访问模式
 *
 * 性能优势：
 * - 没有线程束分支
 * - 更高的 SIMD 效率
 * - 更好的缓存利用率
 */
__global__ void ReduceNeighboredWithoutDivergence(float *d_idata, float *d_odata, unsigned int n)
{
    /**
     * 线程索引计算
     */
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /**
     * 将全局内存指针转换为当前线程块的局部指针
     */
    float *idata = d_idata + blockIdx.x * blockDim.x;

    /**
     * 边界检查
     */
    if(idx >= n) return;

    /**
     * 优化后的归约主循环
     *
     * 核心优化：使用 index = 2 * stride * tid
     * 避免了模运算和分支判断
     *
     * 示例（blockDim.x = 8）：
     * 迭代0: stride=1
     *   tid=0: index=0, 处理 idata[0] += idata[1]
     *   tid=1: index=2, 处理 idata[2] += idata[3]
     *   tid=2: index=4, 处理 idata[4] += idata[5]
     *   tid=3: index=6, 处理 idata[6] += idata[7]
     *   tid=4,5,6,7: index>=8, 不参与
     *
     * 迭代1: stride=2
     *   tid=0: index=0, 处理 idata[0] += idata[2]
     *   tid=1: index=4, 处理 idata[4] += idata[6]
     *   tid>=2: index>=8, 不参与
     *
     * 优势：连续的线程（tid=0,1,2,...）参与计算
     * 避免了同一个 warp 内的分支
     */
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        /**
         * 计算当前线程处理的数据索引
         * 这种方式保证了连续的线程参与
         */
        int index = 2 * stride * tid;

        /**
         * 判断索引是否有效
         * 没有使用模运算，没有分支
         */
        if (index < blockDim.x)
        {
            // 归约操作
            idata[index] += idata[index + stride];
        }

        /**
         * 线程同步
         */
        __syncthreads();
    }

    /**
     * 线程0将最终结果写入输出数组
     */
    if (tid == 0) d_odata[blockIdx.x] = idata[0];
}


/**
 * GPU 归约函数封装 - 带分支版本
 *
 * @param h_idata: 主机端输入数组
 * @param h_odata: 主机端输出数组
 * @param size: 数组大小
 * @param blockSize: 线程块大小
 *
 * 完整流程：
 * 1. 内存分配（设备端）
 * 2. 数据传输（主机->设备）
 * 3. 内核启动
 * 4. 结果回传（设备->主机）
 * 5. 资源清理
 */
void ReduceOnGPUWithDivergence(float *h_idata, float *h_odata, int size, int blockSize)
{
    /**
     * 计算内存大小
     * ibytes: 输入数组字节数
     * obytes: 输出数组字节数（每个线程块一个结果）
     */
    int ibytes = size * sizeof(float);
    int obytes = size / blockSize * sizeof(float);

    /**
     * 清零输出数组
     */
    memset(h_odata, 0, obytes);

    /**
     * 设备内存指针
     */
    float* d_idata = nullptr;
    float* d_odata = nullptr;

    /**
     * 分配设备内存
     */
    CUDA_CHECK(cudaMalloc(&d_idata, ibytes));
    CUDA_CHECK(cudaMalloc(&d_odata, obytes));

    /**
     * 将输入数据从主机复制到设备
     */
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));

    /**
     * 配置内核启动参数
     * block: 每个线程块 blockSize 个线程
     * grid: size/blockSize 个线程块
     */
    dim3 block(blockSize);
    dim3 grid(size / blockSize);

    /**
     * 启动 GPU 内核
     */
    ReduceNeighboredWithDivergence <<<grid, block>>> (d_idata, d_odata, size);

    /**
     * 将结果从设备复制回主机
     */
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));

    /**
     * 等待所有 GPU 操作完成
     */
    CUDA_CHECK(cudaDeviceSynchronize());

    /**
     * 检查内核执行是否有错误
     * 注意：在同步后检测内核错误
     */
    CUDA_KERNEL_CHECK();

    /**
     * 释放设备内存
     */
    CUDA_CHECK(cudaFree(d_odata));
    CUDA_CHECK(cudaFree(d_idata));
}

/**
 * GPU 归约函数封装 - 无分支优化版本
 *
 * @param h_idata: 主机端输入数组
 * @param h_odata: 主机端输出数组
 * @param size: 数组大小
 * @param blockSize: 线程块大小
 *
 * 与带分支版本的区别：
 * - 调用不同的内核函数
 * - 内核实现使用优化算法
 * - 性能更高
 */
void ReduceOnGPUWithoutDivergence(float *h_idata, float *h_odata, int size, int blockSize)
{
    int ibytes = size * sizeof(float);
    int obytes = size / blockSize * sizeof(float);

    memset(h_odata, 0, obytes);

    float* d_idata = nullptr;
    float* d_odata = nullptr;

    CUDA_CHECK(cudaMalloc(&d_idata, ibytes));
    CUDA_CHECK(cudaMalloc(&d_odata, obytes));

    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));

    dim3 block(blockSize);
    dim3 grid(size / blockSize);

    /**
     * 启动优化后的内核
     */
    ReduceNeighboredWithoutDivergence <<<grid, block>>> (d_idata, d_odata, size);

    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaFree(d_odata));
    CUDA_CHECK(cudaFree(d_idata));
}

/**
 * 学习要点总结：
 *
 * 1. 线程束分支的影响：
 *    - 同一个 warp(32个线程)必须执行相同指令
 *    - 分支导致串行执行，性能下降
 *    - 避免分支是 GPU 优化的关键
 *
 * 2. 归约算法优化技巧：
 *    - 使用索引计算替代模运算
 *    - 保持连续线程参与
 *    - 减少同步次数
 *    - 使用共享内存（进一步优化）
 *
 * 3. 内存访问模式：
 *    - 合并访问：连续线程访问连续内存
 *    - 缓存利用：重复访问的数据
 *    - 带宽利用：最大化内存吞吐量
 *
 * 4. 性能分析方法：
 *    - 比较不同实现的性能
 *    - 使用 nvprof/nsight 分析瓶颈
 *    - 识别分支、内存等瓶颈
 *
 * 5. 进一步优化方向：
 *    - 使用共享内存减少全局内存访问
 *    - 循环展开减少同步次数
 *    - Warp 级别原语（shuffle 指令）
 *    - 多级归约处理大数据集
 *
 * 6. 实际应用场景：
 *    - 数值计算：求和、均值、方差
 *    - 图像处理：直方图统计
 *    - 机器学习：损失函数计算
 *    - 科学计算：积分、累加
 */

