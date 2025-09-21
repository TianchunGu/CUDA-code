/**
 * 课程 3.4: CUDA 运行时 API 设备查询详解
 * 文件名: query.cu
 * 作者: 权 双
 * 日期: 2023-08-13
 * 功能: 使用 CUDA 运行时 API 查询 GPU 设备信息
 *
 * 本程序演示：
 * 1. 设备属性查询的基本方法
 * 2. 重要硬件参数的获取和显示
 * 3. GPU 计算能力和资源限制
 * 4. 为性能优化提供硬件信息基础
 *
 * 查询信息类别：
 * - 基本设备信息（名称、计算能力）
 * - 内存资源（全局内存、常量内存、共享内存）
 * - 执行配置限制（网格大小、线程块大小）
 * - 硬件架构（SM数量、寄存器数量）
 *
 * 学习目标：
 * - 掌握 cudaGetDeviceProperties API 的使用
 * - 理解各项设备属性的含义和作用
 * - 学会分析硬件规格对程序设计的影响
 * - 为后续性能优化奠定基础
 */

#include "../tools/common.cuh"
#include <stdio.h>

/**
 * 主函数 - GPU 设备信息查询演示
 */
int main(void)
{
    /**
     * 第一步：设置要查询的设备
     *
     * 在多 GPU 系统中，可以选择不同的设备进行查询
     * device_id = 0 表示使用第一个可用的 GPU 设备
     */
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    /**
     * 第二步：获取设备属性
     *
     * cudaDeviceProp 结构体包含了设备的详细属性信息
     * cudaGetDeviceProperties() 函数填充这个结构体
     */
    cudaDeviceProp prop;
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);

    printf("=== GPU 设备属性查询结果 ===\n\n");

    /**
     * 1. 基本设备标识信息
     *
     * 这些信息帮助识别具体的 GPU 型号和版本
     */
    printf("设备 ID:                                   %d\n", device_id);
    printf("设备名称:                                  %s\n", prop.name);

    /**
     * 2. 计算能力 (Compute Capability)
     *
     * 计算能力版本决定了 GPU 支持的 CUDA 特性：
     * - major.minor 格式（如 8.6, 7.5, 6.1）
     * - 不同版本支持不同的指令集和特性
     * - 影响代码编译目标和优化策略
     *
     * 主要版本对应的架构：
     * - 8.x: Ampere 架构（RTX 30系列, A100等）
     * - 7.x: Turing/Volta 架构（RTX 20系列, V100等）
     * - 6.x: Pascal 架构（GTX 10系列等）
     * - 5.x: Maxwell 架构
     */
    printf("计算能力:                                  %d.%d\n",
           prop.major, prop.minor);

    /**
     * 3. 内存资源信息
     *
     * 内存是 GPU 计算的关键资源，直接影响：
     * - 可处理的数据规模
     * - 内存访问性能
     * - 程序的可行性
     */
    printf("\n--- 内存资源 ---\n");

    // 全局内存：GPU 的主要存储空间
    printf("全局内存总量:                             %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024 * 1024));

    // 常量内存：只读内存，有缓存加速
    printf("常量内存总量:                             %.2f KB\n",
           prop.totalConstMem / 1024.0);

    // 共享内存（每个线程块）：高速缓存，线程块内共享
    printf("每个线程块的共享内存:                     %.2f KB\n",
           prop.sharedMemPerBlock / 1024.0);

    // 共享内存（每个 SM）：SM 级别的共享内存总量
    printf("每个 SM 的共享内存:                       %.2f KB\n",
           prop.sharedMemPerMultiprocessor / 1024.0);

    /**
     * 4. 执行配置限制
     *
     * 这些限制决定了内核启动时的配置上限：
     * - 影响 <<<grid, block>>> 配置
     * - 决定最大并行度
     * - 指导线程组织策略
     */
    printf("\n--- 执行配置限制 ---\n");

    // 最大网格维度：grid 的最大尺寸
    printf("最大网格大小 (grid):                      %d × %d × %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // 最大线程块维度：block 的最大尺寸
    printf("最大线程块大小 (block):                   %d × %d × %d\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

    // 每个线程块的最大线程数
    printf("每个线程块的最大线程数:                   %d\n",
           prop.maxThreadsPerBlock);

    /**
     * 5. 硬件架构信息
     *
     * 这些参数反映了 GPU 的物理架构：
     * - SM 数量决定并行处理能力
     * - 寄存器数量影响线程占用率
     * - 线程数量决定并发执行能力
     */
    printf("\n--- 硬件架构 ---\n");

    // SM (Streaming Multiprocessor) 数量
    printf("SM (流多处理器) 数量:                     %d\n",
           prop.multiProcessorCount);

    // 寄存器资源（每个线程块）
    printf("每个线程块的最大寄存器数:                 %d K\n",
           prop.regsPerBlock / 1024);

    // 寄存器资源（每个 SM）
    printf("每个 SM 的最大寄存器数:                   %d K\n",
           prop.regsPerMultiprocessor / 1024);

    // 线程执行能力（每个 SM）
    printf("每个 SM 的最大线程数:                     %d\n",
           prop.maxThreadsPerMultiProcessor);

    /**
     * 6. 性能分析和优化建议
     */
    printf("\n--- 性能分析建议 ---\n");

    // 计算总的理论并行线程数
    int maxThreads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    printf("理论最大并行线程数:                       %d\n", maxThreads);

    // 内存带宽估算建议
    float memoryGB = prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f);
    if (memoryGB >= 8.0f) {
        printf("内存容量:                                大容量 GPU，适合大数据处理\n");
    } else if (memoryGB >= 4.0f) {
        printf("内存容量:                                中等容量 GPU，适合中等规模计算\n");
    } else {
        printf("内存容量:                                小容量 GPU，适合学习和轻量级计算\n");
    }

    // SM 数量分析
    if (prop.multiProcessorCount >= 80) {
        printf("计算能力:                                高端 GPU，强大的并行计算能力\n");
    } else if (prop.multiProcessorCount >= 40) {
        printf("计算能力:                                中端 GPU，良好的并行计算能力\n");
    } else {
        printf("计算能力:                                入门级 GPU，基础的并行计算能力\n");
    }

    // 线程块配置建议
    printf("\n--- 配置建议 ---\n");
    printf("推荐线程块大小:                           128-512 个线程 (32的倍数)\n");
    printf("推荐占用率目标:                           75%% 以上\n");
    printf("内存合并访问:                             连续线程访问连续内存\n");

    printf("\n设备查询完成\n");

    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. 设备查询的重要性：
 *    - 了解硬件限制和能力
 *    - 指导程序设计和优化
 *    - 确保兼容性和可移植性
 *    - 预测性能表现
 *
 * 2. 关键设备属性：
 *    基础信息：
 *    - name: 设备名称，识别具体型号
 *    - major.minor: 计算能力，决定支持的特性
 *
 *    内存资源：
 *    - totalGlobalMem: 全局内存总量
 *    - totalConstMem: 常量内存总量
 *    - sharedMemPerBlock: 每线程块共享内存
 *    - sharedMemPerMultiprocessor: 每SM共享内存
 *
 *    执行限制：
 *    - maxGridSize: 最大网格维度
 *    - maxThreadsDim: 最大线程块维度
 *    - maxThreadsPerBlock: 每线程块最大线程数
 *
 *    硬件架构：
 *    - multiProcessorCount: SM数量
 *    - regsPerBlock/regsPerMultiprocessor: 寄存器资源
 *    - maxThreadsPerMultiProcessor: 每SM最大线程数
 *
 * 3. 性能优化指导：
 *    线程配置：
 *    - 选择合适的线程块大小
 *    - 确保是warp大小(32)的倍数
 *    - 考虑硬件资源限制
 *
 *    内存使用：
 *    - 根据内存容量设计数据结构
 *    - 合理使用共享内存加速
 *    - 注意常量内存的使用限制
 *
 *    占用率优化：
 *    - 平衡寄存器使用和线程数
 *    - 避免资源过度消耗
 *    - 追求高的SM占用率
 *
 * 4. 实际应用：
 *    - 自适应算法设计
 *    - 跨平台兼容性
 *    - 性能基准测试
 *    - 资源规划和预算
 *
 * 5. 扩展学习：
 *    - 比较不同GPU架构的特点
 *    - 学习计算能力版本的区别
 *    - 了解新特性的硬件要求
 *    - 掌握性能分析工具的使用
 */