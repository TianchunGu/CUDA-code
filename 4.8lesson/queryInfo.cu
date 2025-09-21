/**
 * 课程 4.8: CUDA GPU 设备属性详细查询
 * 文件名: queryInfo.cu
 * 作者: 权 双
 * 日期: 2023-12-30
 * 功能: 查询和显示 GPU 硬件规格和性能参数
 *
 * 查询内容包括：
 * 1. 硬件架构信息 - SM 数量、计算能力等
 * 2. 内存层次信息 - 缓存大小、内存带宽等
 * 3. 执行资源信息 - 线程数、寄存器数等
 * 4. 特性支持信息 - 各种 CUDA 特性支持情况
 *
 * 学习目标：
 * - 了解 GPU 硬件架构参数
 * - 掌握设备查询的方法和技巧
 * - 理解各参数对性能的影响
 * - 为程序优化提供硬件信息基础
 */

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

/**
 * 主函数 - GPU 设备属性详细查询
 */
int main(int argc, char **argv)
{
    /**
     * 设备初始化和基本信息获取
     */
    int devID = 0;  // 使用设备 0
    cudaDeviceProp deviceProps;  // 设备属性结构体

    // 获取设备属性
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));

    std::cout << "=== GPU 设备详细信息查询 ===" << std::endl;
    std::cout << "设备 ID: " << devID << std::endl;
    std::cout << "设备名称: " << deviceProps.name << std::endl;

    /**
     * 1. 计算核心架构信息
     *
     * SM (Streaming Multiprocessor) 是 GPU 的基本计算单元
     * - 每个 SM 包含多个 CUDA 核心
     * - SM 数量直接影响并行计算能力
     * - 不同架构的 SM 设计差异很大
     */
    std::cout << "\n=== 计算架构信息 ===" << std::endl;
    std::cout << "SM (流多处理器) 数量: " << deviceProps.multiProcessorCount << std::endl;
    std::cout << "计算能力: " << deviceProps.major << "." << deviceProps.minor << std::endl;

    // 估算 CUDA 核心数（简化计算，实际依赖架构）
    int cudaCores = 0;
    if (deviceProps.major == 8) {  // Ampere 架构
        cudaCores = deviceProps.multiProcessorCount * 128;
    } else if (deviceProps.major == 7) {  // Turing/Volta 架构
        cudaCores = deviceProps.multiProcessorCount * 64;
    } else if (deviceProps.major == 6) {  // Pascal 架构
        cudaCores = deviceProps.multiProcessorCount * 128;
    }
    if (cudaCores > 0) {
        std::cout << "估算 CUDA 核心数: " << cudaCores << std::endl;
    }

    /**
     * 2. 内存层次和缓存信息
     *
     * GPU 内存层次结构：
     * - L2 缓存：所有 SM 共享，连接 SM 和全局内存
     * - L1 缓存：每个 SM 私有（如果支持）
     * - 共享内存：每个 SM 的高速存储
     * - 全局内存：GPU 主内存
     */
    std::cout << "\n=== 内存层次信息 ===" << std::endl;

    // L2 缓存信息
    float l2CacheMB = deviceProps.l2CacheSize / (1024.0f * 1024.0f);
    std::cout << "L2 缓存大小: " << l2CacheMB << " MB" << std::endl;

    // L1 缓存支持情况
    std::cout << "L1 缓存支持 (全局内存): "
              << (deviceProps.globalL1CacheSupported ? "是" : "否") << std::endl;
    std::cout << "L1 缓存支持 (本地内存): "
              << (deviceProps.localL1CacheSupported ? "是" : "否") << std::endl;

    // 共享内存信息（每个 SM）
    float sharedMemKB = deviceProps.sharedMemPerMultiprocessor / 1024.0f;
    std::cout << "每个 SM 的共享内存: " << sharedMemKB << " KB" << std::endl;

    // 全局内存信息
    float globalMemGB = deviceProps.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f);
    std::cout << "全局内存总量: " << globalMemGB << " GB" << std::endl;

    /**
     * 3. 执行资源和并行能力
     *
     * 这些参数决定了 GPU 的并行执行能力：
     * - 线程数：影响占用率和并行度
     * - 寄存器数：影响线程块大小和占用率
     * - 线程块数：影响 SM 资源利用效率
     */
    std::cout << "\n=== 执行资源信息 ===" << std::endl;

    // 线程相关限制
    std::cout << "每个 SM 最大驻留线程数: " << deviceProps.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个线程块最大线程数: " << deviceProps.maxThreadsPerBlock << std::endl;
    std::cout << "每个 SM 最大线程块数: " << deviceProps.maxBlocksPerMultiProcessor << std::endl;

    // 寄存器资源
    int regsPerSM_K = deviceProps.regsPerMultiprocessor / 1024;
    std::cout << "每个 SM 的 32 位寄存器数: " << regsPerSM_K << "K 个" << std::endl;
    std::cout << "每个线程块最大寄存器数: " << deviceProps.regsPerBlock << " 个" << std::endl;

    /**
     * 4. 内存带宽和性能指标
     *
     * 内存性能直接影响 GPU 计算效率：
     * - 内存时钟频率：影响数据传输速度
     * - 内存总线宽度：影响并行传输能力
     * - 理论带宽 = 频率 × 总线宽度 × 2 (DDR)
     */
    std::cout << "\n=== 内存性能信息 ===" << std::endl;

    // 内存规格
    std::cout << "内存总线宽度: " << deviceProps.memoryBusWidth << " 位" << std::endl;

    // 内存频率（转换为 GHz）
    float memClockGHz = deviceProps.memoryClockRate / (1024.0f * 1024.0f);
    std::cout << "内存时钟频率: " << memClockGHz << " GHz" << std::endl;

    // 计算理论内存带宽
    // 公式：带宽 = 频率 × 总线宽度 × 2 (DDR) / 8 (位转字节)
    float theoreticalBandwidth = (deviceProps.memoryClockRate * 2.0f * deviceProps.memoryBusWidth) / (8.0f * 1024.0f * 1024.0f * 1024.0f);
    std::cout << "理论内存带宽: " << theoreticalBandwidth << " GB/s" << std::endl;

    /**
     * 5. 高级特性支持情况
     *
     * 现代 GPU 支持多种高级特性：
     * - 流优先级：允许设置不同流的执行优先级
     * - 并发内核：支持多个内核同时执行
     * - 统一内存：简化内存管理
     */
    std::cout << "\n=== 高级特性支持 ===" << std::endl;

    std::cout << "流优先级支持: "
              << (deviceProps.streamPrioritiesSupported ? "是" : "否") << std::endl;
    std::cout << "并发内核执行支持: "
              << (deviceProps.concurrentKernels ? "是" : "否") << std::endl;
    std::cout << "统一内存支持: "
              << (deviceProps.managedMemory ? "是" : "否") << std::endl;
    std::cout << "ECC 内存支持: "
              << (deviceProps.ECCEnabled ? "是" : "否") << std::endl;

    /**
     * 6. 网格和线程块限制
     *
     * 这些限制影响内核启动配置：
     * - 网格维度：限制可启动的线程块数量
     * - 线程块维度：限制线程组织方式
     */
    std::cout << "\n=== 执行配置限制 ===" << std::endl;

    std::cout << "最大网格维度: ("
              << deviceProps.maxGridSize[0] << ", "
              << deviceProps.maxGridSize[1] << ", "
              << deviceProps.maxGridSize[2] << ")" << std::endl;

    std::cout << "最大线程块维度: ("
              << deviceProps.maxThreadsDim[0] << ", "
              << deviceProps.maxThreadsDim[1] << ", "
              << deviceProps.maxThreadsDim[2] << ")" << std::endl;

    /**
     * 7. 纹理和表面内存支持
     */
    std::cout << "\n=== 纹理和表面内存 ===" << std::endl;
    std::cout << "1D 纹理最大尺寸: " << deviceProps.maxTexture1D << std::endl;
    std::cout << "2D 纹理最大尺寸: ("
              << deviceProps.maxTexture2D[0] << ", "
              << deviceProps.maxTexture2D[1] << ")" << std::endl;
    std::cout << "3D 纹理最大尺寸: ("
              << deviceProps.maxTexture3D[0] << ", "
              << deviceProps.maxTexture3D[1] << ", "
              << deviceProps.maxTexture3D[2] << ")" << std::endl;

    /**
     * 8. 性能分析和优化建议
     */
    std::cout << "\n=== 性能分析建议 ===" << std::endl;

    // 计算理论峰值性能指标
    int totalCudaCores = cudaCores;
    if (totalCudaCores > 0) {
        std::cout << "建议线程块大小: 128-512 (warp 的倍数)" << std::endl;
        std::cout << "建议占用率目标: >75%" << std::endl;
    }

    // 根据 SM 数量给出建议
    if (deviceProps.multiProcessorCount >= 80) {
        std::cout << "高端 GPU：适合大规模并行计算" << std::endl;
    } else if (deviceProps.multiProcessorCount >= 40) {
        std::cout << "中端 GPU：适合中等规模并行计算" << std::endl;
    } else {
        std::cout << "入门级 GPU：适合学习和轻量级计算" << std::endl;
    }

    // 内存带宽建议
    if (theoreticalBandwidth > 500) {
        std::cout << "高带宽内存：适合内存密集型应用" << std::endl;
    } else {
        std::cout << "标准带宽内存：注意内存访问优化" << std::endl;
    }

    /**
     * 清理和退出
     */
    CUDA_CHECK(cudaDeviceReset());

    std::cout << "\n设备查询完成" << std::endl;
    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. 设备查询的重要性：
 *    - 了解硬件能力和限制
 *    - 指导程序优化策略
 *    - 适配不同的 GPU 架构
 *    - 预测性能表现
 *
 * 2. 关键硬件参数：
 *    - SM 数量：决定并行度
 *    - 内存带宽：影响数据传输
 *    - 缓存大小：影响访问效率
 *    - 寄存器数量：影响占用率
 *
 * 3. 性能优化指导：
 *    - 线程块大小选择
 *    - 占用率目标设定
 *    - 内存访问模式优化
 *    - 资源使用平衡
 *
 * 4. 架构适配策略：
 *    - 不同架构的特点
 *    - 特性支持差异
 *    - 性能优化重点
 *    - 兼容性考虑
 *
 * 5. 实际应用价值：
 *    - 性能调优基础
 *    - 硬件选型参考
 *    - 算法适配指导
 *    - 瓶颈分析依据
 */