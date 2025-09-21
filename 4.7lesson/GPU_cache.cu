/**
 * 课程 4.7: GPU 缓存系统详解
 * 文件名: GPU_cache.cu
 * 作者: 权 双
 * 日期: 2023-12-30
 * 功能: GPU 缓存的查询和理解
 *
 * GPU 缓存系统概述：
 * 1. L1 缓存 - 每个 SM 私有，用于全局内存和共享内存
 * 2. L2 缓存 - 全 GPU 共享，连接所有 SM 和全局内存
 * 3. 常量缓存 - 专用于常量内存的缓存
 * 4. 纹理缓存 - 专用于纹理内存的缓存
 * 5. 指令缓存 - 存储已编译的 GPU 指令
 *
 * 缓存的重要性：
 * - 减少内存访问延迟
 * - 提高内存带宽利用率
 * - 改善数据局部性访问性能
 * - 优化重复数据访问模式
 */

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

/**
 * 演示内核函数 - 目前为空实现
 *
 * 在实际应用中，这里可以添加各种内存访问模式来测试缓存效果：
 * 1. 顺序访问 - 测试合并访问和缓存行利用率
 * 2. 随机访问 - 测试缓存命中率
 * 3. 重复访问 - 测试缓存保持和重用
 * 4. 步长访问 - 测试缓存行冲突
 */
__global__ void kernel(void)
{
    /**
     * 这里可以添加不同的内存访问模式来研究缓存性能：
     *
     * 示例1 - 顺序访问模式：
     * int tid = threadIdx.x + blockIdx.x * blockDim.x;
     * float value = global_array[tid];  // 利用缓存行的顺序访问
     *
     * 示例2 - 重复访问模式：
     * float shared_value = global_array[0];  // 多个线程访问同一位置
     *
     * 示例3 - 步长访问模式：
     * int stride = 1024;
     * float value = global_array[tid * stride];  // 测试缓存行利用率
     */
}

/**
 * 主函数 - GPU 缓存信息查询和分析
 */
int main(int argc, char **argv)
{
    /**
     * 第一步：获取 GPU 设备信息
     */
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备: " << deviceProps.name << std::endl;

    /**
     * 第二步：查询和显示缓存相关信息
     */
    std::cout << "\n=== GPU 缓存系统信息 ===" << std::endl;

    /**
     * L1 缓存支持查询
     *
     * globalL1CacheSupported 属性说明：
     * - true: 支持全局内存的 L1 缓存
     * - false: 不支持或默认禁用 L1 缓存
     *
     * 历史背景：
     * - Fermi 架构: 支持可配置的 L1 缓存
     * - Kepler 架构: 部分型号支持
     * - Maxwell/Pascal: 通常不支持全局内存 L1 缓存
     * - Volta/Turing/Ampere: 重新引入并改进了 L1 缓存
     */
    if (deviceProps.globalL1CacheSupported) {
        std::cout << "✓ 支持全局内存 L1 缓存" << std::endl;

        /**
         * L1 缓存的特点和优势：
         * - 访问延迟: ~25-30 个时钟周期
         * - 带宽: 非常高的内部带宽
         * - 作用范围: 单个 SM 内的所有线程块
         * - 一致性: 在 SM 内自动维护
         */
        std::cout << "  - L1 缓存加速全局内存访问" << std::endl;
        std::cout << "  - 改善数据局部性访问性能" << std::endl;
        std::cout << "  - 减少内存访问延迟" << std::endl;
    } else {
        std::cout << "✗ 不支持全局内存 L1 缓存" << std::endl;
        std::cout << "  - 全局内存访问直接到 L2 缓存" << std::endl;
        std::cout << "  - 可通过共享内存手动实现类似功能" << std::endl;
    }

    /**
     * L2 缓存信息查询
     *
     * L2 缓存是现代 GPU 的核心组件：
     * - 所有 SM 共享
     * - 连接 SM 和全局内存
     * - 自动管理，程序员无需干预
     * - 支持原子操作
     */
    float l2SizeMB = deviceProps.l2CacheSize / (1024.0f * 1024.0f);
    std::cout << "\nL2 缓存大小: " << l2SizeMB << " MB" << std::endl;

    /**
     * L2 缓存大小的意义：
     * - 影响整体内存性能
     * - 决定可同时缓存的数据量
     * - 影响多 SM 间的数据共享效率
     */
    if (l2SizeMB >= 4.0f) {
        std::cout << "  - 大容量 L2 缓存，适合大数据集处理" << std::endl;
    } else if (l2SizeMB >= 1.0f) {
        std::cout << "  - 中等容量 L2 缓存，平衡性能与成本" << std::endl;
    } else {
        std::cout << "  - 小容量 L2 缓存，适合计算密集型任务" << std::endl;
    }

    /**
     * 第三步：显示其他缓存相关的设备信息
     */
    std::cout << "\n=== 其他内存层次信息 ===" << std::endl;

    /**
     * 全局内存信息
     */
    float globalMemGB = deviceProps.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f);
    std::cout << "全局内存总量: " << globalMemGB << " GB" << std::endl;

    /**
     * 共享内存信息（每个线程块）
     */
    float sharedMemKB = deviceProps.sharedMemPerBlock / 1024.0f;
    std::cout << "共享内存每块: " << sharedMemKB << " KB" << std::endl;

    /**
     * 常量内存信息
     */
    float constMemKB = deviceProps.totalConstMem / 1024.0f;
    std::cout << "常量内存总量: " << constMemKB << " KB" << std::endl;

    /**
     * 寄存器信息
     */
    std::cout << "每个线程块寄存器: " << deviceProps.regsPerBlock << " 个" << std::endl;

    /**
     * 第四步：缓存优化建议
     */
    std::cout << "\n=== 缓存优化建议 ===" << std::endl;

    if (deviceProps.globalL1CacheSupported) {
        std::cout << "1. L1 缓存优化:" << std::endl;
        std::cout << "   - 使用连续的内存访问模式" << std::endl;
        std::cout << "   - 避免不规则的内存访问" << std::endl;
        std::cout << "   - 重用已加载的数据" << std::endl;
    }

    std::cout << "2. L2 缓存优化:" << std::endl;
    std::cout << "   - 优化线程块调度以提高数据重用" << std::endl;
    std::cout << "   - 使用合并内存访问" << std::endl;
    std::cout << "   - 考虑数据布局对缓存友好性的影响" << std::endl;

    std::cout << "3. 通用缓存优化策略:" << std::endl;
    std::cout << "   - 利用共享内存作为用户控制的缓存" << std::endl;
    std::cout << "   - 使用常量内存缓存只读数据" << std::endl;
    std::cout << "   - 优化内存访问模式以减少缓存丢失" << std::endl;

    /**
     * 第五步：启动演示内核（当前为空）
     */
    dim3 block(1);
    dim3 grid(1);

    std::cout << "\n启动演示内核..." << std::endl;
    kernel<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "内核执行完成" << std::endl;

    /**
     * 第六步：清理资源
     */
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. GPU 缓存层次结构：
 *    - L1 缓存: SM 级别，快速但小容量
 *    - L2 缓存: GPU 级别，中等速度和容量
 *    - 全局内存: 大容量但访问较慢
 *    - 专用缓存: 常量缓存、纹理缓存等
 *
 * 2. 缓存对性能的影响：
 *    - 缓存命中: 大幅减少内存访问延迟
 *    - 缓存丢失: 需要访问更慢的内存层次
 *    - 带宽利用: 改善整体内存带宽使用
 *    - 能耗效率: 缓存访问比内存访问更节能
 *
 * 3. 缓存优化策略：
 *    a) 空间局部性:
 *       - 连续内存访问
 *       - 合并内存事务
 *       - 避免分散的内存访问
 *
 *    b) 时间局部性:
 *       - 重复访问相同数据
 *       - 在缓存驻留期间最大化数据使用
 *       - 避免过大的工作集
 *
 *    c) 访问模式优化:
 *       - 顺序访问优于随机访问
 *       - 避免大步长访问
 *       - 考虑缓存行大小（通常 128 字节）
 *
 * 4. 程序设计考虑：
 *    - 数据结构布局: 结构体数组 vs 数组结构体
 *    - 算法设计: 分块算法以适应缓存大小
 *    - 内存访问模式: 预测和优化访问序列
 *    - 共享内存使用: 作为程序员控制的缓存
 *
 * 5. 调试和分析工具：
 *    - NVIDIA Nsight Compute: 详细的缓存分析
 *    - nvprof: 基本的内存性能分析
 *    - 设备查询: 了解硬件缓存特性
 *    - 微基准测试: 验证缓存行为
 *
 * 6. 不同架构的缓存特点：
 *    - Fermi: 可配置 L1/共享内存比例
 *    - Kepler: 只读数据缓存
 *    - Maxwell: 统一内存架构
 *    - Pascal: 改进的内存压缩
 *    - Volta: 独立线程调度
 *    - Turing: RT 核心和 Tensor 核心
 *    - Ampere: 多实例 GPU 支持
 *
 * 7. 实际应用场景：
 *    - 科学计算: 矩阵运算的分块优化
 *    - 图像处理: 邻域操作的数据重用
 *    - 机器学习: 卷积操作的缓存优化
 *    - 数据库: 连接操作的缓存策略
 *
 * 8. 性能测量和验证：
 *    - 缓存命中率监控
 *    - 内存带宽利用率测量
 *    - 延迟分析
 *    - 能耗效率评估
 */