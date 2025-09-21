/**
 * 课程 4.10: 高级 CUDA 项目结构 - 并行归约算法主程序
 * 文件名: main.cpp
 * 作者: 权 双
 * 日期: 2023-12-31
 * 功能: 并行归约算法的性能对比主程序
 *
 * 程序功能：
 * 1. CPU vs GPU 归约算法性能对比
 * 2. GPU 分支优化对比（带分支 vs 不带分支）
 * 3. 高精度性能测量和分析
 * 4. 模块化设计演示
 *
 * 归约算法介绍：
 * - 归约(Reduction)是将数组中所有元素合并为单个值的操作
 * - 常见操作：求和、求最大值、求最小值、求积等
 * - 并行归约是 GPU 计算的经典应用场景
 * - 涉及线程间协作和同步优化
 *
 * 性能优化要点：
 * - 线程束分支(Warp Divergence)的影响
 * - 内存合并访问模式
 * - 线程块内同步优化
 * - 全局内存访问模式优化
 */

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <string>
#include "utils.hpp"        // 工具函数：错误检查、数据初始化等
#include "timer.hpp"        // 性能计时器：CPU和GPU时间测量
#include "reduce.hpp"       // 归约算法：CPU和GPU实现
#include <cstring>
#include <memory>
#include <cmath>

/**
 * 全局变量：随机数种子
 * 用于确保每次运行使用相同的测试数据，便于性能对比
 */
int seed;

/**
 * 主函数 - 并行归约算法性能对比
 *
 * @param argc: 命令行参数个数
 * @param argv: 命令行参数数组
 *               argv[1]: 数组大小（元素个数）
 *               argv[2]: GPU线程块大小
 *
 * 程序流程：
 * 1. 参数解析和验证
 * 2. 数据准备和初始化
 * 3. CPU 归约性能测试
 * 4. GPU 预热运行
 * 5. GPU 归约性能测试（带分支版本）
 * 6. GPU 归约性能测试（无分支优化版本）
 * 7. 结果验证和性能分析
 */
int main(int argc, char *argv[])
{
    /**
     * 第一步：命令行参数解析和验证
     *
     * 参数要求：
     * - argc == 3: 程序名 + 2个参数
     * - argv[1]: 数组大小，建议使用2的幂次方便于GPU处理
     * - argv[2]: 线程块大小，通常为64/128/256/512
     */
    if (argc != 3)
    {
        std::cerr << "用法: ./build/reduction [size] [blockSize]" << std::endl;
        std::cerr << "示例: ./build/reduction 4096 256" << std::endl;
        std::cerr << "说明:" << std::endl;
        std::cerr << "  size: 数组大小，建议使用2的幂次方" << std::endl;
        std::cerr << "  blockSize: GPU线程块大小，建议64/128/256/512" << std::endl;
        return -1;
    }

    /**
     * 第二步：参数转换和配置初始化
     */
    Timer timer;                            // 性能计时器实例
    char str[100];                         // 字符串缓冲区，用于格式化输出
    int size = std::stoi(argv[1]);         // 数组大小转换
    int blockSize = std::stoi(argv[2]);    // 线程块大小转换

    /**
     * 计算网格大小
     * gridsize = 总元素数 / 每个线程块的线程数
     * 每个线程块处理 blockSize 个元素，输出1个归约结果
     */
    int gridsize = size / blockSize;

    /**
     * 第三步：内存分配和数据准备
     *
     * 内存布局：
     * - h_idata: 主机端输入数据数组
     * - h_odata: 主机端输出数据数组（每个线程块一个结果）
     */
    float* h_idata = nullptr;
    float* h_odata = nullptr;
    h_idata = (float*)malloc(size * sizeof(float));      // 输入数据：size个元素
    h_odata = (float*)malloc(gridsize * sizeof(float));  // 输出数据：gridsize个中间结果

    /**
     * 数据初始化
     * - 使用固定种子确保结果可重现
     * - 输入数据随机初始化
     * - 输出数组清零
     */
    seed = 1;
    initMatrix(h_idata, size, seed);                    // 初始化输入数据
    memset(h_odata, 0, gridsize * sizeof(float));      // 清零输出数组

    std::cout << "\n==== 并行归约算法性能对比测试 ====" << std::endl;
    std::cout << "数组大小: " << size << " 个元素" << std::endl;
    std::cout << "线程块大小: " << blockSize << " 个线程" << std::endl;
    std::cout << "网格大小: " << gridsize << " 个线程块" << std::endl;
    std::cout << "总线程数: " << gridsize * blockSize << std::endl;

    /**
     * 第四步：CPU 归约性能测试
     *
     * CPU 算法特点：
     * - 单线程顺序处理
     * - 内存访问模式简单
     * - 无需考虑线程同步
     * - 作为性能对比的基准
     */
    std::cout << "\n==== CPU 归约测试 ====" << std::endl;
    timer.start_cpu();
    float sumOnCPU = ReduceOnCPU(h_idata, size);
    timer.stop_cpu();
    std::sprintf(str, "CPU 归约计算，结果: %f", sumOnCPU);
    timer.duration_cpu<Timer::ms>(str);

    /**
     * 第五步：GPU 预热运行
     *
     * GPU 预热的重要性：
     * - 首次 CUDA 调用有初始化开销
     * - 驱动程序和运行时初始化
     * - GPU 时钟频率调整到最佳状态
     * - 确保后续测试的准确性
     */
    std::cout << "\n==== GPU 预热运行 ====" << std::endl;
    timer.start_gpu();
    ReduceOnGPUWithDivergence(h_idata, h_odata, size, blockSize);
    timer.stop_gpu();
    std::cout << "GPU 预热完成" << std::endl;

    /**
     * 第六步：GPU 归约性能测试 - 带分支版本
     *
     * 带分支算法特点：
     * - 使用模运算判断线程是否参与计算
     * - 存在线程束分支 (Warp Divergence)
     * - 性能相对较低但实现简单
     * - 教学演示线程束分支的影响
     */
    std::cout << "\n==== GPU 归约测试（带分支版本）====" << std::endl;
    timer.start_gpu();
    ReduceOnGPUWithDivergence(h_idata, h_odata, size, blockSize);
    timer.stop_gpu();

    /**
     * 收集和汇总所有线程块的归约结果
     * - 每个线程块产生一个中间结果
     * - 主机端最终汇总这些中间结果
     * - 在实际应用中可以递归使用 GPU 继续归约
     */
    float sumOnGPUWithDivergence = 0;
    for (int i = 0; i < gridsize; i++)
        sumOnGPUWithDivergence += h_odata[i];

    std::sprintf(str, "GPU 归约计算（带分支），结果: %f", sumOnGPUWithDivergence);
    timer.duration_gpu(str);

    /**
     * 第七步：GPU 归约性能测试 - 无分支优化版本
     *
     * 无分支算法特点：
     * - 通过改变索引计算避免分支
     * - 减少线程束分支，提高效率
     * - 更复杂的索引计算逻辑
     * - 展示 GPU 优化的重要技巧
     */
    std::cout << "\n==== GPU 归约测试（无分支优化版本）====" << std::endl;
    timer.start_gpu();
    ReduceOnGPUWithoutDivergence(h_idata, h_odata, size, blockSize);
    timer.stop_gpu();

    /**
     * 收集无分支版本的归约结果
     */
    float sumOnGPUWithoutDivergence = 0;
    for (int i = 0; i < gridsize; i++)
        sumOnGPUWithoutDivergence += h_odata[i];

    std::sprintf(str, "GPU 归约计算（无分支优化），结果: %f", sumOnGPUWithoutDivergence);
    timer.duration_gpu(str);

    /**
     * 第八步：结果验证和错误分析
     *
     * 验证策略：
     * - 比较 CPU 和 GPU 计算结果
     * - 使用浮点数容差比较
     * - 分析可能的误差来源
     */
    std::cout << "\n==== 结果验证 ====" << std::endl;
    float epsilon = 1e-6f;

    bool resultCorrect = true;
    if (std::abs(sumOnCPU - sumOnGPUWithDivergence) > epsilon) {
        std::cout << "❌ 带分支 GPU 归约结果不正确！" << std::endl;
        std::cout << "CPU 结果: " << sumOnCPU << std::endl;
        std::cout << "GPU 结果: " << sumOnGPUWithDivergence << std::endl;
        std::cout << "误差: " << std::abs(sumOnCPU - sumOnGPUWithDivergence) << std::endl;
        resultCorrect = false;
    } else {
        std::cout << "✅ 带分支 GPU 归约结果正确" << std::endl;
    }

    if (std::abs(sumOnCPU - sumOnGPUWithoutDivergence) > epsilon) {
        std::cout << "❌ 无分支 GPU 归约结果不正确！" << std::endl;
        std::cout << "CPU 结果: " << sumOnCPU << std::endl;
        std::cout << "GPU 结果: " << sumOnGPUWithoutDivergence << std::endl;
        std::cout << "误差: " << std::abs(sumOnCPU - sumOnGPUWithoutDivergence) << std::endl;
        resultCorrect = false;
    } else {
        std::cout << "✅ 无分支 GPU 归约结果正确" << std::endl;
    }

    /**
     * 第九步：性能分析和总结
     */
    if (resultCorrect) {
        std::cout << "\n==== 性能分析总结 ====" << std::endl;
        std::cout << "1. 算法正确性：所有实现都产生了正确的归约结果" << std::endl;
        std::cout << "2. 线程束分支影响：对比带分支和无分支版本的性能差异" << std::endl;
        std::cout << "3. 并行化效果：观察 GPU 相对于 CPU 的加速比" << std::endl;
        std::cout << "4. 优化空间：还可以使用共享内存进一步优化" << std::endl;

        std::cout << "\n性能优化建议：" << std::endl;
        std::cout << "• 使用合适的线程块大小（128-512通常最优）" << std::endl;
        std::cout << "• 避免线程束分支以提高 SIMD 效率" << std::endl;
        std::cout << "• 考虑使用共享内存减少全局内存访问" << std::endl;
        std::cout << "• 对于大数据集，可以实现多级归约" << std::endl;
    }

    /**
     * 第十步：资源清理
     */
    free(h_idata);
    free(h_odata);

    std::cout << "\n程序执行完成，内存资源已清理" << std::endl;
    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. 项目结构设计：
 *    - 模块化设计：头文件与源文件分离
 *    - 职责分离：不同功能模块独立实现
 *    - 易于维护：清晰的目录结构和命名规范
 *    - 可扩展性：便于添加新的算法和功能
 *
 * 2. 归约算法原理：
 *    - 分治思想：将大问题分解为小问题
 *    - 树形归约：层次化合并减少计算步骤
 *    - 并行模式：多个线程同时参与计算
 *    - 同步机制：确保计算顺序和数据一致性
 *
 * 3. GPU 性能优化：
 *    - 线程束分支：影响 SIMD 执行效率
 *    - 内存合并：提高内存带宽利用率
 *    - 占用率优化：平衡线程数和资源使用
 *    - 算法设计：选择适合 GPU 架构的算法
 *
 * 4. 性能测试方法：
 *    - 预热运行：消除初始化开销
 *    - 多次测量：获得稳定的性能数据
 *    - 结果验证：确保优化不影响正确性
 *    - 对比分析：量化优化效果
 *
 * 5. 工程实践：
 *    - 错误处理：完善的错误检查机制
 *    - 代码复用：工具函数的模块化设计
 *    - 性能监控：精确的时间测量工具
 *    - 文档规范：详细的注释和说明
 *
 * 6. 扩展方向：
 *    - 共享内存优化：进一步提升性能
 *    - 多 GPU 支持：扩展到多设备计算
 *    - 不同数据类型：支持整数、双精度等
 *    - 其他归约操作：最大值、最小值、积等
 */
