/**
 * 课程 4.10: 高级 CUDA 项目结构 - 归约算法头文件
 * 文件名: reduce.hpp
 * 作者: 权 双
 * 日期: 2023-12-31
 * 功能: 并行归约算法接口声明
 *
 * 归约算法介绍：
 * 归约（Reduction）是将数组中的所有元素通过某种操作（如求和、求最大值等）
 * 合并为单个值的过程。这是并行计算中的基本算法之一。
 *
 * 并行归约的挑战：
 * 1. 线程同步 - 确保计算顺序正确
 * 2. 内存访问模式 - 优化带宽利用
 * 3. 线程束分支 - 避免 SIMD 效率损失
 * 4. 工作负载均衡 - 保持所有线程忙碌
 *
 * 本文件提供三种归约实现：
 * - CPU 版本：基准实现
 * - GPU 带分支版本：教学演示
 * - GPU 无分支版本：优化实现
 */

#ifndef __REDUCE_HPP__
#define __REDUCE_HPP__

/**
 * GPU 归约算法 - 带分支版本
 *
 * @param h_idata: 主机端输入数组
 * @param h_odata: 主机端输出数组（每个线程块一个结果）
 * @param size: 输入数组大小
 * @param blockSize: GPU 线程块大小
 *
 * 特点：
 * - 使用模运算(%)判断线程参与
 * - 存在线程束分支（Warp Divergence）
 * - 实现简单直观但效率较低
 * - 适合教学演示分支对性能的影响
 *
 * 算法流程：
 * 1. 每个线程块处理 blockSize 个元素
 * 2. 通过步长递增的方式进行树形归约
 * 3. 使用 if (tid % stride == 0) 控制线程参与
 * 4. 每个线程块产生一个部分结果
 */
void ReduceOnGPUWithDivergence(float *h_idata, float *h_odata, int size, int blockSize);

/**
 * GPU 归约算法 - 无分支优化版本
 *
 * @param h_idata: 主机端输入数组
 * @param h_odata: 主机端输出数组（每个线程块一个结果）
 * @param size: 输入数组大小
 * @param blockSize: GPU 线程块大小
 *
 * 特点：
 * - 通过索引计算避免分支
 * - 减少线程束分支，提高 SIMD 效率
 * - 更好的内存访问模式
 * - 性能优于带分支版本
 *
 * 优化技巧：
 * 1. 使用 index = 2 * stride * tid 计算参与线程
 * 2. 避免模运算和条件分支
 * 3. 保持连续的线程参与模式
 * 4. 提高线程束执行效率
 */
void ReduceOnGPUWithoutDivergence(float *h_idata, float *h_odata, int size, int blockSize);

/**
 * CPU 归约算法 - 基准实现
 *
 * @param data: 输入数组
 * @param size: 数组大小
 * @return 归约结果（所有元素之和）
 *
 * 特点：
 * - 单线程顺序处理
 * - 实现简单直观
 * - 作为正确性验证的参考
 * - 用于性能对比基准
 *
 * extern 说明：
 * - 函数实现在单独的 .cpp 文件中
 * - 避免重复定义
 * - 便于模块化编译
 */
extern float ReduceOnCPU(float *data, int const size);

#endif

/**
 * 学习要点总结：
 *
 * 1. 并行归约原理：
 *    - 分治法思想
 *    - 树形归约模式
 *    - log(N) 步完成计算
 *    - 需要同步协调
 *
 * 2. GPU 优化策略：
 *    - 避免线程束分支
 *    - 优化内存访问模式
 *    - 减少同步开销
 *    - 平衡工作负载
 *
 * 3. 算法对比：
 *    - CPU: O(N) 时间，单线程
 *    - GPU 带分支: O(log N) 步，有分支开销
 *    - GPU 无分支: O(log N) 步，优化执行效率
 *
 * 4. 实际应用：
 *    - 数值计算：求和、均值、方差
 *    - 图形处理：像素统计
 *    - 机器学习：损失函数计算
 *    - 科学计算：数值积分
 */
