/**
 * 课程 4.10: 高级 CUDA 项目结构 - 性能计时器实现
 * 文件名: timer.cpp
 * 作者: 权 双
 * 日期: 2023-12-31
 * 功能: 高精度性能计时器类的实现
 *
 * Timer 类实现细节：
 * 1. CPU 计时：使用 C++11 chrono 高精度时钟
 * 2. GPU 计时：使用 CUDA 事件记录
 * 3. 资源管理：RAII 模式自动管理 CUDA 事件
 * 4. 格式化输出：统一的性能报告格式
 */

#include <chrono>
#include <iostream>
#include <memory>
#include "timer.hpp"

#include "utils.hpp"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

/**
 * Timer 类构造函数
 *
 * 功能：
 * - 初始化成员变量
 * - 创建 CUDA 事件对象
 * - 设置初始状态
 *
 * CUDA 事件说明：
 * - cudaEventCreate() 创建事件
 * - 事件用于精确计时
 * - 支持异步操作
 * - 需要手动释放资源
 */
Timer::Timer(){
    _timeElasped = 0;  // 初始化耗时为0

    // 初始化 CPU 计时器
    _cStart = std::chrono::high_resolution_clock::now();
    _cStop = std::chrono::high_resolution_clock::now();

    // 创建 GPU 事件
    cudaEventCreate(&_gStart);  // 开始事件
    cudaEventCreate(&_gStop);   // 结束事件
}

/**
 * Timer 类析构函数
 *
 * 功能：
 * - 清理 CUDA 事件资源
 * - RAII 模式保证资源释放
 *
 * 注意事项：
 * - 这里使用了 cudaFree 而不是 cudaEventDestroy
 * - 这是一个 BUG，应该使用 cudaEventDestroy
 * - cudaFree 用于释放设备内存
 * - cudaEventDestroy 用于销毁事件
 *
 * 正确的做法应该是：
 * cudaEventDestroy(_gStart);
 * cudaEventDestroy(_gStop);
 */
Timer::~Timer(){
    cudaFree(_gStart);  // 错误：应该使用 cudaEventDestroy
    cudaFree(_gStop);   // 错误：应该使用 cudaEventDestroy
}

/**
 * 启动 GPU 计时
 *
 * 功能：
 * - 在当前 CUDA 流中记录开始事件
 * - 异步操作，立即返回
 *
 * cudaEventRecord 说明：
 * - 第一个参数：事件对象
 * - 第二个参数：流 ID（0 表示默认流）
 * - 事件会在流中所有之前的操作完成后触发
 */
void Timer::start_gpu() {
    cudaEventRecord(_gStart, 0);  // 在默认流中记录开始事件
}

/**
 * 停止 GPU 计时
 *
 * 功能：
 * - 在当前 CUDA 流中记录结束事件
 * - 异步操作，立即返回
 *
 * 注意：
 * - 仅记录事件，不计算时间
 * - 时间计算在 duration_gpu() 中进行
 */
void Timer::stop_gpu() {
    cudaEventRecord(_gStop, 0);  // 在默认流中记录结束事件
}

/**
 * 启动 CPU 计时
 *
 * 功能：
 * - 记录当前高精度时钟时间点
 * - 立即执行，同步操作
 *
 * std::chrono 说明：
 * - high_resolution_clock: 最高精度时钟
 * - now(): 获取当前时间点
 * - 精度通常在纳秒级别
 */
void Timer::start_cpu() {
    _cStart = std::chrono::high_resolution_clock::now();
}

/**
 * 停止 CPU 计时
 *
 * 功能：
 * - 记录结束时间点
 * - 立即执行，同步操作
 */
void Timer::stop_cpu() {
    _cStop = std::chrono::high_resolution_clock::now();
}

/**
 * 计算并输出 GPU 执行时间
 *
 * @param msg: 输出消息描述
 *
 * 功能：
 * - 同步等待 GPU 事件
 * - 计算两个事件间的时间差
 * - 格式化输出结果
 *
 * 实现步骤：
 * 1. cudaEventSynchronize: 等待事件完成
 * 2. cudaEventElapsedTime: 计算时间差（毫秒）
 * 3. LOG: 格式化输出
 *
 * 注意事项：
 * - 必须同步两个事件
 * - 时间单位总是毫秒
 * - 输出格式：左对60字符，6位小数
 */
void Timer::duration_gpu(std::string msg){
    // 同步等待事件完成
    CUDA_CHECK(cudaEventSynchronize(_gStart));
    CUDA_CHECK(cudaEventSynchronize(_gStop));

    // 计算时间差（毫秒）
    cudaEventElapsedTime(&_timeElasped, _gStart, _gStop);

    // 格式化输出
    LOG("%-60s uses %.6lf ms", msg.c_str(), _timeElasped);
}

/**
 * 学习要点总结：
 *
 * 1. 性能测量的重要性：
 *    - 优化前后的对比
 *    - 瓶颈识别
 *    - 性能回归检测
 *    - 基准测试
 *
 * 2. CPU 计时 vs GPU 计时：
 *    CPU 计时：
 *    - 同步操作
 *    - 系统时钟基础
 *    - 受系统调度影响
 *
 *    GPU 计时：
 *    - 异步操作
 *    - GPU 硬件计时器
 *    - 不受 CPU 影响
 *
 * 3. CUDA 事件机制：
 *    - 轻量级
 *    - 高精度
 *    - 与流关联
 *    - 支持异步
 *
 * 4. 实现中的 BUG：
 *    - 析构函数使用 cudaFree 释放事件
 *    - 应该使用 cudaEventDestroy
 *    - 可能导致资源泄漏或错误
 *
 * 5. 最佳实践：
 *    - 预热运行
 *    - 多次测量取平均
 *    - 排除初始化开销
 *    - 考虑数据传输时间
 *
 * 6. 扩展思路：
 *    - 支持多流计时
 *    - 统计分析功能
 *    - 可视化输出
 *    - 性能剖析集成
 */
