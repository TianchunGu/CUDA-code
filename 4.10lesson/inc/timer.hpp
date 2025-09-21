/**
 * 课程 4.10: 高级 CUDA 项目结构 - 性能计时器头文件
 * 文件名: timer.hpp
 * 作者: 权 双
 * 日期: 2023-12-31
 * 功能: CPU 和 GPU 高精度性能计时
 *
 * Timer 类特点：
 * 1. 支持 CPU 和 GPU 两种计时模式
 * 2. CPU 使用 chrono 高精度时钟
 * 3. GPU 使用 CUDA 事件计时
 * 4. 支持多种时间单位（秒、毫秒、微秒、纳秒）
 * 5. RAII 设计模式，自动资源管理
 *
 * 使用示例：
 * Timer timer;
 * timer.start_cpu();
 * // ... 执行 CPU 代码 ...
 * timer.stop_cpu();
 * timer.duration_cpu<Timer::ms>("代码执行完成");
 */

#include <chrono>
#include <ratio>
#include <string>
#include "cuda_runtime.h"


/**
 * Timer 类 - 高精度性能计时器
 *
 * 功能：
 * - 提供 CPU 和 GPU 代码的性能测量
 * - 支持灵活的时间单位选择
 * - 自动管理 CUDA 事件资源
 * - 格式化输出性能测量结果
 */
class Timer {
public:
    /**
     * 时间单位类型定义
     * 使用 std::ratio 模板定义不同精度
     *
     * s:  秒 (1/1)
     * ms: 毫秒 (1/1000)
     * us: 微秒 (1/1000000)
     * ns: 纳秒 (1/1000000000)
     *
     * 这些类型用于模板参数，确定输出时间单位
     */
    using s  = std::ratio<1, 1>;           // 秒
    using ms = std::ratio<1, 1000>;        // 毫秒
    using us = std::ratio<1, 1000000>;     // 微秒
    using ns = std::ratio<1, 1000000000>;  // 纳秒

public:
    /**
     * 构造函数
     * 初始化计时器，创建 CUDA 事件
     */
    Timer();

    /**
     * 析构函数
     * 清理资源，销毁 CUDA 事件
     */
    ~Timer();

public:
    /**
     * 启动 CPU 计时
     * 记录当前高精度时钟时间点
     */
    void start_cpu();

    /**
     * 启动 GPU 计时
     * 在当前 CUDA 流中记录开始事件
     */
    void start_gpu();

    /**
     * 停止 CPU 计时
     * 记录结束时间点
     */
    void stop_cpu();

    /**
     * 停止 GPU 计时
     * 在当前 CUDA 流中记录结束事件
     */
    void stop_gpu();

    /**
     * 计算并输出 CPU 执行时间
     *
     * @tparam span: 时间单位类型（s/ms/us/ns）
     * @param msg: 输出消息描述
     *
     * 模板函数，根据模板参数选择输出单位
     * 使用示例： timer.duration_cpu<Timer::ms>("处理完成");
     */
    template <typename span>
    void duration_cpu(std::string msg);

    /**
     * 计算并输出 GPU 执行时间
     *
     * @param msg: 输出消息描述
     *
     * GPU 时间始终以毫秒为单位
     * CUDA 事件计时的精度通常在微秒级别
     */
    void duration_gpu(std::string msg);

private:
    /**
     * CPU 计时成员变量
     * 使用 chrono 高精度时钟
     */
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStart;  // 开始时间点
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;   // 结束时间点

    /**
     * GPU 计时成员变量
     * 使用 CUDA 事件计时
     */
    cudaEvent_t _gStart;        // GPU 开始事件
    cudaEvent_t _gStop;         // GPU 结束事件
    float _timeElasped;         // GPU 执行时间（毫秒）
};

/**
 * CPU 时间计算和输出的模板函数实现
 *
 * 模板实现必须在头文件中，以便编译器实例化
 *
 * 实现步骤：
 * 1. 根据模板参数确定时间单位字符串
 * 2. 计算时间差并转换为指定单位
 * 3. 格式化输出结果
 *
 * 使用 std::is_same 判断模板参数类型
 */
template <typename span>
void Timer::duration_cpu(std::string msg){
    std::string str;  // 时间单位字符串

    // 根据模板参数选择单位字符串
    if(std::is_same<span, s>::value) { str = "s"; }        // 秒
    else if(std::is_same<span, ms>::value) { str = "ms"; }  // 毫秒
    else if(std::is_same<span, us>::value) { str = "us"; }  // 微秒
    else if(std::is_same<span, ns>::value) { str = "ns"; }  // 纳秒

    // 计算时间差并转换为指定单位
    std::chrono::duration<double, span> time = _cStop - _cStart;

    // 格式化输出：消息(左对40字符) + 时间(6位小数) + 单位
    LOG("%-40s uses %.6lf %s", msg.c_str(), time.count(), str.c_str());
}

/**
 * 学习要点总结：
 *
 * 1. 高精度计时技术：
 *    - CPU: std::chrono 高精度时钟
 *    - GPU: CUDA 事件计时
 *    - 不同平台需要不同方法
 *
 * 2. 模板元编程应用：
 *    - std::ratio 定义时间单位
 *    - std::is_same 判断类型
 *    - 编译时确定时间单位
 *
 * 3. CUDA 事件特点：
 *    - 异步记录
 *    - 需要同步等待
 *    - 精度高，开销小
 *
 * 4. RAII 设计模式：
 *    - 构造函数创建资源
 *    - 析构函数释放资源
 *    - 避免资源泄漏
 */