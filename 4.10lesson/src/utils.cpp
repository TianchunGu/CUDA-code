/**
 * 课程 4.10: 高级 CUDA 项目结构 - 工具函数实现
 * 文件名: utils.cpp
 * 作者: 权 双
 * 日期: 2023-12-31
 * 功能: 通用工具函数的实现
 *
 * 提供的功能：
 * 1. 数组初始化
 * 2. 数组打印
 * 3. 结果比较验证
 */

#include "utils.hpp"
#include <math.h>
#include <random>

/**
 * 矩阵/数组初始化函数
 *
 * @param data: 待初始化的数组指针
 * @param size: 数组元素个数
 * @param seed: 随机数种子
 *
 * 功能：
 * - 使用随机数初始化数组
 * - 值范围：[0, 1] 的浮点数
 * - 固定种子确保可重现性
 *
 * 实现说明：
 * - srand() 设置随机种子
 * - rand() 生成 [0, RAND_MAX] 的整数
 * - 除以 RAND_MAX 归一化到 [0, 1]
 */
void initMatrix(float* data, int size, int seed)
{
    srand(seed);  // 设置随机种子
    for (int i = 0; i < size; i ++)
    {
        data[i] = float(rand()) / RAND_MAX;  // 归一化到 [0, 1]
    }
}

/**
 * 矩阵/数组打印函数
 *
 * @param data: 要打印的数组指针
 * @param size: 数组元素个数
 *
 * 功能：
 * - 格式化输出数组内容
 * - 8位小数精度
 * - 逗号分隔
 *
 * 用途：
 * - 调试输出
 * - 结果验证
 * - 中间状态检查
 */
void printMat(float* data, int size)
{
    for (int i = 0; i < size; i ++) {
        printf("%.8lf", data[i]);    // 8位小数精度
        if (i != size - 1) {
            printf(", ");            // 逗号分隔
        } else {
            printf("\n");            // 最后换行
        }
    }
}

/**
 * CPU 和 GPU 结果比较函数
 *
 * @param h_data: CPU 计算结果（参考答案）
 * @param d_data: GPU 计算结果（待验证）
 * @param size: 数组大小
 *
 * 功能：
 * - 逐元素比较两个数组
 * - 使用浮点数容差判断
 * - 发现不匹配时报告位置
 * - 显示具体差异值
 *
 * 容差设置：
 * - 1.0E-4：适合大多数单精度计算
 * - 可根据具体需求调整
 *
 * 错误报告：
 * - 显示不匹配的具体值
 * - 计算并显示坐标位置
 * - 方便定位问题
 */
void compareMat(float* h_data, float* d_data, int size)
{
    double precision = 1.0E-4;  // 浮点数容差

    for (int i = 0; i < size; i ++) {
        // 使用绝对值差判断
        if (abs(h_data[i] - d_data[i]) > precision) {
            // 计算二维坐标（假设方阵）
            int y = i / size;  // 行坐标
            int x = i % size;  // 列坐标

            // 输出错误信息
            printf("矩阵结果不匹配\n");
            printf("CPU: %.8lf, GPU: %.8lf, 位置:[%d, %d]\n",
                   h_data[i], d_data[i], x, y);
            break;  // 发现第一个错误即停止
        }
    }
}

/**
 * 学习要点总结：
 *
 * 1. 浮点数比较：
 *    - 不能直接使用 ==
 *    - 必须使用容差比较
 *    - 容差需根据精度要求设定
 *
 * 2. 随机数生成：
 *    - 固定种子确保可重现
 *    - 归一化到特定范围
 *    - 考虑使用更好的随机数生成器
 *
 * 3. 调试输出：
 *    - 适当的精度显示
 *    - 清晰的格式
 *    - 有助于问题定位
 *
 * 4. 验证策略：
 *    - 快速失败（发现错误立即停止）
 *    - 详细信息（位置、值等）
 *    - 友好的错误提示
 */
