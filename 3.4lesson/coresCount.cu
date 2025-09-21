/**
 * 课程 3.4: CUDA 核心数量计算详解
 * 文件名: coresCount.cu
 * 作者: 权 双
 * 日期: 2023-08-13
 * 功能: 根据 GPU 架构和计算能力精确计算 CUDA 核心数量
 *
 * 本程序演示：
 * 1. 不同 GPU 架构的 CUDA 核心配置规律
 * 2. 如何根据计算能力版本确定核心数量
 * 3. 各代 GPU 架构的演进历史和特点
 * 4. 准确的硬件规格查询方法
 *
 * 支持的 GPU 架构：
 * - Fermi (2.x): 第一代统一架构
 * - Kepler (3.x): 能效优化架构
 * - Maxwell (5.x): 移动优化架构
 * - Pascal (6.x): FinFET 工艺架构
 * - Volta/Turing (7.x): Tensor 核心架构
 * - Ampere (8.x): 第二代 RT 核心架构
 * - Hopper (9.x): 数据中心专用架构
 *
 * 学习目标：
 * - 了解 GPU 架构演进历史
 * - 掌握 CUDA 核心数量的计算方法
 * - 理解不同架构的设计思路
 * - 为性能分析提供准确的硬件参数
 */

#include <stdio.h>
#include "../tools/common.cuh"

/**
 * 计算 GPU 的 CUDA 核心数量
 *
 * @param devProp: GPU 设备属性结构体
 * @return: CUDA 核心总数
 *
 * 计算原理：
 * CUDA 核心总数 = SM 数量 × 每个 SM 的核心数
 * 每个 SM 的核心数因架构而异，需要根据计算能力版本确定
 *
 * 注意事项：
 * - 不同架构的 SM 设计差异很大
 * - 相同主版本号可能有不同的核心配置
 * - 新架构会引入新的计算单元（如 Tensor 核心）
 * - 此函数只计算传统的 CUDA 核心，不包括特殊计算单元
 */
int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;  // CUDA 核心总数
    int mp = devProp.multiProcessorCount;  // SM 数量

    /**
     * 根据计算能力主版本号确定架构和核心配置
     * 每个 case 对应一代 GPU 架构
     */
    switch (devProp.major) {

    /**
     * Fermi 架构 (2010-2012)
     * 计算能力: 2.0, 2.1
     * 特点: 第一代统一架构，引入缓存层次
     */
    case 2: // Fermi 架构
        if (devProp.minor == 1) {
            // Fermi 2.1 (GeForce GTX 400/500 系列高端)
            cores = mp * 48;  // 每个 SM 48 个核心
        } else {
            // Fermi 2.0 (GeForce GTX 400/500 系列主流)
            cores = mp * 32;  // 每个 SM 32 个核心
        }
        printf("架构: Fermi (费米) - 第一代统一架构\n");
        break;

    /**
     * Kepler 架构 (2012-2014)
     * 计算能力: 3.0, 3.2, 3.5, 3.7
     * 特点: 能效优化，动态并行，更多核心
     */
    case 3: // Kepler 架构
        cores = mp * 192;  // 每个 SM 192 个核心
        printf("架构: Kepler (开普勒) - 能效优化架构\n");
        break;

    /**
     * Maxwell 架构 (2014-2016)
     * 计算能力: 5.0, 5.2, 5.3
     * 特点: 移动端优化，第一代 Maxwell 为 5.0，第二代为 5.2/5.3
     */
    case 5: // Maxwell 架构
        cores = mp * 128;  // 每个 SM 128 个核心
        printf("架构: Maxwell (麦克斯韦) - 移动优化架构\n");
        break;

    /**
     * Pascal 架构 (2016-2017)
     * 计算能力: 6.0, 6.1, 6.2
     * 特点: FinFET 工艺，统一内存，NVLink
     */
    case 6: // Pascal 架构
        if ((devProp.minor == 1) || (devProp.minor == 2)) {
            // Pascal 6.1 (GeForce GTX 10 系列)
            // Pascal 6.2 (嵌入式平台)
            cores = mp * 128;  // 每个 SM 128 个核心
        } else if (devProp.minor == 0) {
            // Pascal 6.0 (Tesla P100)
            cores = mp * 64;   // 每个 SM 64 个核心 (双精度优化)
        } else {
            printf("未知的 Pascal 子版本: %d.%d\n", devProp.major, devProp.minor);
        }
        printf("架构: Pascal (帕斯卡) - FinFET 工艺架构\n");
        break;

    /**
     * Volta/Turing 架构 (2017-2020)
     * 计算能力: 7.0, 7.2, 7.5
     * 特点: Tensor 核心，RT 核心(Turing)，混合精度计算
     */
    case 7: // Volta 和 Turing 架构
        if ((devProp.minor == 0) || (devProp.minor == 2)) {
            // Volta 7.0 (Tesla V100)
            // Volta 7.2 (Jetson AGX Xavier)
            cores = mp * 64;   // 每个 SM 64 个 CUDA 核心
            printf("架构: Volta (伏特) - Tensor 核心架构\n");
        } else if (devProp.minor == 5) {
            // Turing 7.5 (GeForce RTX 20 系列, GTX 16 系列)
            cores = mp * 64;   // 每个 SM 64 个 CUDA 核心
            printf("架构: Turing (图灵) - RT 核心 + Tensor 核心架构\n");
        } else {
            printf("未知的 Volta/Turing 子版本: %d.%d\n", devProp.major, devProp.minor);
        }
        break;

    /**
     * Ampere 架构 (2020-2022)
     * 计算能力: 8.0, 8.6, 8.7, 8.9
     * 特点: 第二代 RT 核心，第三代 Tensor 核心，稀疏性支持
     */
    case 8: // Ampere 架构
        if (devProp.minor == 0) {
            // Ampere 8.0 (A100, A30)
            cores = mp * 64;   // 每个 SM 64 个核心
            printf("架构: Ampere (安培) - 数据中心专用架构\n");
        } else if (devProp.minor == 6) {
            // Ampere 8.6 (GeForce RTX 30 系列)
            cores = mp * 128;  // 每个 SM 128 个核心
            printf("架构: Ampere (安培) - 游戏优化架构\n");
        } else if (devProp.minor == 7) {
            // Ampere 8.7 (Jetson AGX Orin)
            cores = mp * 128;  // 每个 SM 128 个核心
            printf("架构: Ampere (安培) - 嵌入式平台架构\n");
        } else if (devProp.minor == 9) {
            // Ada Lovelace 8.9 (GeForce RTX 40 系列)
            cores = mp * 128;  // 每个 SM 128 个核心
            printf("架构: Ada Lovelace (艾达·洛夫莱斯) - 下一代游戏架构\n");
        } else {
            printf("未知的 Ampere 子版本: %d.%d\n", devProp.major, devProp.minor);
        }
        break;

    /**
     * Hopper 架构 (2022-)
     * 计算能力: 9.0
     * 特点: 数据中心专用，第四代 Tensor 核心，Transformer 引擎
     */
    case 9: // Hopper 架构
        if (devProp.minor == 0) {
            // Hopper 9.0 (H100, H800)
            cores = mp * 128;  // 每个 SM 128 个核心
            printf("架构: Hopper (霍普) - AI 专用数据中心架构\n");
        } else {
            printf("未知的 Hopper 子版本: %d.%d\n", devProp.major, devProp.minor);
        }
        break;

    /**
     * 未知架构处理
     * 用于处理新发布的或不支持的架构
     */
    default:
        printf("未知的 GPU 架构: 计算能力 %d.%d\n", devProp.major, devProp.minor);
        printf("请更新程序以支持此架构\n");
        break;
    }

    return cores;
}

/**
 * 主函数 - GPU 核心数量查询演示
 */
int main()
{
    /**
     * 设置要查询的设备
     */
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    /**
     * 获取设备属性
     */
    cudaDeviceProp prop;
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);

    /**
     * 显示设备基本信息
     */
    printf("=== GPU CUDA 核心数量查询 ===\n");
    printf("设备名称: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("SM 数量: %d\n", prop.multiProcessorCount);

    /**
     * 计算并显示 CUDA 核心数量
     */
    int totalCores = getSPcores(prop);
    if (totalCores > 0) {
        printf("CUDA 核心总数: %d\n", totalCores);
        printf("每个 SM 的核心数: %d\n", totalCores / prop.multiProcessorCount);

        /**
         * 性能分析提示
         */
        printf("\n=== 性能分析 ===\n");
        if (totalCores >= 4096) {
            printf("高性能 GPU: 适合大规模并行计算和深度学习\n");
        } else if (totalCores >= 2048) {
            printf("中等性能 GPU: 适合中等规模并行计算\n");
        } else if (totalCores >= 1024) {
            printf("入门级 GPU: 适合学习和轻量级并行计算\n");
        } else {
            printf("低端 GPU: 适合 CUDA 编程学习\n");
        }

        /**
         * 架构特点说明
         */
        printf("\n=== 架构特点 ===\n");
        switch (prop.major) {
        case 2:
            printf("- 第一代统一架构\n- 支持 C++ 异常处理\n- 引入 L1/L2 缓存\n");
            break;
        case 3:
            printf("- 动态并行支持\n- Hyper-Q 技术\n- 更高的能效比\n");
            break;
        case 5:
            printf("- 统一内存支持\n- 移动端优化\n- 更好的能效控制\n");
            break;
        case 6:
            printf("- FinFET 16nm 工艺\n- NVLink 高速互连\n- 统一内存优化\n");
            break;
        case 7:
            printf("- Tensor 核心（AI 加速）\n- 混合精度计算\n- Volta: 数据中心专用\n- Turing: RT 核心（光线追踪）\n");
            break;
        case 8:
            printf("- 第二代 RT 核心\n- 第三代 Tensor 核心\n- 稀疏性计算支持\n- AV1 编码支持\n");
            break;
        case 9:
            printf("- 第四代 Tensor 核心\n- Transformer 引擎\n- FP8 精度支持\n- 数据中心 AI 专用\n");
            break;
        }
    } else {
        printf("无法确定 CUDA 核心数量\n");
    }

    printf("\n查询完成\n");
    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. GPU 架构演进：
 *    时间线：
 *    - Fermi (2010): 统一架构奠基
 *    - Kepler (2012): 能效优化
 *    - Maxwell (2014): 移动优化
 *    - Pascal (2016): 工艺升级
 *    - Volta (2017): AI 计算
 *    - Turing (2018): 光线追踪
 *    - Ampere (2020): 综合提升
 *    - Hopper (2022): AI 专用
 *
 * 2. 核心数量计算：
 *    公式: 总核心数 = SM 数量 × 每 SM 核心数
 *    变化规律:
 *    - 早期架构: 核心数较少但功能强
 *    - 中期架构: 核心数大幅增加
 *    - 近期架构: 平衡核心数与专用单元
 *
 * 3. 架构特点对比：
 *    计算重点：
 *    - Fermi/Kepler: 通用并行计算
 *    - Maxwell/Pascal: 能效和统一内存
 *    - Volta/Turing: AI 和光线追踪
 *    - Ampere/Hopper: AI 和数据中心
 *
 * 4. 性能分析指导：
 *    - 核心数量影响并行度上限
 *    - 架构特性影响优化策略
 *    - 专用单元提供特殊加速
 *    - 新架构通常有新的编程模型
 *
 * 5. 实际应用价值：
 *    - 选择合适的 GPU 型号
 *    - 设计适合的算法策略
 *    - 预测程序性能表现
 *    - 进行成本效益分析
 *
 * 6. 注意事项：
 *    - CUDA 核心只是性能的一个指标
 *    - 内存带宽同样重要
 *    - 专用单元可能更适合特定任务
 *    - 软件优化比硬件规格更重要
 */