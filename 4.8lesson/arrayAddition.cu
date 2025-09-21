/**
 * 课程 4.8: CPU vs GPU 性能对比详解
 * 文件名: arrayAddition.cu
 * 作者: 权 双
 * 日期: 2023-12-31
 * 功能: 数组相加的 CPU 和 GPU 实现对比
 *
 * 本程序展示了 CUDA 编程的完整流程：
 * 1. CPU 和 GPU 两种实现方式的对比
 * 2. 结果正确性验证
 * 3. 性能分析和比较
 * 4. 内存管理最佳实践
 * 5. 错误检查和调试技巧
 *
 * 学习目标：
 * - 理解 CPU 和 GPU 计算模式的差异
 * - 掌握 CUDA 程序的标准开发流程
 * - 学习性能测试和结果验证方法
 * - 体验并行计算的优势和挑战
 */

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

/**
 * 数据初始化函数 - 生成测试数据
 *
 * @param addr: 要初始化的数组地址
 * @param elemCount: 数组元素数量
 *
 * 数据生成策略：
 * - 使用位运算限制随机数范围
 * - 生成 0.0-25.5 范围的浮点数
 * - 确保数据的可重现性（通过固定随机种子）
 */
void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        /**
         * 随机数生成解析：
         * - rand(): 生成随机整数
         * - & 0xFF: 位运算，限制为 0-255 范围
         * - / 10.f: 转换为 0.0-25.5 的浮点数
         *
         * 这种方法的优点：
         * - 数值范围可控，避免溢出
         * - 计算结果便于验证
         * - 生成速度快
         */
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

/**
 * GPU 内核函数 - 并行数组加法
 *
 * @param A: 输入数组 A (设备内存)
 * @param B: 输入数组 B (设备内存)
 * @param C: 输出数组 C (设备内存)
 * @param N: 数组元素数量
 *
 * 并行策略：
 * - 每个线程处理一个数组元素
 * - 线程间完全独立，无需同步
 * - 利用 GPU 的大规模并行能力
 */
__global__ void addFromGPU(float *A, float *B, float *C, const int N)
{
    /**
     * 计算当前线程的全局索引
     * 这是 CUDA 编程的核心模式
     */
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    /**
     * 边界检查 - 防止数组越界
     * 当线程数不是数组大小的整数倍时必须进行边界检查
     */
    if (id >= N) return;

    /**
     * 核心计算：并行数组加法
     * 每个线程独立计算一个元素
     * 这里直接使用加法运算符，也可以调用设备函数
     */
    C[id] = A[id] + B[id];
}

/**
 * CPU 函数 - 顺序数组加法
 *
 * @param A: 输入数组 A
 * @param B: 输入数组 B
 * @param C: 输出数组 C
 * @param N: 数组元素数量
 *
 * 顺序策略：
 * - 单线程依次处理每个元素
 * - 内存访问模式友好（顺序访问）
 * - 作为 GPU 版本正确性验证的参考
 */
void addFromCPU(float *A, float *B, float *C, const int N)
{
    /**
     * 简单的顺序循环处理
     * 时间复杂度 O(N)，但只使用单个 CPU 核心
     */
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
    return;
}

/**
 * 结果验证函数 - 比较 CPU 和 GPU 计算结果
 *
 * @param hostRef: CPU 计算结果（参考答案）
 * @param gpuRef: GPU 计算结果（待验证）
 * @param N: 数组元素数量
 *
 * 验证策略：
 * - 使用浮点数容差比较
 * - 发现不匹配时立即停止
 * - 提供详细的错误信息
 */
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    /**
     * 浮点数比较容差
     * 由于浮点运算的精度限制，不能直接使用 == 比较
     */
    double epsilon = 1.0E-8;
    bool match = 1;

    /**
     * 逐个元素比较
     */
    for (int i = 0; i < N; i++)
    {
        /**
         * 浮点数容差比较
         * abs() 计算绝对值差
         * 超过容差则认为不匹配
         */
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;  // 发现错误立即退出
        }
    }

    /**
     * 输出验证结果
     */
    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

/**
 * 主函数 - CPU vs GPU 性能对比的完整实现
 */
int main(int argc, char **argv)
{
    /**
     * 第一步：GPU 设备初始化和信息查询
     */
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备: " << deviceProps.name << std::endl;
    std::cout << "计算能力: " << deviceProps.major << "." << deviceProps.minor << std::endl;
    std::cout << "全局内存: " << deviceProps.totalGlobalMem / (1024*1024) << " MB" << std::endl;

    /**
     * 第二步：配置测试参数
     */
    int iElemCount = 2048;                              // 数组大小（元素数量）
    size_t stBytesCount = iElemCount * sizeof(float);   // 内存大小（字节数）

    std::cout << "\n测试配置:" << std::endl;
    std::cout << "数组大小: " << iElemCount << " 个元素" << std::endl;
    std::cout << "内存使用: " << stBytesCount * 4 / 1024 << " KB (4个数组)" << std::endl;

    /**
     * 第三步：分配主机内存
     *
     * 内存用途说明：
     * - fpHost_A, fpHost_B: 输入数据
     * - fpHost_C: CPU 计算结果
     * - fpDeviceRef: GPU 计算结果（从设备复制回来的）
     */
    float *fpHost_A = nullptr;
    float *fpHost_B = nullptr;
    float *fpHost_C = nullptr;      // CPU 结果
    float *fpDeviceRef = nullptr;   // GPU 结果

    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);
    fpDeviceRef = (float *)malloc(stBytesCount);

    /**
     * 内存分配检查
     */
    if (!fpHost_A || !fpHost_B || !fpHost_C || !fpDeviceRef) {
        std::cout << "主机内存分配失败!" << std::endl;
        exit(-1);
    }

    /**
     * 设置随机种子并初始化数据
     */
    srand(666);  // 固定种子确保结果可重现
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);
    memset(fpHost_C, 0, stBytesCount);      // 初始化为 0
    memset(fpDeviceRef, 0, stBytesCount);   // 初始化为 0

    std::cout << "\n数据初始化完成" << std::endl;

    /**
     * 第四步：分配设备内存
     */
    float *fpDevice_A = nullptr;
    float *fpDevice_B = nullptr;
    float *fpDevice_C = nullptr;

    CUDA_CHECK(cudaMalloc((float**)&fpDevice_A, stBytesCount));
    CUDA_CHECK(cudaMalloc((float**)&fpDevice_B, stBytesCount));
    CUDA_CHECK(cudaMalloc((float**)&fpDevice_C, stBytesCount));

    /**
     * 将输入数据从主机传输到设备
     */
    CUDA_CHECK(cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(fpDevice_C, 0, stBytesCount));  // 设备端结果数组初始化

    std::cout << "设备内存分配和数据传输完成" << std::endl;

    /**
     * 第五步：CPU 计算
     */
    std::cout << "\n开始 CPU 计算..." << std::endl;
    addFromCPU(fpHost_A, fpHost_B, fpHost_C, iElemCount);
    std::cout << "CPU 计算完成" << std::endl;

    /**
     * 第六步：GPU 计算
     */
    std::cout << "\n开始 GPU 计算..." << std::endl;

    /**
     * 配置内核启动参数
     *
     * 配置策略：
     * - 线程块大小：64（平衡占用率和资源使用）
     * - 网格大小：向上取整确保覆盖所有元素
     */
    dim3 block(64);
    dim3 grid((iElemCount + block.x - 1) / 64);

    std::cout << "内核配置: " << grid.x << " 个线程块，每块 " << block.x << " 个线程" << std::endl;
    std::cout << "总线程数: " << grid.x * block.x << std::endl;

    /**
     * 启动 GPU 内核
     */
    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);

    /**
     * 等待 GPU 计算完成
     */
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "GPU 计算完成" << std::endl;

    /**
     * 第七步：将 GPU 结果传回主机
     */
    CUDA_CHECK(cudaMemcpy(fpDeviceRef, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost));

    /**
     * 第八步：验证计算结果
     */
    std::cout << "\n验证计算结果..." << std::endl;
    checkResult(fpHost_C, fpDeviceRef, iElemCount);

    /**
     * 第九步：显示部分结果
     */
    std::cout << "前 10 个元素的计算结果:" << std::endl;
    std::cout << "索引\tA\t\tB\t\t结果" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        printf("%2d\t%.2f\t\t%.2f\t\t%.2f\n",
               i+1, fpHost_A[i], fpHost_B[i], fpDeviceRef[i]);
    }

    /**
     * 第十步：性能分析总结
     */
    std::cout << "\n性能分析总结:" << std::endl;
    std::cout << "1. CPU 计算：单线程顺序处理" << std::endl;
    std::cout << "   - 优点：内存访问模式简单，缓存友好" << std::endl;
    std::cout << "   - 缺点：无法利用并行能力" << std::endl;

    std::cout << "2. GPU 计算：" << grid.x * block.x << " 个线程并行处理" << std::endl;
    std::cout << "   - 优点：大规模并行，理论上快" << grid.x * block.x << "倍" << std::endl;
    std::cout << "   - 缺点：数据传输开销，适合计算密集型任务" << std::endl;

    std::cout << "3. 适用场景分析:" << std::endl;
    std::cout << "   - 小数据集：CPU 可能更快（避免传输开销）" << std::endl;
    std::cout << "   - 大数据集：GPU 优势明显" << std::endl;
    std::cout << "   - 重复计算：GPU 传输开销可摊销" << std::endl;

    /**
     * 第十一步：清理内存资源
     */
    // 释放主机内存
    free(fpDeviceRef);
    free(fpHost_C);
    free(fpHost_B);
    free(fpHost_A);

    // 释放设备内存
    CUDA_CHECK(cudaFree(fpDevice_C));
    CUDA_CHECK(cudaFree(fpDevice_B));
    CUDA_CHECK(cudaFree(fpDevice_A));

    /**
     * 重置 CUDA 设备
     */
    CUDA_CHECK(cudaDeviceReset());

    std::cout << "\n程序执行完成，资源清理完毕" << std::endl;
    return 0;
}

/**
 * 学习要点总结：
 *
 * 1. CUDA 编程的标准流程：
 *    a) 设备初始化和查询
 *    b) 主机内存分配和数据准备
 *    c) 设备内存分配
 *    d) 数据传输（主机 -> 设备）
 *    e) 内核启动和执行
 *    f) 数据传输（设备 -> 主机）
 *    g) 结果验证和分析
 *    h) 资源清理
 *
 * 2. CPU vs GPU 计算模式对比：
 *    CPU：
 *    - 单线程或少量线程
 *    - 复杂控制逻辑能力强
 *    - 缓存层次复杂
 *    - 适合串行算法
 *
 *    GPU：
 *    - 大量简单线程
 *    - 适合数据并行
 *    - 高内存带宽
 *    - 需要算法并行化
 *
 * 3. 性能考虑因素：
 *    - 数据传输开销
 *    - 计算复杂度
 *    - 内存访问模式
 *    - 线程配置优化
 *
 * 4. 正确性验证：
 *    - 浮点数容差比较
 *    - 边界条件测试
 *    - 多种输入数据验证
 *    - 性能和正确性平衡
 *
 * 5. 内存管理最佳实践：
 *    - 及时检查分配失败
 *    - 对称的分配和释放
 *    - 错误检查每个 CUDA API
 *    - 资源清理的完整性
 *
 * 6. 调试技巧：
 *    - 使用错误检查宏
 *    - 分步验证中间结果
 *    - 比较已知正确的实现
 *    - 使用适当的测试数据
 *
 * 7. 优化方向：
 *    - 增加计算复杂度（如矩阵乘法）
 *    - 优化内存访问模式
 *    - 使用共享内存
 *    - 实现重叠计算和传输
 *
 * 8. 实际应用价值：
 *    - 为复杂 CUDA 项目提供框架
 *    - 性能测试的标准方法
 *    - 算法并行化的基础
 *    - 教学和学习的完整示例
 */